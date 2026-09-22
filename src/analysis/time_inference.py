"""Measure what the submitted system costs to run on the 503 test instances.

Fine-tuned weights were never saved (encoder.py predicts every split inside the
training run), so each neural component is timed on its base checkpoint with the
same architecture, input view, truncation, batch size and precision as the
submitted run. Forward-pass cost does not depend on the trained values of the
weights, so the timing carries over; the predictions do not, and none are kept.

  ENC-L4     ModernBERT-large + 3 heads, level-4 view, 2,048 tok, batch 16, bf16 autocast, x6 seeds
  ENC-L1     same, level-1 view, 1,024 tok, x6 seeds
  QLoRA-ST3  Qwen2.5-7B-Instruct, 4-bit NF4 + LoRA r=16 (untrained adapter), batch 8, x3 seeds
  TF-IDF-ST2 CPU: fit on train+dev (training) and transform + predict test (inference)

Every seed is a separate full pass, as in the submitted ensemble. Model loading is
reported separately from inference. Results go to work/time_inference.json.

Usage:
    python src/analysis/time_inference.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from data import ST1, ST2, ST3, load_split, view, domain, _g

dev = "cuda"
test = load_split("test")
out = {"n_test": len(test), "gpu": torch.cuda.get_device_name(0),
       "torch": torch.__version__}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------- encoders
from transformers import AutoModel, AutoTokenizer


class MultiTask(nn.Module):
    """Same forward as encoder.py: mean-pooled backbone, three linear heads."""

    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.h1, self.h2, self.h3 = nn.Linear(h, len(ST1)), nn.Linear(h, len(ST2)), nn.Linear(h, len(ST3))

    def forward(self, input_ids, attention_mask):
        o = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (o * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.h1(pooled), self.h2(pooled), self.h3(pooled)


def time_encoder(level, maxlen, seeds, name="answerdotai/ModernBERT-large"):
    t0 = sync()
    tok = AutoTokenizer.from_pretrained(name)
    model = MultiTask(name).to(dev).eval()
    load = sync() - t0
    texts = [view(x, level) for x in test]
    passes = []
    for _ in range(seeds):
        t0 = sync()
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], truncation=True, max_length=maxlen,
                          padding=True, return_tensors="pt").to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    o1, o2, o3 = model(**enc)
                o1.float().softmax(-1).cpu(), o2.float().sigmoid().cpu(), o3.float().sigmoid().cpu()
        passes.append(sync() - t0)
    del model
    torch.cuda.empty_cache()
    return {"load_s": load, "pass_s": passes, "total_s": sum(passes)}


print("ENC-L4 ...", flush=True)
out["ENC-L4"] = time_encoder(level=4, maxlen=2048, seeds=6)
print(" ", out["ENC-L4"], flush=True)
print("ENC-L1 ...", flush=True)
out["ENC-L1"] = time_encoder(level=1, maxlen=1024, seeds=6)
print(" ", out["ENC-L1"], flush=True)

# ---------------------------------------------------------------- QLoRA-ST3
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

# RUBRIC and text_of are taken verbatim from llm_finetune.py, without running its argparse.
src = (ROOT / "src" / "models" / "llm_finetune.py").read_text()
exec(src[src.index("RUBRIC = ("):src.index("mlb = MultiLabelBinarizer")])


def time_qlora(seeds, name="Qwen/Qwen2.5-7B-Instruct", maxlen=1024):
    t0 = sync()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(ST3), problem_type="multi_label_classification",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map="cuda")
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.eval()
    load = sync() - t0
    texts = [text_of(x) for x in test]
    passes = []
    for _ in range(seeds):
        t0 = sync()
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                enc = tok(texts[i:i + 8], truncation=True, max_length=maxlen,
                          padding=True, return_tensors="pt").to(dev)
                torch.sigmoid(model(**enc).logits.float()).cpu()
        passes.append(sync() - t0)
    del model
    torch.cuda.empty_cache()
    return {"load_s": load, "pass_s": passes, "total_s": sum(passes),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}


print("QLoRA-ST3 ...", flush=True)
out["QLoRA-ST3"] = time_qlora(seeds=3)
print(" ", out["QLoRA-ST3"], flush=True)

# ---------------------------------------------------------------- TF-IDF-ST2
# texts() is taken verbatim from classical.py, without running its argparse.
_c = (ROOT / "src" / "models" / "classical.py").read_text()
exec(_c[_c.index("def texts(insts):"):_c.index("def build(")])

pool = load_split("train") + load_split("dev")
Y = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in pool])
t0 = time.perf_counter()
fp, fb, fs = texts(pool)
vp = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=300_000, sublinear_tf=True)
vb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000)
vs = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
Xtr = sp.hstack([vp.fit_transform(fp), vb.fit_transform(fb), vs.fit_transform(fs)]).tocsr()
models = [LogisticRegression(max_iter=3000, C=8, class_weight="balanced").fit(Xtr, Y[:, j])
          if Y[:, j].sum() >= 3 else None for j in range(Y.shape[1])]
fit_s = time.perf_counter() - t0
t0 = time.perf_counter()
p, b, s = texts(test)
Xte = sp.hstack([vp.transform(p), vb.transform(b), vs.transform(s)]).tocsr()
Q = np.column_stack([m.predict_proba(Xte)[:, 1] if m is not None else np.zeros(len(test))
                     for m in models])
out["TF-IDF-ST2"] = {"fit_train_dev_s": fit_s, "predict_test_s": time.perf_counter() - t0}
print(" TF-IDF-ST2", out["TF-IDF-ST2"], flush=True)

gpu = sum(out[k]["total_s"] for k in ("ENC-L4", "ENC-L1", "QLoRA-ST3"))
out["gpu_inference_total_s"] = gpu
out["gpu_load_total_s"] = sum(out[k]["load_s"] for k in ("ENC-L4", "ENC-L1", "QLoRA-ST3"))
json.dump(out, open(ROOT / "work" / "time_inference.json", "w"), indent=2)
print(f"\nGPU inference over {len(test)} test instances: {gpu / 60:.1f} min "
      f"(+ {out['gpu_load_total_s'] / 60:.1f} min loading)")
