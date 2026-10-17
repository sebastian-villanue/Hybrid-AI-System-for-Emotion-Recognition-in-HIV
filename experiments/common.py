"""Shared helpers for the experiments in this folder.

Run every script from the repository root (the folder that contains config.py and notebooks/).
"""
import json, os, re
from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

TARGET_EMOTIONS = ["anger", "disgust", "embarrassment", "fear", "grief", "nervousness", "sadness"]
T8 = TARGET_EMOTIONS + ["neutral"]
RESULTS_800 = "results/evaluation_results_800.json"


def load_prompt_builder(nb_path="notebooks/01_train_validate_test.ipynb"):
    """Return the exact build_llm_prompt() used in the paper, read from notebook 01,
    so the refinement prompt is byte-identical to the one behind Tables 3-5."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "def build_llm_prompt" in "".join(c["source"]))
    start = src.index("def build_llm_prompt")
    end = src.index("def call_llm", start)
    ns = {"TARGET_EMOTIONS": TARGET_EMOTIONS}
    exec(src[start:end], ns)
    return ns["build_llm_prompt"]


def load_test_records():
    """The 800 test messages with gold labels, backbone probabilities from the released
    checkpoint, and a domain tag. Within each 100-message emotion block the first 50 are
    HIV-contextualized and the last 50 general-domain (the layout of the released file)."""
    recs = json.load(open(RESULTS_800, encoding="utf-8"))
    assert len(recs) == 800, len(recs)
    for i, r in enumerate(recs):
        r["domain"] = "hiv" if (i % 100) < 50 else "general"
    blocks = [Counter(r["gold_label"] for r in recs[b:b + 100]) for b in range(0, 800, 100)]
    assert all(len(c) == 1 for c in blocks), "unexpected ordering of the results file"
    return recs


def parse_label(text):
    """Pull final_label out of a model reply; tolerant to code fences, <think> blocks and prose."""
    if text is None:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            lab = json.loads(m.group(0)).get("final_label")
            if isinstance(lab, str) and lab.strip().lower() in T8:
                return lab.strip().lower()
        except Exception:
            pass
    m = re.search(r'"final_label"\s*:\s*"([a-z]+)"', text)
    if m and m.group(1) in T8:
        return m.group(1)
    return None


def scores(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=T8, average="macro", zero_division=0)
    acc = float(np.mean([g == q for g, q in zip(gold, pred)]))
    return {"macro_p": round(float(p), 3), "macro_r": round(float(r), 3),
            "macro_f1": round(float(f), 3), "micro_f1": round(acc, 3)}


def scores_by_domain(recs, pred_key):
    out = {"all": scores([r["gold_label"] for r in recs], [r[pred_key] for r in recs])}
    for d in ("hiv", "general"):
        sub = [r for r in recs if r["domain"] == d]
        out[d] = scores([r["gold_label"] for r in sub], [r[pred_key] for r in sub])
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
