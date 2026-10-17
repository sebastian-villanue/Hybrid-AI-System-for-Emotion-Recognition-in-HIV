"""Ablation — backbone trained WITHOUT any HIV data (GoEmotions-only control, Table 3).

Trains the 7-emotion classifier on the GoEmotions component only, with the same phase-1
recipe as notebook 01 (4 epochs, lr 2e-5, batch 16 x grad-acc 2, max_len 384), tunes
F1-oriented thresholds on the GoEmotions validation split (no HIV data anywhere), and
evaluates on the same balanced 800-message test set with the paper's standalone scoring
convention (highest-probability emotion above threshold; neutral when none crosses).

Two initializations:
  --init roberta-large                     -> no HIV data at all (strict ablation; default)
  --init models/roberta_large_dapt_hiv     -> HIV DAPT kept, HIV supervised data removed
The paper reports the first (strict) variant. Running both separates the effect of DAPT
from the effect of the HIV-contextualized training messages.

Optional --with-llm runs GPT-5-mini refinement on top of the ablated backbone (same prompt),
giving the pipeline-level ablation. Needs OPENAI_API_KEY.

Usage (repo root, GPU):
    PYTHONPATH=. python experiments/ablation_goemotions_only.py
    PYTHONPATH=. python experiments/ablation_goemotions_only.py --init models/roberta_large_dapt_hiv
    PYTHONPATH=. python experiments/ablation_goemotions_only.py --with-llm
Outputs: runs/ablation/<tag>/{metrics.json,per_message.json,thresholds.json}
"""
import argparse, copy, json, os

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed)

import sys
sys.path.insert(0, os.getcwd())   # config.py lives at the repo root
import config
from common import (TARGET_EMOTIONS, load_prompt_builder, load_test_records, parse_label,
                    save_json, scores_by_domain)

MAX_LEN = 384


def vec(lab):
    return [1.0 if e in lab.split(",") else 0.0 for e in TARGET_EMOTIONS]


def tune_thr_f1(probs, labels):
    grid = np.linspace(0.05, 0.95, 19)
    best = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for i in range(labels.shape[1]):
        f1s = [precision_recall_fscore_support(labels[:, i], (probs[:, i] >= t).astype(int),
                                               average="binary", zero_division=0)[2] for t in grid]
        best[i] = grid[int(np.argmax(f1s))]
    return best


class KeepBest(TrainerCallback):
    """Keep the weights of the best epoch (by val macro-F1 @0.5) in memory — no checkpoints on disk."""
    def __init__(self): self.best, self.state, self.epoch = -1, None, None
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        f = (metrics or {}).get("eval_macro_f1")
        if f is not None and f > self.best:
            self.best, self.epoch = f, state.epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="roberta-large")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--out", default="runs/ablation")
    a = ap.parse_args()
    set_seed(a.seed)
    tag = ("no_hiv_data" if a.init == "roberta-large" else "dapt_only") + f"_seed{a.seed}"
    out = f"{a.out}/{tag}"

    raw = load_dataset(config.HF_DATASET_REPO, token=config.get_hf_token())
    _all = raw[list(raw)[0]]   # the released dataset is one table with a "split" column
    sub = lambda split: _all.filter(lambda x: x["source"] == "goemotions" and x["split"] == split)
    go = DatasetDict({
        "train": sub("train"), "validation": sub("validation")})
    assert [len(go[s]) for s in ("train", "validation")] == [4621, 566]

    tok = AutoTokenizer.from_pretrained("roberta-large")
    def prep(ds):
        d = Dataset.from_dict({"text": ds["text"], "labels": [vec(l) for l in ds["labels"]]})
        return d.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=MAX_LEN),
                     batched=True, remove_columns=["text"])
    ds = {s: prep(go[s]) for s in go}

    cfg = AutoConfig.from_pretrained(a.init, num_labels=7, problem_type="multi_label_classification",
                                     id2label=dict(enumerate(TARGET_EMOTIONS)),
                                     label2id={e: i for i, e in enumerate(TARGET_EMOTIONS)})
    model = AutoModelForSequenceClassification.from_pretrained(a.init, config=cfg)

    class Coll(DataCollatorWithPadding):
        def __call__(self, feats):
            b = super().__call__(feats); b["labels"] = b["labels"].float(); return b

    def metrics(p):
        probs = 1 / (1 + np.exp(-p.predictions))
        f = precision_recall_fscore_support(p.label_ids, (probs >= 0.5).astype(int), average="macro", zero_division=0)[2]
        return {"macro_f1": f}

    args = TrainingArguments(output_dir=f"/tmp/{tag}", per_device_train_batch_size=16, gradient_accumulation_steps=2,
                             per_device_eval_batch_size=32, num_train_epochs=4, learning_rate=2e-5, warmup_ratio=0.1,
                             weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=200,
                             report_to="none", fp16=torch.cuda.is_available(), seed=a.seed)
    keep = KeepBest()
    tr = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                 data_collator=Coll(tok), compute_metrics=metrics, callbacks=[keep])
    tr.train()
    model.load_state_dict(keep.state)
    print(f"best epoch {keep.epoch} | val macro-F1@0.5 {keep.best:.4f}")

    pv = tr.predict(ds["validation"])
    thr = tune_thr_f1(1 / (1 + np.exp(-pv.predictions)), pv.label_ids)

    # ---- test: the same 800 messages, same standalone convention as Table 3 / Appendix E.5
    recs = load_test_records()
    device = model.device
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(recs), 32):
            enc = tok([r["text"] for r in recs[i:i + 32]], truncation=True, padding="max_length",
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    probs_all = np.concatenate(probs_all)
    for r, p in zip(recs, probs_all):
        active = np.where(p >= thr)[0]
        r["abl_probs"] = {e: float(x) for e, x in zip(TARGET_EMOTIONS, p)}
        r["abl_label"] = "neutral" if len(active) == 0 else TARGET_EMOTIONS[active[np.argmax(p[active])]]
    res = {"init": a.init, "seed": a.seed, "best_epoch": keep.epoch,
           "backbone": scores_by_domain(recs, "abl_label"),
           "paper_backbone_for_reference": scores_by_domain(recs, "m1_label")}

    if a.with_llm:
        from openai import OpenAI
        build = load_prompt_builder()
        client = OpenAI(api_key=config.get_openai_key())
        sys_msg = ("You are an assistant that ALWAYS responds with a single JSON object. "
                   "This message explicitly contains the word json to satisfy the API requirement.")
        fails = 0
        for r in recs:
            try:
                resp = client.chat.completions.create(
                    model="gpt-5-mini", response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": sys_msg},
                              {"role": "user", "content": build(r["text"], {"probs": r["abl_probs"]})}])
                lab = parse_label(resp.choices[0].message.content)
            except Exception:
                lab = None
            if lab is None:
                fails += 1; lab = r["abl_label"]
            r["abl_pipeline_label"] = lab
        res["pipeline"] = scores_by_domain(recs, "abl_pipeline_label")
        res["pipeline_llm_failures"] = fails

    save_json(res, f"{out}/metrics.json")
    save_json({e: float(t) for e, t in zip(TARGET_EMOTIONS, thr)}, f"{out}/thresholds.json")
    save_json([{k: r.get(k) for k in ("text", "gold_label", "domain", "m1_label", "abl_label", "abl_pipeline_label")}
               for r in recs], f"{out}/per_message.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
