# Hybrid AI System for Emotion Recognition in HIV — code and results

Two notebooks, one config file, and the per-message results behind every number in the paper.

| Path | What it is |
|---|---|
| `notebooks/01_train_validate_test.ipynb` | Full training pipeline (DAPT → two-phase fine-tuning → both threshold sets) plus validation and test evaluation. Pulls all data from the HIVEmotions dataset repo: https://huggingface.co/datasets/SebastianVla/HIVEmotions |
| `notebooks/02_inference.ipynb` | Inference on your own messages (list or CSV). Pulls the released checkpoint + thresholds from the model repo: https://huggingface.co/SebastianVla/HIVRoBERTa |
| `config.py` | Single place for your Hugging Face and OpenAI tokens plus the two repo ids. Environment variables `HF_TOKEN` / `OPENAI_API_KEY` take precedence; **never commit real tokens**. |
| `results/` | The evaluation record behind the paper's tables — see `results/README.md`. |

**Setup:** open either notebook in Colab or Jupyter, fill `config.py` (or export the env vars), run top to bottom. Training needs a GPU (~2–4 h on a T4); inference runs anywhere. The OpenAI key is only needed for the GPT-5-mini refinement stage — the backbone works without it.

**Reproducibility note:** the paper's DAPT corpus included four legacy emotion categories not in the released dataset; notebook 01 documents this and uses the released subset. Retraining lands near, not on, the reported numbers (single-run, GPU nondeterminism); the released checkpoint on the model repo *is* the reported model.

Code is MIT-licensed (`LICENSE`); the dataset (CC BY 4.0) and model weights (CC BY-NC 4.0) carry their own licences in their repos.
