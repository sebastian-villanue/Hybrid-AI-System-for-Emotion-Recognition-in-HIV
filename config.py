"""config.py — the single place for credentials and repo ids; both notebooks import this.

SECURITY: if you paste real tokens below, NEVER commit the change
(`git update-index --skip-worktree config.py` after cloning, or export
env vars instead — environment variables always take precedence).
"""
import os

HF_TOKEN        = "hf_xxxxxxxxxxxxxxxx-PLACEHOLDER"   # only needed for private repos / uploads
OPENAI_API_KEY  = "sk-xxxxxxxxxxxxxxxx-PLACEHOLDER"   # only needed for the GPT-5-mini refinement stage
HF_DATASET_REPO = "SebastianVla/HIVEmotions"
HF_MODEL_REPO   = "SebastianVla/HIVRoBERTa"

def get_hf_token():
    t = os.environ.get("HF_TOKEN")
    if t: return t
    return None if "PLACEHOLDER" in HF_TOKEN else HF_TOKEN   # None is fine for public repos

def get_openai_key():
    k = os.environ.get("OPENAI_API_KEY")
    if k: return k
    if "PLACEHOLDER" in OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY (env) or fill config.OPENAI_API_KEY — "
                           "needed only for the LLM refinement stage.")
    return OPENAI_API_KEY
