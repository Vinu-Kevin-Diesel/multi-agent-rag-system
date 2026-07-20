"""Trained router classifier — the fast path that replaces the LLM router.

A logistic regression over the same MiniLM embedding the app already computes. Inference is a
single (384,) @ (3, 384) matmul plus an argmax — ~microseconds, no LLM call, and no second 8B
model resident in VRAM. Trained offline by scripts/train_router_classifier.py; only the weights
are shipped (router_model.npz), so this needs numpy, not scikit-learn.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.utils.embeddings import embed_single

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "router_model.npz"


@lru_cache(maxsize=1)
def _weights():
    """Load (coef, intercept, classes) once. Raises if the model was never trained."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{_MODEL_PATH.name} not found — run scripts/train_router_classifier.py "
            "or set ROUTER_MODE to 'llm'."
        )
    data = np.load(_MODEL_PATH)
    return data["coef"], data["intercept"], data["classes"]


async def classify_by_embedding(question: str) -> str:
    """Return one of: factual, comparative, multihop — via the trained classifier."""
    coef, intercept, classes = _weights()
    x = np.asarray(await embed_single(question), dtype=np.float32)
    logits = x @ coef.T + intercept
    category = str(classes[int(np.argmax(logits))])
    logger.info("[router:clf] Q=%r -> %s", question[:100], category)
    return category
