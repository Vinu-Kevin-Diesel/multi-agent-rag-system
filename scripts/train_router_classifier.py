"""Train the router classifier: MiniLM embeddings -> logistic regression -> 3 classes.

Replaces the LLM router (an ~0.35s no-think model call, plus a second 8B resident in VRAM)
with a ~1ms numpy matmul over the embedding the app already computes. Reports accuracy and
per-class F1 on a held-out split, and a confusion matrix, then exports the fitted weights.

Only the weights are exported (coef / intercept / classes) to a .npz, not a pickled sklearn
object — so runtime inference needs numpy only, not scikit-learn. See app/agents/router_classifier.py.

Usage (in the app container, which has sentence-transformers + the dev extras):
    docker compose run --rm --no-deps app python scripts/train_router_classifier.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "router_questions.jsonl"
OUT = ROOT / "app" / "agents" / "router_model.npz"
SEED = 20260722
CLASSES = ("factual", "comparative", "multihop")


def main():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [r["question"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"loaded {len(rows)} questions")

    # Same embedding model and normalisation the app uses at query time — the classifier must
    # see exactly the vectors it will be asked to classify in production.
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X = encoder.encode(questions, normalize_embeddings=True)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, C=10.0, random_state=SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n== held-out (20%) ==")
    print(classification_report(y_test, y_pred, digits=3))
    print("confusion matrix (rows=true, cols=pred), order", clf.classes_.tolist())
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

    # Export weights for numpy-only inference. Order of `classes` is the column order of the
    # decision function, so argmax over (x @ coef.T + intercept) maps straight back to a label.
    np.savez(
        OUT,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        classes=np.array(clf.classes_),
    )
    print(f"\nsaved weights -> {OUT}  (coef {clf.coef_.shape}, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
