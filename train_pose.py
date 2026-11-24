"""
Train a static pose classifier from landmark JSON stills.

Uses data/landmarks/<label>/*.json with type == "still" (or a single-frame landmarks list).
Labels expected: scroll, shaka, idle
(right-hand index or index+middle / right-hand thumb+pinky / everything else).

Cross-validation keeps stills captured close together in the same fold (they are
near-duplicates), then picks the SVM margin that runtime.py needs before acting.

Run:
  .\\.venv\\Scripts\\python train_pose.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pose import POSE_LABELS, REQUIRED_HAND, ROOT, labels_from_scores, landmarks_to_features

LANDMARK_DIR = ROOT / "data" / "landmarks"
MODEL_PATH = ROOT / "models" / "pose_classifier.joblib"

# Stills captured within the same window count as one group for CV.
GROUP_SECONDS = 30
# Margin tuning may give up at most this much scroll/shaka recall to cut false triggers.
MAX_RECALL_LOSS = 0.05


def load_still_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels, and a capture-time group per still."""
    X: list[np.ndarray] = []
    y: list[str] = []
    groups: list[str] = []

    for label in POSE_LABELS:
        folder = LANDMARK_DIR / label
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            # Skip video sequences for this pose model.
            if data.get("type") == "video" or "frames" in data:
                continue
            landmarks = data.get("landmarks")
            if not landmarks:
                continue
            hand = data.get("handedness")
            if label in REQUIRED_HAND and hand != REQUIRED_HAND[label]:
                print(f"skip {path.name}: {label} must be {REQUIRED_HAND[label]} hand, got {hand}")
                continue
            # shortcut: older stills have no captured_at, so file mtime stands in; lost on git checkout.
            t = data.get("captured_at", path.stat().st_mtime)
            X.append(landmarks_to_features(landmarks))
            y.append(data.get("label", label))
            groups.append(f"{label}:{int(t // GROUP_SECONDS)}")

    if not X:
        raise FileNotFoundError(
            f"No still landmark JSONs found under {LANDMARK_DIR}. "
            "Capture scroll / shaka / idle stills first."
        )
    return np.stack(X), np.array(y), np.array(groups)


def pick_margin(scores: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Raise the margin until idle stops triggering actions or action recall drops too far."""
    action = y != "idle"
    recall = lambda pred: np.mean(pred[action] == y[action])
    floor = recall(labels_from_scores(scores, classes, 0.0)) - MAX_RECALL_LOSS
    best = 0.0
    for m in np.arange(0.0, 3.01, 0.05):
        pred = labels_from_scores(scores, classes, m)
        if recall(pred) < floor:
            break
        best = float(m)
        if np.all(pred[~action] == "idle"):
            break
    return round(best, 2)


def report(title: str, y: np.ndarray, pred: np.ndarray, classes: np.ndarray) -> None:
    print(f"\n{title} accuracy: {float(np.mean(pred == y)):.3f}")
    print(classification_report(y, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(y, pred, labels=classes))
    false_triggers = int(np.sum((y == "idle") & (pred != "idle")))
    print(f"idle stills that would trigger an action: {false_triggers}/{int(np.sum(y == 'idle'))}")


def main() -> None:
    X, y, groups = load_still_dataset()
    classes = np.unique(y)
    print("Dataset (stills / capture groups):")
    for c in classes:
        print(f"  {c}: {int(np.sum(y == c))} / {len(set(groups[y == c]))}")

    if len(classes) < 2:
        raise RuntimeError("Need at least 2 classes to train.")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=5000, dual="auto")),
        ]
    )

    # Every sample is tested once by a model that never saw its capture group.
    min_margin = 0.0
    folds = min(5, min(len(set(groups[y == c])) for c in classes))
    if folds >= 2:
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=42)
        scores = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="decision_function")
        report(f"{folds}-fold grouped CV (margin 0)", y, labels_from_scores(scores, classes, 0.0), classes)
        min_margin = pick_margin(scores, y, classes)
        if min_margin > 0:
            report(f"Grouped CV (margin {min_margin})", y, labels_from_scores(scores, classes, min_margin), classes)
    else:
        print("\nSkipping CV: some class has < 2 capture groups. Capture more sessions.")
    clf.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": list(classes), "min_margin": min_margin}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH} (min_margin={min_margin})")

    if "idle" not in classes:
        print(
            "\nNote: no idle stills yet. Add some (wrong poses / near-misses) "
            "so the model learns when NOT to scroll or switch."
        )


if __name__ == "__main__":
    main()
