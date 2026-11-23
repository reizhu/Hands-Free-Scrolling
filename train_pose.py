"""
Train a static pose classifier from landmark JSON stills.

Uses data/landmarks/<label>/*.json with type == "still" (or a single-frame landmarks list).
Labels expected: scroll_up, scroll_down, switch, reject
(index+middle / index / middle+ring+pinky / everything else).

Run:
  .\\.venv\\Scripts\\python train_pose.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
LANDMARK_DIR = ROOT / "data" / "landmarks"
MODEL_PATH = ROOT / "models" / "pose_classifier.joblib"

POSE_LABELS = ("scroll_up", "scroll_down", "switch", "reject")


def landmarks_to_features(landmarks: list[dict]) -> np.ndarray:
    """21 points -> wrist-relative, hand-size-normalized xyz vector (63 floats)."""
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")

    coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float64)
    wrist = coords[0].copy()
    coords -= wrist

    # Scale by mean distance from wrist so camera distance matters less.
    dists = np.linalg.norm(coords[1:], axis=1)
    scale = float(np.mean(dists)) if np.any(dists > 0) else 1.0
    if scale < 1e-6:
        scale = 1.0
    coords /= scale
    return coords.reshape(-1)


def load_still_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X: list[np.ndarray] = []
    y: list[str] = []
    samples: list[str] = []

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
            X.append(landmarks_to_features(landmarks))
            y.append(data.get("label", label))
            samples.append(data.get("sample", path.stem))

    if not X:
        raise FileNotFoundError(
            f"No still landmark JSONs found under {LANDMARK_DIR}. "
            "Capture scroll_up / scroll_down / switch / reject stills first."
        )
    return np.stack(X), np.array(y), samples


def main() -> None:
    X, y, _samples = load_still_dataset()
    classes, counts = np.unique(y, return_counts=True)
    print("Dataset:")
    for c, n in zip(classes, counts):
        print(f"  {c}: {n}")

    if len(classes) < 2:
        raise RuntimeError("Need at least 2 classes to train.")

    # Stratified split; tiny sets fall back to a smaller test fraction.
    test_size = 0.25 if len(y) >= 12 else 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=5000, dual="auto")),
        ]
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("\nTest accuracy:", float(np.mean(pred == y_test)))
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(y_test, pred, labels=classes))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": list(classes)}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")

    if "reject" not in classes:
        print(
            "\nNote: no reject stills yet. Add some (wrong poses / near-misses) "
            "so the model learns when NOT to scroll."
        )


if __name__ == "__main__":
    main()
