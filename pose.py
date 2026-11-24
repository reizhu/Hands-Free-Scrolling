"""Shared hand-landmark helpers for capture.py, train_pose.py and runtime.py."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"

POSE_LABELS = ("scroll", "shaka", "idle")
# Hand each gesture must be made with; idle may be either hand.
REQUIRED_HAND = {"scroll": "Right", "shaka": "Right"}

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


HAND_CONNECTIONS = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


def draw_hand(frame: np.ndarray, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    points = []
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2, lineType=cv2.LINE_AA)


def landmarks_to_dicts(hand_landmarks) -> list[dict]:
    return [
        {"name": LANDMARK_NAMES[i], "x": lm.x, "y": lm.y, "z": lm.z}
        for i, lm in enumerate(hand_landmarks)
    ]


def mirror_x(landmarks: list[dict]) -> list[dict]:
    """Flip a hand left-right, so a left hand looks like the right hand the model was trained on."""
    return [{**lm, "x": 1.0 - lm["x"]} for lm in landmarks]


def user_hand(mediapipe_label: str) -> str:
    """Frames are mirrored for a selfie view, which swaps MediaPipe's Left/Right."""
    return {"Left": "Right", "Right": "Left"}.get(mediapipe_label, mediapipe_label)


def make_landmarker(num_hands: int) -> vision.HandLandmarker:
    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"Missing model at {HAND_MODEL}. Download hand_landmarker.task into models/."
        )
    options = vision.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class FrameClock:
    """Real, strictly increasing ms timestamps for MediaPipe VIDEO mode."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.last = -1

    def now_ms(self) -> int:
        self.last = max(self.last + 1, int((time.monotonic() - self.t0) * 1000))
        return self.last


def landmarks_to_features(landmarks: list[dict]) -> np.ndarray:
    """21 points -> wrist-relative, hand-size-normalized xyz (63) + 9 distances = 72 floats."""
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
    # Distances a linear model can't derive from raw xyz: fingertip-to-wrist
    # (finger extension) and thumb-tip-to-fingertip (pinch).
    tips = coords[[4, 8, 12, 16, 20]]
    extension = np.linalg.norm(tips, axis=1)
    thumb_gap = np.linalg.norm(tips[1:] - tips[0], axis=1)
    return np.concatenate([coords.reshape(-1), extension, thumb_gap])


def labels_from_scores(scores: np.ndarray, classes: np.ndarray, min_margin: float) -> np.ndarray:
    """SVM decision scores (n, k) or (n,) -> labels; idle when the winner's score <= min_margin."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:  # binary model: sign picks the class
        idx, margin = (scores > 0).astype(int), np.abs(scores)
    else:
        idx = np.argmax(scores, axis=1)
        margin = scores[np.arange(len(scores)), idx]
    return np.where(margin > min_margin, np.asarray(classes)[idx], "idle")
