"""
Webcam capture + MediaPipe hand landmarks + labeled saves.

Keys:
  1  set label scroll_up   (still)
  2  set label scroll_down (still)
  3  set label switch      (manual video record)
  0  set label idle        (still)
  s  save still (pose / idle labels only)
  r  start/stop switch recording (only when label is switch)
  q  quit

Pose labels save:
  data/images/<label>/<timestamp>.jpg
  data/landmarks/<label>/<timestamp>.json

switch (press r to start, r again to stop):
  data/videos/switch/<timestamp>.mp4
  data/landmarks/switch/<timestamp>.json   (per-frame landmark sequence)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
IMAGE_DIR = ROOT / "data" / "images"
LANDMARK_DIR = ROOT / "data" / "landmarks"
VIDEO_DIR = ROOT / "data" / "videos"

LABELS = {
    ord("1"): "scroll_up",
    ord("2"): "scroll_down",
    ord("3"): "switch",
    ord("0"): "idle",
}

VIDEO_LABELS = {"switch"}
TARGET_FPS = 30
MAX_RECORD_SECONDS = 5.0
MIN_HAND_FRAMES = 5

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


def ensure_dirs() -> None:
    for label in LABELS.values():
        (IMAGE_DIR / label).mkdir(parents=True, exist_ok=True)
        (LANDMARK_DIR / label).mkdir(parents=True, exist_ok=True)
        if label in VIDEO_LABELS:
            (VIDEO_DIR / label).mkdir(parents=True, exist_ok=True)


def make_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"


def landmarks_to_list(hand_landmarks) -> list[dict]:
    out = []
    for i, lm in enumerate(hand_landmarks):
        out.append(
            {
                "name": LANDMARK_NAMES[i],
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
            }
        )
    return out


def draw_hand(frame: np.ndarray, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    points = []
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2, lineType=cv2.LINE_AA)


def save_still(
    frame: np.ndarray,
    hand_landmarks,
    label: str,
    handedness: str | None,
) -> Path:
    stamp = make_stamp()
    img_path = IMAGE_DIR / label / f"{stamp}.jpg"
    json_path = LANDMARK_DIR / label / f"{stamp}.json"

    cv2.imwrite(str(img_path), frame)

    payload = {
        "label": label,
        "type": "still",
        "timestamp": stamp,
        "handedness": handedness,
        "image": str(img_path.relative_to(ROOT)).replace("\\", "/"),
        "landmarks": landmarks_to_list(hand_landmarks),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def save_switch_clip(buffer: list[dict]) -> tuple[Path | None, str]:
    """Write recorded frames as mp4 + per-frame landmark JSON."""
    if not buffer:
        return None, "switch: empty recording"

    frames_with_hand = sum(1 for item in buffer if item["landmarks"] is not None)
    if frames_with_hand < MIN_HAND_FRAMES:
        return None, f"switch: need ≥{MIN_HAND_FRAMES} hand frames (got {frames_with_hand})"

    stamp = make_stamp()
    video_path = VIDEO_DIR / "switch" / f"{stamp}.mp4"
    json_path = LANDMARK_DIR / "switch" / f"{stamp}.json"

    h, w = buffer[0]["frame"].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (w, h),
    )

    t0 = buffer[0]["t_ms"]
    sequence = []
    for i, item in enumerate(buffer):
        writer.write(item["frame"])
        sequence.append(
            {
                "frame_index": i,
                "t_ms": item["t_ms"],
                "handedness": item["handedness"],
                "landmarks": item["landmarks"],
            }
        )
    writer.release()

    duration_s = (buffer[-1]["t_ms"] - t0) / 1000.0
    payload = {
        "label": "switch",
        "type": "video",
        "timestamp": stamp,
        "fps": TARGET_FPS,
        "duration_seconds": round(duration_s, 3),
        "num_frames": len(sequence),
        "frames_with_hand": frames_with_hand,
        "video": str(video_path.relative_to(ROOT)).replace("\\", "/"),
        "frames": sequence,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path, f"saved clip {json_path.name} ({len(sequence)} frames, {duration_s:.2f}s)"


def make_landmarker() -> vision.HandLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. "
            "Download hand_landmarker.task into models/."
        )

    options = vision.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main() -> None:
    ensure_dirs()
    landmarker = make_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    label = "idle"
    saved_count = 0
    last_save_msg = ""
    frame_ts_ms = 0

    recording = False
    record_buffer: list[dict] = []
    record_started_ms = 0

    print(
        "Capture running. Keys: 1/2/3/0 label | s=save still | r=start/stop switch | q=quit"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            # Mirror so it feels like a mirror; landmarks match the saved media.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_ts_ms += 33  # ~30 fps monotonic timestamps for VIDEO mode
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            hand_landmarks = None
            handedness = None
            landmark_list = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                landmark_list = landmarks_to_list(hand_landmarks)
                if result.handedness:
                    handedness = result.handedness[0][0].category_name

            if recording:
                record_buffer.append(
                    {
                        "frame": frame.copy(),  # clean, no HUD
                        "t_ms": frame_ts_ms,
                        "handedness": handedness,
                        "landmarks": landmark_list,
                    }
                )
                elapsed = (frame_ts_ms - record_started_ms) / 1000.0
                if elapsed >= MAX_RECORD_SECONDS:
                    path, msg = save_switch_clip(record_buffer)
                    recording = False
                    record_buffer = []
                    if path is not None:
                        saved_count += 1
                    last_save_msg = f"auto-stop: {msg}"
                    print(last_save_msg)

            display = frame.copy()
            if hand_landmarks is not None:
                draw_hand(display, hand_landmarks)

            if label == "switch":
                mode = "REC" if recording else "ready — press r"
            else:
                mode = "still"
            hud = f"label: {label} ({mode})  saved: {saved_count}"
            if handedness:
                hud += f"  hand: {handedness}"
            cv2.putText(
                display,
                hud,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if recording:
                elapsed = (frame_ts_ms - record_started_ms) / 1000.0
                cv2.circle(display, (display.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"REC {elapsed:.1f}s  (r=stop)",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif label == "switch":
                cv2.putText(
                    display,
                    "r start/stop switch clip | q quit",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "1/2/3/0 label | s save still | q quit",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            if last_save_msg:
                cv2.putText(
                    display,
                    last_save_msg,
                    (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Hands-Free capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in LABELS:
                if recording:
                    recording = False
                    record_buffer = []
                    last_save_msg = "recording cancelled (label changed)"
                label = LABELS[key]
                if not last_save_msg.startswith("recording cancelled"):
                    last_save_msg = f"label -> {label}"
                continue

            if key == ord("r"):
                if label != "switch":
                    last_save_msg = "r only works when label is switch (press 3)"
                    continue
                if not recording:
                    recording = True
                    record_buffer = []
                    record_started_ms = frame_ts_ms
                    last_save_msg = "recording…"
                else:
                    path, msg = save_switch_clip(record_buffer)
                    recording = False
                    record_buffer = []
                    if path is not None:
                        saved_count += 1
                    last_save_msg = msg
                    print(last_save_msg)
                continue

            if key == ord("s"):
                if label in VIDEO_LABELS:
                    last_save_msg = "switch uses r to start/stop, not s"
                elif hand_landmarks is None:
                    last_save_msg = "no hand — not saved"
                else:
                    path = save_still(frame, hand_landmarks, label, handedness)
                    saved_count += 1
                    last_save_msg = f"saved {path.name}"
                    print(last_save_msg)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
