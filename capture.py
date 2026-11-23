"""
Webcam capture + MediaPipe hand landmarks + labeled saves.

Keys:
  1  set label scroll_up   (index + middle out, still)
  2  set label scroll_down (index out, still)
  3  set label switch      (middle + ring + pinky out, still)
  0  set label reject      (still + video)
  s  save still (scroll_up / scroll_down / switch / reject)
  r  start/stop video recording (reject)
  q  quit

Stills:
  data/images/<label>/<label>_<n>.jpg
  data/landmarks/<label>/<label>_<n>.json

Videos (r start, r stop):
  data/videos/<label>/<label>_<n>.mp4
  data/landmarks/<label>/<label>_<n>.json   (per-frame landmark sequence)
"""

from __future__ import annotations

import json
import re
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
    ord("0"): "reject",
}

# Poses are stills. reject also allows video for motion negatives.
VIDEO_ONLY_LABELS: set[str] = set()
RECORDABLE_LABELS = {"reject"}
STILL_LABELS = {"scroll_up", "scroll_down", "switch", "reject"}

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
        (LANDMARK_DIR / label).mkdir(parents=True, exist_ok=True)
        if label in STILL_LABELS:
            (IMAGE_DIR / label).mkdir(parents=True, exist_ok=True)
        if label in RECORDABLE_LABELS:
            (VIDEO_DIR / label).mkdir(parents=True, exist_ok=True)


def next_sample_name(label: str) -> str:
    """Return next unused name: <label>_0, <label>_1, ... across images/videos/landmarks."""
    pattern = re.compile(rf"^{re.escape(label)}_(\d+)$")
    used: set[int] = set()
    for folder in (IMAGE_DIR / label, VIDEO_DIR / label, LANDMARK_DIR / label):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.stem)
            if match:
                used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return f"{label}_{n}"


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


def wrap_text(text: str, max_width_px: int, font_scale: float, thickness: int) -> list[str]:
    """Split text into lines that fit within max_width_px (OpenCV putText)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split(" ")
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if w <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
            (w, _), _ = cv2.getTextSize(current, font, font_scale, thickness)
            # Hard-split an oversized token so it can't blow past the edge.
            while w > max_width_px and len(current) > 1:
                cut = max(1, len(current) // 2)
                # Shrink until the left piece fits.
                while cut > 1:
                    (lw, _), _ = cv2.getTextSize(current[:cut], font, font_scale, thickness)
                    if lw <= max_width_px:
                        break
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
                (w, _), _ = cv2.getTextSize(current, font, font_scale, thickness)
    lines.append(current)
    return lines


def draw_wrapped_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (0, 200, 255),
    thickness: int = 1,
    line_height: int = 22,
    margin_right: int = 12,
) -> None:
    max_width = max(40, frame.shape[1] - x - margin_right)
    for i, line in enumerate(wrap_text(text, max_width, font_scale, thickness)):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def save_still(
    frame: np.ndarray,
    hand_landmarks,
    label: str,
    handedness: str | None,
) -> Path:
    name = next_sample_name(label)
    img_path = IMAGE_DIR / label / f"{name}.jpg"
    json_path = LANDMARK_DIR / label / f"{name}.json"

    # Bake skeleton into review image (training uses JSON only).
    save_frame = frame.copy()
    draw_hand(save_frame, hand_landmarks)
    cv2.imwrite(str(img_path), save_frame)

    payload = {
        "label": label,
        "type": "still",
        "sample": name,
        "handedness": handedness,
        "image": str(img_path.relative_to(ROOT)).replace("\\", "/"),
        "landmarks": landmarks_to_list(hand_landmarks),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def save_video_clip(buffer: list[dict], label: str) -> tuple[Path | None, str]:
    """Write recorded frames as mp4 + matching per-frame landmark JSON (same name)."""
    if not buffer:
        return None, f"{label}: empty recording"

    frames_with_hand = sum(1 for item in buffer if item["landmarks"] is not None)
    if frames_with_hand < MIN_HAND_FRAMES:
        return None, f"{label}: need >={MIN_HAND_FRAMES} hand frames (got {frames_with_hand})"

    # Same name -> same clip: reject_0.mp4 <-> reject_0.json
    name = next_sample_name(label)
    video_path = VIDEO_DIR / label / f"{name}.mp4"
    json_path = LANDMARK_DIR / label / f"{name}.json"

    h, w = buffer[0]["frame"].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (w, h),
    )

    sequence = []
    for i, item in enumerate(buffer):
        writer.write(item["frame"])
        sequence.append(
            {
                "frame_index": i,
                "handedness": item["handedness"],
                "landmarks": item["landmarks"],
            }
        )
    writer.release()

    written_frames = len(sequence)
    verified_frames = count_video_frames(video_path)
    video_rel = str(video_path.relative_to(ROOT)).replace("\\", "/")
    json_rel = str(json_path.relative_to(ROOT)).replace("\\", "/")

    payload = {
        "label": label,
        "type": "video",
        "sample": name,
        "pair_id": name,
        "fps": TARGET_FPS,
        "num_frames": written_frames,
        "video_frame_count": verified_frames,
        "frames_match_video": verified_frames == written_frames,
        "frames_with_hand": frames_with_hand,
        "video": video_rel,
        "frames": sequence,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    match = "OK" if verified_frames == written_frames else "MISMATCH"
    msg = (
        f"saved pair {name} | {video_rel} + {json_rel} | "
        f"{written_frames} frames, video has {verified_frames} [{match}]"
    )
    return json_path, msg


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


def mode_text(label: str, recording: bool) -> str:
    if recording:
        return "REC"
    if label in RECORDABLE_LABELS and label in STILL_LABELS:
        return "s=still / r=video"
    if label in RECORDABLE_LABELS:
        return "ready - press r"
    return "still"


def hint_text(label: str, recording: bool) -> str:
    if recording:
        return "r=stop"
    if label in RECORDABLE_LABELS and label in STILL_LABELS:
        return "s save still | r start/stop video | q quit"
    if label in RECORDABLE_LABELS:
        return "r start/stop video | q quit"
    return "1/2/3/0 label | s save still | q quit"


def main() -> None:
    ensure_dirs()
    landmarker = make_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    label = "reject"
    saved_count = 0
    last_save_msg = ""
    frame_ts_ms = 0

    recording = False
    record_buffer: list[dict] = []
    record_started_ms = 0

    print(
        "Capture running. Keys: 1/2/3/0 label | s=still | r=video (reject) | q=quit"
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
                    # Frame is mirrored above for a selfie view, which swaps
                    # MediaPipe's Left/Right. Flip the label back to the user's
                    # real hand.
                    raw = result.handedness[0][0].category_name
                    handedness = {"Left": "Right", "Right": "Left"}.get(raw, raw)

            if recording:
                # Bake skeleton into review video (no HUD text); JSON is training data.
                rec_frame = frame.copy()
                if hand_landmarks is not None:
                    draw_hand(rec_frame, hand_landmarks)
                record_buffer.append(
                    {
                        "frame": rec_frame,
                        "t_ms": frame_ts_ms,
                        "handedness": handedness,
                        "landmarks": landmark_list,
                    }
                )
                elapsed = (frame_ts_ms - record_started_ms) / 1000.0
                if elapsed >= MAX_RECORD_SECONDS:
                    path, msg = save_video_clip(record_buffer, label)
                    recording = False
                    record_buffer = []
                    if path is not None:
                        saved_count += 1
                    last_save_msg = f"auto-stop: {msg}"
                    print(last_save_msg)

            display = frame.copy()
            if hand_landmarks is not None:
                draw_hand(display, hand_landmarks)

            hud = f"label: {label} ({mode_text(label, recording)})  saved: {saved_count}"
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
            else:
                cv2.putText(
                    display,
                    hint_text(label, recording),
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            if last_save_msg:
                draw_wrapped_text(display, last_save_msg, x=12, y=84)

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
                if label not in RECORDABLE_LABELS:
                    last_save_msg = "r only for reject (0)"
                    continue
                if not recording:
                    recording = True
                    record_buffer = []
                    record_started_ms = frame_ts_ms
                    last_save_msg = f"recording {label}..."
                else:
                    path, msg = save_video_clip(record_buffer, label)
                    recording = False
                    record_buffer = []
                    if path is not None:
                        saved_count += 1
                    last_save_msg = msg
                    print(last_save_msg)
                continue

            if key == ord("s"):
                if label not in STILL_LABELS:
                    last_save_msg = f"s not supported for {label}"
                elif hand_landmarks is None:
                    last_save_msg = "no hand - not saved"
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
