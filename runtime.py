"""
Live demo: MediaPipe landmarks -> pose classifier -> one-way scroll / Alt+Tab.

Poses:
  scroll_up:   index + middle out; tips move DOWN -> scroll up; tip up ignored
  scroll_down: index out; tip moves UP -> scroll down; tip down ignored
  switch:      middle + ring + pinky out -> Alt+Tab once on pose enter
  reject:      do nothing

Keys: q quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Controller as MouseController

from train_pose import landmarks_to_features

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
POSE_MODEL = ROOT / "models" / "pose_classifier.joblib"

DEAD_ZONE = 0.008
SCROLL_GAIN = 40.0
SWITCH_COOLDOWN_S = 1.5
INDEX_TIP = 8
MIDDLE_TIP = 12

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


def mp_landmarks_to_dicts(hand_landmarks) -> list[dict]:
    return [
        {"name": LANDMARK_NAMES[i], "x": lm.x, "y": lm.y, "z": lm.z}
        for i, lm in enumerate(hand_landmarks)
    ]


def make_landmarker() -> vision.HandLandmarker:
    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Missing {HAND_MODEL}")
    options = vision.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def load_pose_classifier():
    if not POSE_MODEL.exists():
        raise FileNotFoundError(
            f"Missing {POSE_MODEL}. Run: .\\.venv\\Scripts\\python train_pose.py"
        )
    blob = joblib.load(POSE_MODEL)
    return blob["model"]


def send_scroll(amount: float, mouse: MouseController) -> None:
    # pynput: positive dy scrolls up on Windows
    ticks = int(round(amount))
    if ticks == 0:
        return
    mouse.scroll(0, ticks)


def send_alt_tab(keyboard: KeyboardController) -> None:
    modifier = Key.cmd if sys.platform == "darwin" else Key.alt
    keyboard.press(modifier)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(modifier)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open webcam and verify real frames (Windows often lies with isOpened())."""
    backends: list[int] = []
    if sys.platform == "win32":
        # DirectShow is the reliable Windows backend; MSMF often opens then fails reads.
        backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])
    backends.append(cv2.CAP_ANY)

    tried: list[str] = []
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        label = {
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_ANY: "ANY",
        }.get(backend, str(backend))
        tried.append(label)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = False, None
        for _ in range(40):
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                print(f"Webcam opened (index {index}, backend {label}).")
                return cap
            time.sleep(0.05)

        cap.release()

    raise RuntimeError(
        f"Webcam index {index} opened but returned no frames "
        f"(tried backends: {', '.join(tried)}). "
        "Close Zoom/Teams/capture.py/other camera apps and retry."
    )


def read_frame(cap: cv2.VideoCapture, retries: int = 10) -> np.ndarray | None:
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(0.02)
    return None


def main() -> None:
    clf = load_pose_classifier()
    landmarker = make_landmarker()
    mouse = MouseController()
    keyboard = KeyboardController()

    cap = open_camera(0)

    prev_tip_y: float | None = None
    prev_state: str | None = None
    last_switch_t = 0.0
    frame_ts_ms = 0
    consecutive_misses = 0

    window_name = "Hands-Free runtime"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("Runtime started. Focus a scrollable window. q=quit")

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                consecutive_misses += 1
                if consecutive_misses >= 30:
                    raise RuntimeError(
                        "Lost webcam frames repeatedly. "
                        "Is another app using the camera?"
                    )
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                continue
            consecutive_misses = 0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            state = "reject"

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                lm_dicts = mp_landmarks_to_dicts(hand)
                feats = landmarks_to_features(lm_dicts).reshape(1, -1)
                state = str(clf.predict(feats)[0])
                now = time.time()

                if state == "scroll_up":
                    tip_y = (lm_dicts[INDEX_TIP]["y"] + lm_dicts[MIDDLE_TIP]["y"]) / 2.0
                    if prev_state != "scroll_up":
                        prev_tip_y = None
                    if prev_tip_y is not None:
                        # Tips moved DOWN only -> scroll up
                        dy = tip_y - prev_tip_y
                        if dy > DEAD_ZONE:
                            send_scroll(+SCROLL_GAIN * dy, mouse)
                    prev_tip_y = tip_y

                elif state == "scroll_down":
                    tip_y = lm_dicts[INDEX_TIP]["y"]
                    if prev_state != "scroll_down":
                        prev_tip_y = None
                    if prev_tip_y is not None:
                        # Tip moved UP only -> scroll down
                        dy = prev_tip_y - tip_y
                        if dy > DEAD_ZONE:
                            send_scroll(-SCROLL_GAIN * dy, mouse)
                    prev_tip_y = tip_y

                elif state == "switch":
                    prev_tip_y = None
                    if prev_state != "switch" and (now - last_switch_t) > SWITCH_COOLDOWN_S:
                        send_alt_tab(keyboard)
                        last_switch_t = now

                else:
                    prev_tip_y = None
            else:
                prev_tip_y = None

            prev_state = state

            cv2.putText(
                frame,
                f"state: {state}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "up: index+middle | down: index | switch: mid+ring+pinky | q quit",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
