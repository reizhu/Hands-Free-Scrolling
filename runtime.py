"""
Live demo: MediaPipe landmarks -> pose classifier -> scroll / Alt+Tab.

One hand (right) does everything.
  scroll: index out, or index + middle out; fingertip moves down -> scroll up,
          up -> scroll down (follows the finger like dragging the page).
  shaka:  thumb + pinky out -> Alt+Tab (Cmd+Tab on Mac) once on pose enter
  idle:   do nothing (any other pose)

Keys: ctrl+alt+p (global) or p (preview window) pause/resume, q quit, Ctrl+C in the terminal

Flags: --hand left (use the left hand), --invert (flip scroll direction), --no-preview
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import GlobalHotKeys, Key
from pynput.mouse import Controller as MouseController

from pose import (
    ROOT,
    FrameClock,
    draw_hand,
    labels_from_scores,
    landmarks_to_dicts,
    landmarks_to_features,
    make_landmarker,
    mirror_x,
    user_hand,
)

POSE_MODEL = ROOT / "models" / "pose_classifier.joblib"

# Scroll feel. Positions are in image-height units (0..1).
DEAD_ZONE = 0.008  # fingertip must move this far from where it last scrolled; hides jitter
SMOOTHING = 0.5  # EMA weight of the newest fingertip position (1 = no smoothing)
SCROLL_GAIN = 40.0  # ticks per image-height at slow speed
SPEED_REF = 0.5  # image-heights/s at which a drag scrolls twice as far
MAX_BOOST = 3.0  # cap on the speed multiplier
SWITCH_COOLDOWN_S = 1.5
ACTIONS = {"scroll", "shaka"}  # anything else the classifier says counts as idle
# Frames a pose must hold before it takes effect; shaka is strictest (fires a keystroke).
HOLD_FRAMES = {"scroll": 3, "shaka": 6, "idle": 2}
INDEX_TIP = 8
# Pause/resume from any app. Mac: Option is <alt>; needs Input Monitoring permission.
PAUSE_HOTKEY = "<ctrl>+<alt>+p"


def load_pose_classifier():
    """(model, min_margin) from train_pose.py."""
    if not POSE_MODEL.exists():
        raise FileNotFoundError(
            f"Missing {POSE_MODEL}. Run: python train_pose.py"
        )
    blob = joblib.load(POSE_MODEL)
    return blob["model"], float(blob.get("min_margin", 0.0))


def classify(clf, feats: np.ndarray, min_margin: float = 0.0) -> str:
    """Predicted label, or idle when no class wins by more than min_margin."""
    return str(labels_from_scores(clf.decision_function(feats), clf.classes_, min_margin)[0])


class Toggle:
    """Pause flag flipped by the global hotkey (listener thread) or the preview window key."""

    def __init__(self):
        self.paused = False

    def flip(self) -> None:
        self.paused = not self.paused
        print("PAUSED" if self.paused else "ACTIVE")


class Debouncer:
    """Change state only after the same raw label repeats HOLD_FRAMES[label] times."""

    def __init__(self, hold: dict[str, int]):
        self.hold = hold
        self.state = self.candidate = "idle"
        self.count = 0

    def update(self, raw: str) -> str:
        self.count = self.count + 1 if raw == self.candidate else 1
        self.candidate = raw
        if self.count >= self.hold.get(raw, 1):
            self.state = raw
        return self.state


class Scroller:
    """Fingertip y -> scroll ticks: smoothed, jitter-gated, and faster drags scroll further."""

    def __init__(self, invert: bool = False):
        self.sign = -1.0 if invert else 1.0
        self.reset()

    def reset(self) -> None:
        self.smooth: float | None = None
        self.anchor = self.anchor_t = 0.0
        self.acc = 0.0  # carries sub-tick scroll so slow drags still add up

    def update(self, y: float, t: float) -> int:
        if self.smooth is None:
            self.smooth = self.anchor = y
            self.anchor_t = t
            return 0
        self.smooth += SMOOTHING * (y - self.smooth)
        # Measured from the last scroll point, not the last frame, so slow drags get past the dead zone.
        dy = self.smooth - self.anchor
        if abs(dy) <= DEAD_ZONE:
            return 0
        speed = abs(dy) / max(t - self.anchor_t, 1e-3)
        boost = min(1.0 + speed / SPEED_REF, MAX_BOOST)
        # Image y grows downward: finger down (dy > 0) -> positive ticks -> scroll up.
        self.acc += self.sign * SCROLL_GAIN * boost * dy
        self.anchor, self.anchor_t = self.smooth, t
        ticks = int(self.acc)
        self.acc -= ticks
        return ticks


def mac_accessibility_ok() -> bool:
    """False when macOS will silently drop our synthetic scroll/keys (no Accessibility permission)."""
    if sys.platform != "darwin":
        return True
    try:
        from HIServices import AXIsProcessTrusted  # ships with pynput's pyobjc deps on Mac
    except ImportError:
        return True  # can't tell; don't block
    return bool(AXIsProcessTrusted())


def hands_by_side(result) -> dict:
    """{"Left": landmarks, "Right": landmarks} for hands in view, keyed by the user's real hand."""
    return {
        user_hand(cats[0].category_name): hand
        for hand, cats in zip(result.hand_landmarks, result.handedness)
    }


def send_scroll(ticks: int, mouse: MouseController) -> None:
    # pynput: positive dy scrolls up. Mac "natural scrolling" may flip it; use --invert.
    if ticks:
        mouse.scroll(0, ticks)


def send_alt_tab(keyboard: KeyboardController) -> None:
    modifier = Key.cmd if sys.platform == "darwin" else Key.alt
    keyboard.press(modifier)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(modifier)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hands-free scrolling and window switching.")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="hand that makes the gestures")
    ap.add_argument("--invert", action="store_true", help="flip scroll direction")
    ap.add_argument("--no-preview", action="store_true", help="no camera window; pause with the hotkey, quit with Ctrl+C")
    return ap.parse_args(argv)


def put_text(frame, text: str, y: int, scale: float, color, thickness: int) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    side = args.hand.capitalize()
    if not mac_accessibility_ok():
        print(
            "WARNING: macOS Accessibility permission is missing, so scroll and Cmd+Tab will be ignored.\n"
            "  System Settings > Privacy & Security > Accessibility: enable your terminal/Python, then restart it."
        )

    clf, min_margin = load_pose_classifier()
    landmarker = make_landmarker(num_hands=2)
    mouse = MouseController()
    keyboard = KeyboardController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    debouncer = Debouncer(HOLD_FRAMES)
    scroller = Scroller(invert=args.invert)
    prev_scrolling = False
    prev_shaka = False
    last_switch_t = 0.0
    clock = FrameClock()
    toggle = Toggle()
    hotkeys = GlobalHotKeys({PAUSE_HOTKEY: toggle.flip})
    hotkeys.start()

    quit_hint = "Ctrl+C" if args.no_preview else "q or Ctrl+C"
    print(f"Runtime started ({args.hand} hand). Focus a scrollable window. {PAUSE_HOTKEY}=pause, {quit_hint}=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam stopped sending frames.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, clock.now_ms())

            hand = hands_by_side(result).get(side)
            pts = landmarks_to_dicts(hand) if hand is not None else None
            raw = "idle"
            # Paused feeds idle, so a pose must re-hold after resume before it acts.
            if pts is not None and not toggle.paused:
                model_pts = mirror_x(pts) if side == "Left" else pts  # model is trained on right hands
                raw = classify(clf, landmarks_to_features(model_pts).reshape(1, -1), min_margin)
                if raw not in ACTIONS:
                    raw = "idle"
            state = debouncer.update(raw)

            # Scroll by fingertip motion, only on frames that match the confirmed state.
            scrolling = pts is not None and raw == state == "scroll"
            if scrolling:
                if not prev_scrolling:
                    scroller.reset()
                send_scroll(scroller.update(pts[INDEX_TIP]["y"], time.monotonic()), mouse)
            prev_scrolling = scrolling

            # Alt+Tab once when the confirmed state becomes shaka.
            now = time.monotonic()
            shaka = state == "shaka"
            if shaka and not prev_shaka and (now - last_switch_t) > SWITCH_COOLDOWN_S:
                send_alt_tab(keyboard)
                last_switch_t = now
            prev_shaka = shaka

            if args.no_preview:
                continue
            if hand is not None:
                draw_hand(frame, hand)
            if toggle.paused:
                put_text(frame, "PAUSED", 28, 0.8, (0, 0, 255), 2)
            else:
                put_text(frame, f"state: {state} (raw: {raw})", 28, 0.8, (0, 255, 0), 2)
            put_text(
                frame,
                f"{args.hand} hand | scroll: index / index+middle | switch: shaka | {PAUSE_HOTKEY} or p pause | q quit",
                56,
                0.5,
                (220, 220, 220),
                1,
            )
            cv2.imshow("Hands-Free runtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                toggle.flip()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hotkeys.stop()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
