"""Offline checks for runtime decision logic. Run: python test_runtime.py"""

import json
from pathlib import Path

import joblib
import numpy as np

from types import SimpleNamespace

from pose import POSE_LABELS, FrameClock, landmarks_to_features, mirror_x
from runtime import (
    DEAD_ZONE,
    HOLD_FRAMES,
    POSE_MODEL,
    Debouncer,
    Scroller,
    Toggle,
    classify,
    hands_by_side,
    parse_args,
)


def run(raws):
    d = Debouncer(HOLD_FRAMES)
    return [d.update(r) for r in raws]


# One-frame blips never change state.
assert set(run(["scroll"] * 5 + ["shaka"] + ["scroll"] * 5)) == {"idle", "scroll"}
assert run(["scroll"] * 5 + ["idle"] + ["scroll"] * 5)[-1] == "scroll"
# Shaka (fires a keystroke) needs a longer hold than scroll.
assert "shaka" not in run(["shaka"] * (HOLD_FRAMES["shaka"] - 1))
assert run(["shaka"] * HOLD_FRAMES["shaka"])[-1] == "shaka"


class FakeClf:
    classes_ = np.array(["idle", "scroll", "shaka"])

    def __init__(self, scores):
        self.scores = np.array([scores])

    def decision_function(self, _):
        return self.scores


# No class claims the hand -> idle.
assert classify(FakeClf([-0.5, -0.2, -0.9]), None) == "idle"
assert classify(FakeClf([-0.5, 0.7, -0.9]), None) == "scroll"
# A winner below the tuned margin is not trusted.
assert classify(FakeClf([-0.5, 0.3, -0.9]), None, min_margin=0.5) == "idle"

# MediaPipe VIDEO mode needs strictly increasing timestamps, even for back-to-back frames.
clock = FrameClock()
ts = [clock.now_ms() for _ in range(5)]
assert all(b > a for a, b in zip(ts, ts[1:]))

# Pause, then a held shaka must re-hold after resume (runtime feeds idle while paused).
t = Toggle(); t.flip(); assert t.paused; t.flip(); assert not t.paused

# Frame is mirrored: MediaPipe "Left" is the user's right hand.
cat = lambda name: [SimpleNamespace(category_name=name)]
fake = SimpleNamespace(hand_landmarks=["a", "b"], handedness=[cat("Left"), cat("Right")])
assert hands_by_side(fake) == {"Right": "a", "Left": "b"}
assert hands_by_side(SimpleNamespace(hand_landmarks=[], handedness=[])) == {}



def drag(scroller, ys, dt):
    """Total ticks for fingertip positions ys sampled every dt seconds."""
    return sum(scroller.update(y, i * dt) for i, y in enumerate(ys))


# Slow drag: every frame moves less than the dead zone, but it still scrolls.
step = DEAD_ZONE / 4
slow = drag(Scroller(), [0.5 + i * step for i in range(200)], 1 / 30)
assert slow > 0, slow
# Jitter around one spot never scrolls.
assert drag(Scroller(), [0.5 + (DEAD_ZONE * 0.9 if i % 2 else 0) for i in range(100)], 1 / 30) == 0
# Same distance moved faster scrolls further.
fast = drag(Scroller(), [0.5 + i * (200 * step / 20) for i in range(21)], 1 / 30)
assert fast > slow, (fast, slow)
# --invert flips direction; finger down (y up) scrolls up (positive) by default.
assert drag(Scroller(invert=True), [0.5 + i * step for i in range(200)], 1 / 30) == -slow

# Left hand is mirrored into a right hand: x offsets flip sign, y/z unchanged.
pts = [{"x": 0.3 + 0.01 * i, "y": 0.5 - 0.02 * i, "z": 0.001 * i} for i in range(21)]
f, fm = landmarks_to_features(pts), landmarks_to_features(mirror_x(pts))
assert np.allclose(fm[0:63:3], -f[0:63:3]) and np.allclose(fm[1:63:3], f[1:63:3]) and np.allclose(fm[63:], f[63:])

args = parse_args([])
assert (args.hand, args.invert, args.no_preview) == ("right", False, False)
assert parse_args(["--hand", "left", "--invert", "--no-preview"]).hand == "left"

# Saved stills classify as their own label (with the tuned margin) (skipped until a model is trained on current labels).
blob = joblib.load(POSE_MODEL) if POSE_MODEL.exists() else {"labels": []}
clf, margin = blob.get("model"), blob.get("min_margin", 0.0)
paths = [p for lab in blob["labels"] if lab in POSE_LABELS for p in Path("data/landmarks", lab).glob("*.json")]
for path in paths:
    data = json.loads(path.read_text())
    if data.get("type") == "still":
        got = classify(clf, landmarks_to_features(data["landmarks"]).reshape(1, -1), margin)
        assert got == data["label"], (path, got)

print("ok")
