"""
Pseudocode for the live control loop (see runtime.py for the real implementation).

One hand (right by default, --hand left) does everything.
  scroll: index out, or index + middle out -> page follows the fingertip
  shaka:  thumb + pinky out                -> Alt+Tab / Cmd+Tab once
  idle:   anything else
Ctrl+Alt+P (or p in the preview) pauses everything.
"""

SWITCH_COOLDOWN = 1.5  # seconds - stops Alt+Tab spam
dead_zone = 0.008  # image-heights
gain = 40.0
prev_shaka = False
last_switch_t = 0

while True:
    frame = webcam.read()
    hand = mediapipe_hands(frame)  # gesture hand's landmarks or None
    t = now()

    # Classify (idle unless the winning SVM score beats the trained min_margin),
    # then require the pose to repeat for a few frames before acting,
    # so one bad frame can't scroll or switch. Paused feeds "idle".
    raw = classifier(hand) if hand and not paused else "idle"
    state = debounce(raw)  # scroll | shaka | idle

    if state == "scroll" and raw == "scroll":
        y = smooth(hand["index_tip"].y)  # EMA filter against jitter
        # Measured from the last point that scrolled, so slow drags still add up.
        dy = y - anchor
        if abs(dy) > dead_zone:
            speed = abs(dy) / (t - anchor_t)
            boost = min(1 + speed / speed_ref, max_boost)  # faster drag, further scroll
            send_scroll(gain * boost * dy)  # finger down -> scroll up
            anchor, anchor_t = y, t
    else:
        reset_scroller()

    shaka = state == "shaka"
    if shaka and not prev_shaka and t - last_switch_t > SWITCH_COOLDOWN:
        send_hotkey("alt+tab")  # or "cmd+tab" on Mac
        last_switch_t = t
    prev_shaka = shaka
