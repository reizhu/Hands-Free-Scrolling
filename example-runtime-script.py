"""
Pseudocode for the live control loop (see runtime.py for the real implementation).

Poses:
  scroll_up:   index + middle out
  scroll_down: index out
  switch:      middle + ring + pinky out
  reject:      anything else

One-way scroll:
  scroll_up:   tips move DOWN -> scroll up; opposite ignored
  scroll_down: tip moves UP   -> scroll down; opposite ignored
"""

prev_y = None
last_switch_t = 0
SWITCH_COOLDOWN = 0.8  # seconds - stops Alt+Tab spam
dead_zone = 0.008
gain = 40.0
prev_state = None

while True:
    frame = webcam.read()
    landmarks = mediapipe_hands(frame)
    t = now()

    if not landmarks:
        prev_y = None
        prev_state = None
        continue

    # CLASSIFIER STEP RUNS HERE.
    state = classifier(landmarks)  # scroll_up | scroll_down | switch | reject

    if state == "scroll_up":
        # Track index + middle tips (average)
        y = (landmarks["index_tip"].y + landmarks["middle_tip"].y) / 2
        if prev_y is not None:
            dy = y - prev_y  # tips moved down
            if dy > dead_zone:  # one-way: ignore tips moving up
                send_scroll(+gain * dy)  # up
        prev_y = y

    elif state == "scroll_down":
        y = landmarks["index_tip"].y
        if prev_y is not None:
            dy = prev_y - y  # tip moved up
            if dy > dead_zone:  # one-way: ignore tip moving down
                send_scroll(-gain * dy)  # down
        prev_y = y

    elif state == "switch":
        prev_y = None
        # Fire once on entering the pose
        if prev_state != "switch" and t - last_switch_t > SWITCH_COOLDOWN:
            # Windows: Alt+Tab once
            # Mac: Cmd+Tab once
            send_hotkey("alt+tab")  # or "cmd+tab"
            last_switch_t = t

    else:  # reject
        prev_y = None

    prev_state = state
