prev_y = None
last_switch_t = 0
SWITCH_COOLDOWN = 0.8  # seconds — stops Alt+Tab spam

while True:
    frame = webcam.read()
    landmarks = mediapipe_hands(frame)
    t = now()

    if not landmarks:
        prev_y = None
        continue

    # CLASSIFIER STEP RUNS HERE.
    state = classifier(landmarks)  # scroll_up | scroll_down | switch | idle
    # or: state from classifier, flick confirmed by velocity separately

    if state == "scroll_up":
        y = landmarks["thumb_tip"].y
        if prev_y is not None:
            dy = prev_y - y
            if abs(dy) > dead_zone:
                send_scroll(+gain * dy)   # up
        prev_y = y

    elif state == "scroll_down":
        y = landmarks["index_tip"].y
        if prev_y is not None:
            dy = prev_y - y              # tip moved up
            if abs(dy) > dead_zone:
                send_scroll(-gain * dy)  # down
        prev_y = y

    elif state == "switch":
        prev_y = None
        if t - last_switch_t > SWITCH_COOLDOWN:
            # Windows: Alt+Tab once
            # Mac: Cmd+Tab once
            send_hotkey("alt+tab")  # or "cmd+tab"
            last_switch_t = t

    else:  # idle
        prev_y = None