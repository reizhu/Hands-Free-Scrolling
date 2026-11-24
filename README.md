# Hands-Free Scrolling

CSE 455 project: control scrolling and window switching with hand gestures from a laptop webcam.

## What it does

| Gesture | Pose | Action |
|---|---|---|
| `scroll` | **Right** hand: index out, or index + middle out | While held, the page follows the fingertip: finger **down** → scroll up, finger **up** → scroll down. Drop the pose to reposition. |
| `shaka` | **Right** hand: thumb + pinky out, other fingers curled | Alt+Tab (Windows) / Cmd+Tab (Mac), once with cooldown |
| `idle` | Anything else, or a gesture made with the wrong hand | Do nothing |

One hand does everything: index out scrolls, shaka switches windows. A pose must hold for a few frames before it acts (`HOLD_FRAMES` in `runtime.py`), so single-frame misreads are ignored.

Pipeline: webcam → MediaPipe hand landmarks → pose classifier (JSON-trained) → OS input.

## Setup

### 1. Python deps

**Windows (PowerShell)**
```powershell
cd Hands-Free-Scrolling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac (Terminal)**
```bash
cd Hands-Free-Scrolling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After activate, both use `python` / `pip` from the venv. Re-run the activate line in each new terminal.

### 2. MediaPipe hand model (`hand_landmarker.task`)

`pip install mediapipe` installs the **library**, not the hand model weights. The Tasks API loads a separate `.task` file at runtime (same file on Windows and Mac). Commit it in `models/` or download once:

**Windows**
```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models\hand_landmarker.task"
```

**Mac**
```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

## Capture training data

With the venv activated:

```bash
python capture.py
```

Keys:

| Key | Action |
|---|---|
| `1` | label `scroll` (right hand, index or index + middle out, stills) |
| `2` | label `shaka` (right hand, thumb + pinky out, stills) |
| `0` | label `idle` (stills + video) |
| `s` | save still (scroll / shaka / idle). `scroll` and `shaka` are refused unless the right hand is seen. |
| `r` | start/stop video (`idle` only) |
| `q` | quit |

Saves:

- Stills → `data/images/<label>/` + `data/landmarks/<label>/<label>_N.json`
- Videos → `data/videos/<label>/` + matching landmark sequence JSON

Only the **landmark JSON** is used for training. Images/videos are for review (gitignored).

What to capture (press `s` once per still, change the pose a little between presses):

- `scroll` (right hand): capture both index-only and index + middle at different heights, angles and distances. The model only learns the pose; scrolling is handled in code, so no motion capture is needed.
- `shaka` (right hand): same variety.
- `idle`: as many as possible. Open/relaxed hand, fist, thumb-only, pinky-only, half-open shaka, holding or drinking from a cup, and left-hand gestures.

## Train the pose classifier

```bash
python train_pose.py
```

Reads still JSONs under `data/landmarks/{scroll,shaka,idle}/`, trains a Linear SVM, and writes `models/pose_classifier.joblib`.

Cross-validation is grouped: stills captured within 30 s of each other are near-duplicates, so they stay in the same fold. Each new still records a `captured_at` time for this; older stills fall back to the file's modified time, which resets on a fresh git checkout, so grouped CV numbers can shift slightly for teammates. This gives an honest accuracy, lower than a random split. The script then picks `min_margin`, the SVM score a gesture must beat before the runtime acts. It raises the margin until idle stills stop triggering actions, giving up at most 5 points of scroll/shaka recall. Watch the "idle stills that would trigger an action" line: each one is a false scroll or Alt+Tab. To lower it, capture more idle near-misses across several sessions.

Run the offline checks after training:

```bash
python test_runtime.py
```

## Run the live demo

```bash
python runtime.py
```

Focus a scrollable window, then use the gestures. Press `Ctrl+Alt+P` (Mac: Ctrl+Option+P) from any app, or `p` in the preview window, to pause or resume. While paused nothing is sent, and after resuming a pose must be held again before it acts. Press `q` in the preview window or `Ctrl+C` in the terminal to quit. The preview draws the tracked hand skeleton.

Options:

| Flag | Effect |
|---|---|
| `--hand left` | Make the gestures with the left hand. The model is trained on right-hand stills; the left hand is mirrored before classifying, so no retraining is needed. |
| `--invert` | Flip scroll direction (e.g. if Mac "natural scrolling" makes it feel backwards). |
| `--no-preview` | No camera window. Pause with the hotkey, quit with `Ctrl+C`. |

Scroll feel is tuned by the constants at the top of `runtime.py`: `SMOOTHING` (fingertip jitter filter), `DEAD_ZONE` (how far the finger must move before scrolling), `SCROLL_GAIN` (scroll per finger movement), and `SPEED_REF` / `MAX_BOOST` (faster drags scroll further, up to 3x).

The preview shows `state` (what the app acts on) and `raw` (this frame's prediction).

### Mac permissions

- **Accessibility** for Terminal/Python: without it, scrolling and Cmd+Tab are silently ignored. The runtime prints a warning at start when it is missing.
- **Input Monitoring** for Terminal/Python: needed for the global pause hotkey (`Ctrl+Option+P`). The `p` key in the preview window works without it.

## Project layout

```
pose.py                    # shared landmark/feature/classifier helpers
capture.py                 # data collection
train_pose.py              # train pose model from landmark JSON
runtime.py                 # live MediaPipe + classifier + scroll/Alt+Tab
test_runtime.py            # offline checks for runtime decision logic
example-runtime-script.py  # short pseudocode of the control loop
models/hand_landmarker.task
models/pose_classifier.joblib
data/landmarks/            # training features (commit these)
data/images/               # local review only
data/videos/               # local review only
```

## Notes

- `scroll` and `shaka` stills must be right hand; `capture.py` refuses the wrong hand and `train_pose.py` skips mismatches.
- Pose ML classifies `scroll` / `shaka` / `idle`. Re-record stills after gesture changes, then re-run `train_pose.py`.
- Optional: `idle` video clips for review; the pose model trains on still JSONs.
