# Hands-Free Scrolling

CSE 455 project: control scrolling and window switching with hand gestures from a laptop webcam.

## What it does

| Gesture | Pose | Action |
|---|---|---|
| `scroll_up` | Index + middle out | Tips move **down** → scroll up (opposite ignored) |
| `scroll_down` | Index out | Tip moves **up** → scroll down (opposite ignored) |
| `switch` | Middle + ring + pinky out | Alt+Tab (Windows) / Cmd+Tab (Mac), once with cooldown |
| `reject` | Idle / wrong poses | Do nothing |

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
| `1` | label `scroll_up` (index + middle out, stills) |
| `2` | label `scroll_down` (index out, stills) |
| `3` | label `switch` (middle + ring + pinky out, stills) |
| `0` | label `reject` (stills + video) |
| `s` | save still (scroll / switch / reject) |
| `r` | start/stop video (`reject` only) |
| `q` | quit |

Saves:

- Stills → `data/images/<label>/` + `data/landmarks/<label>/<label>_N.json`
- Videos → `data/videos/<label>/` + matching landmark sequence JSON

Only the **landmark JSON** is used for training. Images/videos are for review (gitignored).

## Train the pose classifier

```bash
python train_pose.py
```

Reads still JSONs under `data/landmarks/{scroll_up,scroll_down,switch,reject}/`, trains a Linear SVM, writes `models/pose_classifier.joblib`.

Collect some `reject` stills (near-misses, wrong finger combos) so it learns when not to scroll or switch.

## Run the live demo

```bash
python runtime.py
```

Focus a scrollable window, then use the gestures. Press `q` in the preview window to quit. On Mac, grant Accessibility permission to Terminal/Python if keyboard/mouse control is blocked.

## Project layout

```
capture.py                 # data collection
train_pose.py              # train pose model from landmark JSON
runtime.py                 # live MediaPipe + classifier + scroll/Alt+Tab
example-runtime-script.py  # short pseudocode of the control loop
models/hand_landmarker.task
models/pose_classifier.joblib
data/landmarks/            # training features (commit these)
data/images/               # local review only
data/videos/               # local review only
```

## Notes

- Prefer **right hand** for positive gestures; put left hand / wrong poses in `reject`.
- Pose ML classifies `scroll_up` / `scroll_down` / `switch` / `reject`. Re-record stills after gesture changes, then re-run `train_pose.py`.
- Optional: `reject` video clips for review; the pose model trains on still JSONs.
