# Park_Sense

A parking lot occupancy detection system built with YOLOv8.
Built step by step — each phase adds a new capability on top of the previous one.

---

## What This Project Does (End Goal)

Given a video of a parking lot, the system will:
1. Detect all vehicles in the scene
2. Automatically figure out where the parking slots are
3. Tell you which slots are **occupied** (red) and which are **empty** (green)
4. Show a live counter: `Occupied: X / Total: Y`

---

## Current State — Phase 1 (Vehicle Detection Only)

Right now the project can:
- Read a video file frame by frame
- Detect cars, buses, and trucks using YOLOv8
- Draw a green bounding box around each detected vehicle
- Show vehicle count and FPS in the top-left corner
- Extract and analyze a single frame by number
- Save output to a file (image or video)

It does **not** know about parking slots yet — that comes in Phase 2.

---

## Project Structure

```
Park_Sense/
├── main.py                  # Entry point — parse args, load config, start pipeline
├── config.yaml              # All settings (model, video, thresholds)
├── requirements.txt         # Python dependencies
├── core/
│   ├── detector.py          # VehicleDetector class — YOLOv8 inference
│   ├── pipeline.py          # run() for full video, run_frame() for single frame
│   └── slot_server.py       # Web-based slot configuration editor (Flask)
├── utils/
│   ├── config.py            # load_config(), apply_overrides()
│   ├── video.py             # open_video(), extract_frame(), get_display_size()
│   └── visualizer.py        # draw_detections(), draw_stats()
├── videos/                  # Put your test video here
├── outputs/                 # Saved frames and videos land here
└── models/                  # YOLOv8 weights (auto-downloaded on first run)
```

---

## File Explanations

### `main.py`
Entry point only — parses CLI args, applies overrides to config, then routes to either `run()` or `run_frame()`.

### `core/detector.py`
A class called `VehicleDetector`. You give it a frame (image), it gives back a list of detections:
```python
[
  {"bbox": [x1, y1, x2, y2], "confidence": 0.87, "class": 2},
  ...
]
```
- `bbox` — rectangle around the vehicle (top-left + bottom-right in pixels)
- `confidence` — how sure the model is (0 to 1)
- `class` — 2=car, 5=bus, 7=truck

### `core/pipeline.py`
Two modes:
- `run()` — full video loop, optional video save
- `run_frame()` — jumps to a specific frame, runs detection, shows and/or saves it

### `core/slot_server.py`
A local Flask server providing an interactive SVG-overlay web interface for drawing, editing, moving, and deleting polygonal parking slots directly over a video frame. Saves coordinates to `outputs/slots.json`.

### `utils/config.py`
- `load_config()` — reads `config.yaml` into a dict
- `apply_overrides()` — overwrites config values with any CLI flags the user passed

### `utils/video.py`
- `open_video()` — opens the file, raises a clear error if not found
- `extract_frame()` — seeks to a specific frame number, validates range
- `get_display_size()` — pulls width/height from config
- `get_video_path()` — resolves `--video` arg or falls back to config

### `utils/visualizer.py`
- `draw_detections()` — draws green boxes + labels on each vehicle
- `draw_stats()` — draws vehicle count and FPS in the top-left corner

### `config.yaml`
Central settings. Change behavior here without touching the code:
```yaml
model:
  path: models/yolov8n.pt   # "n" = nano, smallest/fastest YOLOv8 variant
  confidence: 0.4           # detections below 40% confidence are ignored
  classes: [2, 5, 7]        # COCO IDs: 2=car, 5=bus, 7=truck

video:
  source: videos/test.mp4
  display_width: 1280
  display_height: 720
```

---

## CLI Reference

All flags are optional. Config values are used as defaults — CLI flags override them.

| Flag | Type | Description |
|---|---|---|
| `--video` | path | Video file to process |
| `--config` | path | Config file (default: `config.yaml`) |
| `--confidence` | float | Detection threshold, e.g. `0.6` |
| `--model` | path | YOLO model to use, e.g. `models/yolov8m.pt` |
| `--frame` | int | Extract and analyze a single frame by number |
| `--save` | path | Save output — image if `--frame` used, video otherwise |
| `--no-display` | flag | Skip opening a window (useful when only saving) |
| `--frame-interval` | int | Run detection every N frames (default: 5, set to 1 for every frame) |
| `--max-frames` | int | Stop after N frames — useful for quick tests (e.g. `100`) |
| `--discover-slots` | flag | Open web-based slot configuration editor, save `slots.json` |
| `--slot-frame` | int | Frame number to use for slot configuration (overrides config) |
| `--discovery-image` | path | Use this image directly for slot configuration (skips video frame extraction) |
| `--start-sec` | int | Start processing at this second (e.g. `90`) |
| `--end-sec` | int | Stop processing at this second (e.g. `120`) |
| `--out-fps` | float | Output video FPS (e.g. `5.0` for faster playback) |
| `--save-frames` | path | Directory to save detected frames as individual images |

### Examples

```bash
# Run full video
python main.py --video videos/test.mp4

# Show frame 10
python main.py --video videos/test.mp4 --frame 10

# Show frame 10 and save it
python main.py --video videos/test.mp4 --frame 10 --save outputs/frame10.jpg

# Save frame without opening a window
python main.py --video videos/test.mp4 --frame 10 --save outputs/frame10.jpg --no-display

# Run full video and save the result
python main.py --video videos/test.mp4 --save outputs/result.mp4

# Override confidence and model
python main.py --video videos/test.mp4 --confidence 0.6 --model models/yolov8m.pt

# Quick test — only process first 100 frames
python main.py --video videos/test.mp4 --max-frames 100 --save outputs/test_run.mp4 --no-display

# Run detection every 10 frames
python main.py --video videos/test.mp4 --frame-interval 10 --save outputs/result.mp4 --no-display

# Run on every single frame (slowest, most accurate)
python main.py --video videos/test.mp4 --frame-interval 1 --save outputs/result.mp4 --no-display

# Combine anything
python main.py --video videos/test.mp4 --frame 50 --confidence 0.3 --save outputs/frame50.jpg
```

---

## How to Install

```bash
pip install -r requirements.txt
```

---

## Quick Run

Run full video, save result, no popup window:

```bash
python main.py --video videos/test.mp4 --save outputs/result_final.mp4 --no-display
```

---

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | Detect vehicles with YOLOv8, draw bounding boxes, full CLI | ✅ Done |
| 2 | Manual slot configuration via web-based interactive editor | ⬜ Next |
| 3 | Color-coded overlay (red=occupied, green=empty) + counter | ⬜ Planned |
| 4 | Geometric corrections (bird's eye view, perspective transform) | ⬜ Future |

---

## System Workflow

> See `workflow.html` for the full visual diagrams — just double-click to open in your browser.

---

## Key Concepts

**YOLOv8 (You Only Look Once v8)**
A real-time object detection model. We use the `nano` variant (`yolov8n`) — the fastest, good enough for a parking camera feed.

**COCO Classes**
YOLOv8 is pre-trained on the COCO dataset (80 object categories). We filter to classes 2, 5, 7 (car, bus, truck) so people, bikes, etc. are ignored.

**Confidence Threshold**
Every detection has a confidence score. Below `0.4` (40%) it's discarded to reduce false positives.

**Bounding Box (bbox)**
A rectangle defined by two points: top-left `(x1, y1)` and bottom-right `(x2, y2)`, in pixel coordinates.

**IoU (Intersection over Union)**
A measure of how much two rectangles overlap — used in Phase 2 to check if a detected vehicle is inside a parking slot.
