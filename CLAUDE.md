# Park_Sense — Claude Context

## Project

Parking lot occupancy detection system built with YOLOv8.
Given a video, detect vehicles → discover parking slots → classify occupied/empty → show live counter.

Phase-based development. Each phase builds on the previous one.

```
Park_Sense/
├── main.py                  # Entry point — CLI args → pipeline
├── config.yaml              # All settings (model, video, thresholds)
├── core/
│   ├── detector.py          # VehicleDetector — YOLOv8 inference
│   └── pipeline.py          # run() full video / run_frame() single frame
├── utils/
│   ├── config.py            # load_config(), apply_overrides()
│   ├── video.py             # open_video(), extract_frame()
│   └── visualizer.py        # draw_detections(), draw_stats()
├── videos/                  # Input videos
├── outputs/                 # Saved frames/videos
└── models/                  # YOLOv8 weights (auto-downloaded)
```

## Phase Status

| Phase | What | Status |
|-------|------|--------|
| 1 | YOLOv8 vehicle detection, bbox overlay, full CLI | ✅ Done |
| 2 | Auto-discover parking slots from first N frames | ⬜ Next |
| 3 | Red/green overlay + occupied/total counter | ⬜ Planned |
| 4 | Custom model training on PKLot dataset | ⬜ Future |
| 5 | Bird's eye view, perspective transform | ⬜ Future |

## Run Commands

```bash
# Full video
python main.py --video videos/test.mp4

# Single frame
python main.py --video videos/test.mp4 --frame 10

# Save output, no display
python main.py --video videos/test.mp4 --save outputs/result.mp4 --no-display

# Quick test (first 100 frames)
python main.py --video videos/test.mp4 --max-frames 100 --no-display
```

## Stack

- Python, YOLOv8 (ultralytics), OpenCV
- Config-driven via `config.yaml` — no hardcoded values
- COCO classes: 2=car, 5=bus, 7=truck
- Default model: `yolov8n` (nano — speed over accuracy)

## How I Work — Follow These

**MVP first, always.**
- Start with the smallest working version. Don't build the full solution upfront.
- Validate on 10–20 frames before running on full video.
- If I don't say how many frames to test on, ask.

**Plan before code.**
- For any non-trivial change: show the approach first, wait for approval, then code.
- Keep it short — a few bullet points is enough. No essays.

**Clean code.**
- DRY — no copy-paste logic. Extract to a function/class.
- SOLID — single responsibility per class/function.
- Config in `config.yaml`, never hardcoded in logic files.
- Meaningful names. No `x`, `tmp`, `data2`.

**No over-engineering.**
- Don't add abstractions I didn't ask for.
- Don't refactor things that aren't broken.
- Don't add logging/error handling beyond what's needed.

**Iteration style.**
- Make one change at a time. Don't rewrite the whole file.
- If something is broken, fix that first before adding new features.

**Code presentation.**
1. Short explanation of approach
2. The code
3. Brief note on critical logic (if needed)
- Skip the fluff. Don't explain what `import cv2` does.

## CV-Specific Notes

- IoU will be needed in Phase 2 for slot-vehicle overlap checks
- Frame interval (`--frame-interval`) exists to avoid CPU bottleneck — default 5
- `__NEXT_DATA__` pattern (used in other projects) — parse structured JSON over fragile HTML
- For new detection logic: test on a single frame first, then integrate into pipeline
