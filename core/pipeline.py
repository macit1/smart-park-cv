import cv2
import time
from core.detector import VehicleDetector
from utils.video import open_video, get_display_size, extract_frame
from utils.visualizer import draw_detections, draw_stats


def _make_detector(config: dict) -> VehicleDetector:
    return VehicleDetector(
        model_path=config["model"]["path"],
        confidence=config["model"]["confidence"],
        classes=config["model"]["classes"],
        imgsz=config["model"].get("imgsz", 640),
    )


def run(video_path: str, config: dict, save_path: str = None) -> None:
    """Run detection on every frame, log to console, optionally save output video."""
    detector = _make_detector(config)
    cap = open_video(video_path)
    w, h = get_display_size(config)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = config["video"].get("max_frames", None)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))
        print(f"Saving to: {save_path}")

    print(f"Total frames: {total_frames} | FPS: {fps_src:.1f}")
    print("Processing...")

    frame_idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        detections = detector.detect(frame)

        draw_detections(frame, detections)
        draw_stats(frame, frame_idx, total_frames, len(detections))

        if writer:
            writer.write(frame)

        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % 30 == 0:
            elapsed = time.time() - start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} | Detected: {len(detections)} | Speed: {fps_proc:.1f} fps")

        frame_idx += 1

    elapsed = time.time() - start
    print(f"Done. {frame_idx} frames in {elapsed:.1f}s")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {save_path}")


def run_frame(video_path: str, frame_number: int, config: dict, save_path: str = None, no_display: bool = False) -> None:
    """Extract a single frame, run detection on it, show and/or save it."""
    detector = _make_detector(config)
    cap = open_video(video_path)
    w, h = get_display_size(config)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame = extract_frame(cap, frame_number)
    cap.release()

    frame = cv2.resize(frame, (w, h))
    detections = detector.detect(frame)
    draw_detections(frame, detections)
    draw_stats(frame, frame_number, total_frames, len(detections))

    print(f"Frame {frame_number}/{total_frames} | Detected: {len(detections)} vehicle(s)")

    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"Saved: {save_path}")

    if not no_display:
        cv2.imshow(f"Park_Sense — Frame {frame_number}", frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
