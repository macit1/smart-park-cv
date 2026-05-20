import cv2
import time
import os
from core.detector import VehicleDetector
from utils.video import open_video, get_display_size, extract_frame
from utils.visualizer import draw_detections, draw_stats
from utils.logger import logger


def _make_detector(config: dict) -> VehicleDetector:
    return VehicleDetector(
        model_path=config["model"]["path"],
        confidence=config["model"]["confidence"],
        classes=config["model"]["classes"],
        imgsz=config["model"].get("imgsz", 640),
    )


def run(video_path: str, config: dict, save_path: str = None, save_frames_dir: str = None, out_fps: float = None, start_sec: int = None, end_sec: int = None) -> None:
    detector = _make_detector(config)
    cap = open_video(video_path)
    w, h = get_display_size(config)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = config["video"].get("max_frames", None)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_interval = config["video"].get("frame_interval", 1)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        final_fps = out_fps if out_fps is not None else max(1.0, fps_src / frame_interval)
        writer = cv2.VideoWriter(save_path, fourcc, final_fps, (w, h))
        logger.info(f"Saving video to: {save_path} at {final_fps:.1f} FPS")

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        logger.info(f"Saving detected frames to directory: {save_frames_dir}")

    logger.info(f"Total frames: {total_frames} | FPS: {fps_src:.1f} | Frame interval: {frame_interval}")
    
    start_frame = 0
    if start_sec is not None:
        start_frame = int(start_sec * fps_src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        logger.info(f"Skipping to {start_sec}s (Frame {start_frame})")

    end_frame = None
    if end_sec is not None:
        end_frame = int(end_sec * fps_src)
        logger.info(f"Will stop at {end_sec}s (Frame {end_frame})")

    logger.info("Processing...")

    frame_idx = start_frame
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                detections = detector.detect(frame)
                logger.debug(f"Frame {frame_idx}: ran inference, found {len(detections)} vehicle(s)")

                draw_detections(frame, detections)
                draw_stats(frame, frame_idx, total_frames, len(detections))

                frame = cv2.resize(frame, (w, h))

                if writer:
                    writer.write(frame)
                
                if save_frames_dir:
                    frame_path = os.path.join(save_frames_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)

            frames_processed = frame_idx - start_frame
            if max_frames and frames_processed >= max_frames - 1:
                break
                
            if end_frame and frame_idx >= end_frame:
                logger.info(f"Reached end time ({end_sec}s). Stopping.")
                break

            if frame_idx % (frame_interval * 10) == 0 and frame_idx > start_frame:
                elapsed = time.time() - start
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_idx}/{total_frames} | Speed: {fps_proc:.1f} fps")

            frame_idx += 1
    finally:
        elapsed = time.time() - start
        logger.info(f"Done. {frame_idx} frames in {elapsed:.1f}s")
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved: {save_path}")


def run_frame(video_path: str, frame_number: int, config: dict, save_path: str = None, no_display: bool = False) -> None:
    detector = _make_detector(config)
    cap = open_video(video_path)
    w, h = get_display_size(config)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame = extract_frame(cap, frame_number)
    cap.release()

    detections = detector.detect(frame)
    draw_detections(frame, detections)
    draw_stats(frame, frame_number, total_frames, len(detections))

    frame = cv2.resize(frame, (w, h))

    logger.info(f"Frame {frame_number}/{total_frames} | Detected: {len(detections)} vehicle(s)")

    if save_path:
        cv2.imwrite(save_path, frame)
        logger.info(f"Saved: {save_path}")

    if not no_display:
        cv2.imshow(f"Park_Sense — Frame {frame_number}", frame)
        logger.info("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
