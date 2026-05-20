import cv2
import time
import os
from core.detector import VehicleDetector
from utils.video import open_video, get_display_size, extract_frame
from utils.visualizer import draw_detections, draw_stats
from utils.logger import logger


def _make_detector(config: dict) -> VehicleDetector:
    """Helper function to instantiate a VehicleDetector from the configuration.

    Args:
        config (dict): The loaded configuration dictionary.

    Returns:
        VehicleDetector: Initialized detector instance.
    """
    return VehicleDetector(
        model_path=config["model"]["path"],
        confidence=config["model"]["confidence"],
        classes=config["model"]["classes"],
        imgsz=config["model"].get("imgsz", 640),
    )


def run(video_path: str, config: dict, save_path: str = None, save_frames_dir: str = None, out_fps: float = None, start_sec: int = None, end_sec: int = None) -> None:
    """Main pipeline to process a video, detect vehicles, and optionally save the output.

    Args:
        video_path (str): Path to the input video file.
        config (dict): Configuration dictionary containing model and video settings.
        save_path (str, optional): Path to save the processed output video.
        save_frames_dir (str, optional): Directory to save individual detected frames as images.
        out_fps (float, optional): Custom FPS for the output video playback.
        start_sec (int, optional): Second to start processing from.
        end_sec (int, optional): Second to stop processing at.
    """
    detector = _make_detector(config)
    cap = open_video(video_path)
    w, h = get_display_size(config)

    # Resolve total frames based on video length or config limit
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = config["video"].get("max_frames", None)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_interval = config["video"].get("frame_interval", 1)

    writer = None
    if save_path:
        # Adjust output FPS to maintain real-world playback speed when frames are skipped
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        final_fps = out_fps if out_fps is not None else max(1.0, fps_src / frame_interval)
        writer = cv2.VideoWriter(save_path, fourcc, final_fps, (w, h))
        logger.info(f"Saving video to: {save_path} at {final_fps:.1f} FPS")

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        logger.info(f"Saving detected frames to directory: {save_frames_dir}")

    logger.info(f"Total frames: {total_frames} | FPS: {fps_src:.1f} | Frame interval: {frame_interval}")
    
    # Seek to the requested start time if provided
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

    # Main processing loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame only if it matches the frame interval to save compute
            if frame_idx % frame_interval == 0:
                detections = detector.detect(frame)
                logger.debug(f"Frame {frame_idx}: ran inference, found {len(detections)} vehicle(s)")

                draw_detections(frame, detections)
                draw_stats(frame, frame_idx, total_frames, len(detections))

                # Resize frame for display/saving *after* drawing to prevent bounding box distortion
                frame = cv2.resize(frame, (w, h))

                # Save frame to output video
                if writer:
                    writer.write(frame)
                
                # Save detected frame as an individual image
                if save_frames_dir:
                    frame_path = os.path.join(save_frames_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)

            # Stop processing if maximum frames reached
            frames_processed = frame_idx - start_frame
            if max_frames and frames_processed >= max_frames - 1:
                break
                
            # Stop processing if end time reached
            if end_frame and frame_idx >= end_frame:
                logger.info(f"Reached end time ({end_sec}s). Stopping.")
                break

            # Log progress periodically
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
    """Extract a single frame from the video, run detection, and optionally display or save it.

    Args:
        video_path (str): Path to the input video file.
        frame_number (int): The specific frame index to extract.
        config (dict): Configuration dictionary.
        save_path (str, optional): Path to save the processed image.
        no_display (bool, optional): If True, suppresses the OpenCV display window.
    """
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
