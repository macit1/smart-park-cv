import argparse
from utils.config import load_config, apply_overrides
from utils.video import get_video_path, extract_frame, open_video
from utils.logger import setup_logger, logger
from core.pipeline import run, run_frame


def parse_args():
    """Parse command-line arguments for the application.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Park_Sense — Parking lot occupancy detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--video",          default=None,  help="Path to video file (overrides config)")
    parser.add_argument("--config",         default="config.yaml", help="Path to config file")
    parser.add_argument("--confidence",     default=None,  type=float, help="Detection confidence threshold (e.g. 0.5)")
    parser.add_argument("--model",          default=None,  help="Path to YOLO model (e.g. models/yolov8m.pt)")
    parser.add_argument("--frame",          default=None,  type=int,   help="Extract and analyze a single frame by number")
    parser.add_argument("--save",           default=None,  help="Save output to file (image for --frame, video otherwise)")
    parser.add_argument("--save-frames",    default=None,  help="Directory to save detected frames as images")
    parser.add_argument("--out-fps",        default=None,  type=float, help="Output video FPS (e.g., 5.0 for faster playback)")
    parser.add_argument("--no-display",     action="store_true", help="Do not open a window (useful when only saving)")
    parser.add_argument("--frame-interval", default=None,  type=int,  help="Run detection every N frames (e.g. 5)")
    parser.add_argument("--max-frames",     default=None,  type=int,  help="Stop after processing N frames (e.g. 100)")
    parser.add_argument("--start-sec",      default=None,  type=int,  help="Start processing at this second (e.g. 90)")
    parser.add_argument("--end-sec",        default=None,  type=int,  help="Stop processing at this second (e.g. 120)")
    parser.add_argument("--debug",          action="store_true", help="Enable debug logging")
    parser.add_argument("--discover-slots",   action="store_true", help="Open web-based slot configuration editor, save slots.json")
    parser.add_argument("--slot-frame",       default=None, type=int, help="Frame number to use for slot configuration (overrides config)")
    parser.add_argument("--discovery-image",  default=None, help="Use this image directly for slot configuration (skips video frame extraction)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the global logger instance
    setup_logger(debug_mode=args.debug)

    # Load default configuration and override with CLI arguments
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    # Resolve the correct video source
    video_path = get_video_path(args, config)

    # Route to single frame processing or full video pipeline
    if args.discover_slots:
        import cv2 as _cv2
        from core.slot_server import run_editor
        from utils.slots import load_slots
        import os as _os

        slots_path = config["parking"].get("slots_path", "outputs/slots.json")

        if args.discovery_image:
            frame_img_path = args.discovery_image
            logger.info(f"Using discovery image: {frame_img_path}")
        else:
            frame_num = args.slot_frame if args.slot_frame is not None else config["parking"].get("discovery_frame", 0)
            frame_img_path = "outputs/discovery_frame.jpg"
            cap = open_video(video_path)
            frame = extract_frame(cap, frame_num)
            cap.release()
            _os.makedirs("outputs", exist_ok=True)
            _cv2.imwrite(frame_img_path, frame)
            logger.info(f"Saved discovery frame {frame_num} → {frame_img_path}")

        # Load existing slots if available, otherwise start fresh
        if _os.path.exists(slots_path):
            slots = load_slots(slots_path)
            logger.info(f"Loaded {len(slots)} existing slot(s) from {slots_path}")
        else:
            slots = []
            logger.info(f"No existing slots file found at {slots_path}. Starting with empty slots.")

        run_editor(frame_img_path, slots, slots_path)

    elif args.frame is not None:
        run_frame(video_path, args.frame, config, save_path=args.save, no_display=args.no_display)
    else:
        run(video_path, config, save_path=args.save, save_frames_dir=args.save_frames, out_fps=args.out_fps, start_sec=args.start_sec, end_sec=args.end_sec)
