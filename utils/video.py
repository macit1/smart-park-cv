import cv2


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def extract_frame(cap: cv2.VideoCapture, frame_number: int):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total:
        raise ValueError(f"Frame {frame_number} out of range — video has {total} frames (0 to {total - 1})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number}")
    return frame


def get_display_size(config: dict) -> tuple[int, int]:
    return config["video"]["display_width"], config["video"]["display_height"]


def get_video_path(args, config: dict) -> str:
    return args.video or config["video"]["source"]
