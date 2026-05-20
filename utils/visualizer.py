import cv2

CLASS_NAMES = {2: "car", 5: "bus", 7: "truck"}


def draw_detections(frame, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{CLASS_NAMES.get(det['class'], '?')} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def draw_stats(frame, frame_idx: int, total_frames: int, detected: int) -> None:
    lines = [
        f"Frame:    {frame_idx} / {total_frames}",
        f"Detected: {detected} vehicle(s)",
    ]
    cv2.rectangle(frame, (0, 0), (280, 55), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
