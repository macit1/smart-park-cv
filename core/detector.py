from ultralytics import YOLO
from utils.logger import logger


class VehicleDetector:
    """Detects vehicles in a frame using YOLOv8."""

    def __init__(self, model_path: str, confidence: float, classes: list, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame) -> list[dict]:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_name = self.model.names.get(class_id, "unknown")
            conf = float(box.conf[0])

            logger.debug(f"Detected {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": class_id,
                "class_name": class_name,
            })

        return detections
