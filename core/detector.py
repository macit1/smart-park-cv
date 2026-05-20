from ultralytics import YOLO


class VehicleDetector:
    """Detects vehicles in a frame using YOLOv8."""

    CLASS_NAMES = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str, confidence: float, classes: list, imgsz: int = 640):
        self.model = YOLO(model_path)  # auto-downloads if not found locally
        self.confidence = confidence
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame) -> list[dict]:
        """
        Run inference on a single frame.
        Returns: [{"bbox": [x1,y1,x2,y2], "confidence": float, "class": int}, ...]
        """
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
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]),
            })

        return detections
