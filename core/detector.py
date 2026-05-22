from ultralytics import YOLO
from utils.logger import logger


class VehicleDetector:
    """Detects vehicles in a frame using a YOLO model.
    """

    def __init__(self, model_path: str, confidence: float, classes: list, imgsz: int = 640):
        """Initialize the detector.

        Args:
            model_path (str): Path to the YOLO weights file.
            confidence (float): Minimum confidence threshold for detections.
            classes (list): List of class IDs to detect (e.g., [2, 5, 7] for car, bus, truck).
            imgsz (int, optional): Inference image size. Defaults to 640.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame) -> list[dict]:
        """Run inference on a single frame to detect vehicles.

        Args:
            frame: OpenCV image frame.

        Returns:
            list[dict]: List of detection dictionaries containing 'bbox', 'confidence', 'class_id', and 'class_name'.
        """
        # Run YOLO inference with verbose disabled to keep console clean
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        # Extract and format YOLO detection results
        detections = []
        for box in results.boxes:
            # Get integer coordinates for bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Map class ID to human-readable name using model's internal dictionary
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
