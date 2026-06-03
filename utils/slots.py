import json
import os


def save_slots(slots: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(slots, f, indent=2)


def load_slots(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    normalized = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        # Extract coordinates from points, polygon, or bbox
        points = item.get("points") or item.get("polygon") or item.get("bbox")
        if not points:
            continue

        # If it's a bounding box [x1, y1, x2, y2], convert it to a 4-point polygon
        if len(points) == 4 and not isinstance(points[0], list):
            x1, y1, x2, y2 = points
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        slot_id = item.get("id")
        if slot_id is None:
            slot_id = idx

        normalized.append({
            "id": int(slot_id),
            "polygon": points,
            "points": points  # provide both for backend-frontend compatibility
        })

    return normalized
