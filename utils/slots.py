import json
import os


def save_slots(slots: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(slots, f, indent=2)


def load_slots(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
