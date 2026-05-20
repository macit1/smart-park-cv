import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args) -> dict:
    """Overwrite config values with any CLI args the user explicitly passed."""
    if args.confidence is not None:
        config["model"]["confidence"] = args.confidence
    if args.model is not None:
        config["model"]["path"] = args.model
    if getattr(args, "frame_interval", None) is not None:
        config["video"]["frame_interval"] = args.frame_interval
    if getattr(args, "max_frames", None) is not None:
        config["video"]["max_frames"] = args.max_frames
    return config
