from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load model profiles.") from exc


def load_profile(name: str, profiles_dir: Path) -> Tuple[str, str, Dict[str, Any]]:
    path = profiles_dir / f"{name}.yaml"
    data = _read_yaml(path)
    profile_name = data.get("name") or name
    model_type = data.get("type")
    if not profile_name or not model_type:
        raise ValueError(f"profile must include name and type: {path}")
    params = {k: v for k, v in data.items() if k not in {"name", "type"}}
    return profile_name, model_type, params


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model profile not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid profile format: {path}")
    return data
