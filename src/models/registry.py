from pathlib import Path

from src.models.llm.model import build_llm_model
from src.models.mock.model import MockResponseModel
from src.models.profile import load_profile


def load_response_model(profile_name: str, profiles_dir: Path) -> tuple[str, object]:
    name, model_type, params = load_profile(profile_name, profiles_dir)
    if model_type == "mock":
        return name, MockResponseModel()
    if model_type == "llm":
        required = {"model_name", "api_key_env", "base_url"}
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"llm profile missing required fields: {missing}")
        if not params.get("model_name") or not params.get("api_key_env"):
            raise ValueError("llm profile requires non-empty model_name and api_key_env")
        kwargs = {k: v for k, v in params.items() if k not in required}
        return name, build_llm_model(
            model_name=params["model_name"],
            api_key_env=params["api_key_env"],
            base_url=params["base_url"],
            **kwargs,
        )
    raise ValueError(f"unknown model type: {model_type}")
