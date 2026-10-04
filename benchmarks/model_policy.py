"""Forward-only inference policy; historical artifacts and offline replay are valid.

No provider imports, aliases, model substitutions, or account calls live here.
Opaque deployment names cannot be resolved locally; configure their actual model
explicitly before treating them as an approved experiment.
"""
import re


class RetiredModelError(ValueError):
    """An explicitly retired model was requested for new inference."""


def require_active_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An explicit non-retired inference model is required")
    # Provider routes (openai/, azure/, openai:) and fine-tune prefixes may
    # precede the underlying name. Match family boundaries, not gpt-4.1.
    if re.search(r"(?:^|[/ :])(?:gpt|chatgpt)-4o(?:$|[-/:])", model.strip().lower()):
        raise RetiredModelError(
            "GPT-4o family is retired for new inference. Preserve historical "
            "results; explicitly configure and validate a non-retired model. "
            "No automatic replacement or judge promotion is permitted.")
    return model


def require_active_package(package: dict) -> None:
    """Check all frozen role settings before any paid job is started."""
    for settings in package["settings"].values():
        require_active_model(settings["model"])
