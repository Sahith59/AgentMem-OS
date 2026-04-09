import os
import yaml
import logging

class StorageManager:
    """
    Manages base paths for the MemNAI platform.
    """
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.base_path = os.path.expanduser(self.config.get("storage", {}).get("base_path", "~/.memnai/"))
        self.fallback_path = os.path.expanduser(self.config.get("storage", {}).get("fallback_path", "~/.memnai_fallback/"))
        self.warn_on_fallback = self.config.get("storage", {}).get("warn_on_fallback", True)
        self.active_path = self._resolve_active_path()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_active_path(self):
        if self.base_path == os.path.expanduser("~/.memnai/"):
            return self.base_path

        if os.path.exists(self.base_path):
            return self.base_path
        else:
            if self.warn_on_fallback:
                logging.warning(f"StorageManager: Primary path {self.base_path} is unreachable. Falling back to {self.fallback_path}.")
            return self.fallback_path

    def get_path(self, layer_name: str) -> str:
        target_dir = os.path.join(self.active_path, layer_name)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def is_fallback_active(self) -> bool:
        return self.active_path == self.fallback_path
