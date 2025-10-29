import configparser
import os
import threading
import time
from typing import Dict

class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str = None, reload_interval: int = 30):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(config_path, reload_interval)
        return cls._instance

    def _init(self, config_path: str, reload_interval: int):
        # 默认路径为 ../config/config.ini
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "..", "config", "config.ini")
        self.config_path = os.path.abspath(config_path)
        self.reload_interval = reload_interval
        self._last_mtime = 0
        self._config = configparser.ConfigParser()
        self._cache: Dict[str, Dict[str, str]] = {}

        self._load()
        # 后台线程监控文件变化
        t = threading.Thread(target=self._watch_config, daemon=True)
        t.start()

    def _load(self):
        if not os.path.exists(self.config_path):
            return
        mtime = os.path.getmtime(self.config_path)
        if mtime != self._last_mtime:
            self._config.read(self.config_path, encoding="utf-8")
            self._last_mtime = mtime
            self._cache.clear()  # 清除缓存
            print(f"[ConfigManager] Reloaded config from {self.config_path}")
            print(f"[ConfigManager] Sections: {self._config.sections()}")

    def _watch_config(self):
        while True:
            time.sleep(self.reload_interval)
            self._load()

    def get_section(self, section_name: str) -> Dict[str, str]:
        # 先从缓存拿
        if section_name in self._cache:
            return self._cache[section_name]

        if section_name not in self._config:
            raise ValueError(
                f"Section '{section_name}' not found in {self.config_path}. "
                f"Available sections: {self._config.sections()}"
            )
        section_dict = dict(self._config[section_name])
        self._cache[section_name] = section_dict
        return section_dict
