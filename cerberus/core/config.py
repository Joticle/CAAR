"""
Cerberus Configuration Loader
Reads cerberus.yaml and provides typed, validated access to all config values.
Singleton pattern ensures one config instance across all subsystems.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml


logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "cerberus.yaml"
)


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class CerberusConfig:
    """
    Singleton configuration manager for Cerberus.
    Loads YAML config, validates required sections, and provides
    safe access to nested values with defaults.
    """

    _instance: Optional["CerberusConfig"] = None
    _initialized: bool = False

    REQUIRED_SECTIONS: list[str] = [
        "system", "database", "i2c", "mqtt", "safety",
        "power", "motors", "servos", "navigation", "camera",
        "audio", "status_leds", "sensors", "intelligence",
        "heads", "mission", "network", "health"
    ]

    def __new__(cls, config_path: Optional[str] = None) -> "CerberusConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        if CerberusConfig._initialized:
            return

        self._config_path: str = config_path or _DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self._load()
        self._validate()
        self._resolve_paths()

        CerberusConfig._initialized = True
        logger.info("Configuration loaded from %s", self._config_path)

    def _load(self) -> None:
        """Load YAML config file from disk."""
        path: Path = Path(self._config_path)

        if not path.exists():
            raise ConfigError(f"Config file not found: {self._config_path}")

        if not path.is_file():
            raise ConfigError(f"Config path is not a file: {self._config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw: Any = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parse error in {self._config_path}: {e}") from e
        except OSError as e:
            raise ConfigError(f"Cannot read config file {self._config_path}: {e}") from e

        if not isinstance(raw, dict):
            raise ConfigError("Config file must contain a YAML mapping at the root level")

        self._data = raw

    def _validate(self) -> None:
        """Ensure all required top-level sections exist."""
        missing: list[str] = [
            section for section in self.REQUIRED_SECTIONS
            if section not in self._data
        ]
        if missing:
            raise ConfigError(f"Missing required config sections: {', '.join(missing)}")

    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute based on project root."""
        project_root: str = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))

        path_keys: list[tuple[str, ...]] = [
            ("system", "log_dir"),
            ("database", "path"),
            ("intelligence", "model_dir"),
            ("camera", "capture_dir"),
            ("audio", "audio_dir"),
        ]

        for key_chain in path_keys:
            value: Optional[str] = self.get(*key_chain)
            if value and not os.path.isabs(value):
                resolved: str = os.path.join(project_root, value)
                self._set_nested(key_chain, resolved)

    def _set_nested(self, keys: tuple[str, ...], value: Any) -> None:
        """Set a value in the nested config dict by key chain."""
        target: dict[str, Any] = self._data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safe access to nested config values.

        Usage:
            config.get("mqtt", "broker_host")
            config.get("safety", "thermal_warning_c", default=75)
        """
        current: Any = self._data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def require(self, *keys: str) -> Any:
        """
        Access a nested config value that must exist.
        Raises ConfigError if the value is missing.
        """
        value: Any = self.get(*keys)
        if value is None:
            path: str = " → ".join(keys)
            raise ConfigError(f"Required config value missing: {path}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        """Return an entire config section as a dict."""
        data: Any = self._data.get(name)
        if data is None:
            raise ConfigError(f"Config section not found: {name}")
        if not isinstance(data, dict):
            raise ConfigError(f"Config section '{name}' is not a mapping")
        return data

    @property
    def system(self) -> dict[str, Any]:
        return self.section("system")

    @property
    def database(self) -> dict[str, Any]:
        return self.section("database")

    @property
    def i2c(self) -> dict[str, Any]:
        return self.section("i2c")

    @property
    def mqtt(self) -> dict[str, Any]:
        return self.section("mqtt")

    @property
    def safety(self) -> dict[str, Any]:
        return self.section("safety")

    @property
    def power(self) -> dict[str, Any]:
        return self.section("power")

    @property
    def motors(self) -> dict[str, Any]:
        return self.section("motors")

    @property
    def servos(self) -> dict[str, Any]:
        return self.section("servos")

    @property
    def navigation(self) -> dict[str, Any]:
        return self.section("navigation")

    @property
    def camera(self) -> dict[str, Any]:
        return self.section("camera")

    @property
    def audio(self) -> dict[str, Any]:
        return self.section("audio")

    @property
    def status_leds(self) -> dict[str, Any]:
        return self.section("status_leds")

    @property
    def sensors(self) -> dict[str, Any]:
        return self.section("sensors")

    @property
    def intelligence(self) -> dict[str, Any]:
        return self.section("intelligence")

    @property
    def heads(self) -> dict[str, Any]:
        return self.section("heads")

    @property
    def mission(self) -> dict[str, Any]:
        return self.section("mission")

    @property
    def network(self) -> dict[str, Any]:
        return self.section("network")

    @property
    def health(self) -> dict[str, Any]:
        return self.section("health")

    @property
    def config_path(self) -> str:
        return self._config_path

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing. Not for production use."""
        cls._instance = None
        cls._initialized = False

    def __repr__(self) -> str:
        return f"CerberusConfig(path='{self._config_path}', sections={list(self._data.keys())})"