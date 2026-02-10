"""
Tests for cerberus.core.config — CerberusConfig
Validates singleton behavior, YAML loading, nested access, defaults,
required values, section access, and error handling.
"""

import os
import pytest
from typing import Any

import yaml

from cerberus.core.config import CerberusConfig, ConfigError


class TestConfigLoading:
    """Config file loading and parsing."""

    def test_load_valid_config(self, config: CerberusConfig) -> None:
        assert config is not None
        assert config.get("system", "name") == "cerberus-test"

    def test_missing_file_raises(self, temp_dir: str) -> None:
        CerberusConfig.reset()
        fake_path: str = os.path.join(temp_dir, "nonexistent.yaml")
        with pytest.raises(ConfigError, match="not found"):
            CerberusConfig(fake_path)
        CerberusConfig.reset()

    def test_invalid_yaml_raises(self, temp_dir: str) -> None:
        CerberusConfig.reset()
        bad_yaml: str = os.path.join(temp_dir, "bad.yaml")
        with open(bad_yaml, "w") as f:
            f.write(":\n  bad: {{\n  invalid: yaml: content:")
        with pytest.raises(ConfigError, match="YAML parse error"):
            CerberusConfig(bad_yaml)
        CerberusConfig.reset()

    def test_non_dict_root_raises(self, temp_dir: str) -> None:
        CerberusConfig.reset()
        list_yaml: str = os.path.join(temp_dir, "list.yaml")
        with open(list_yaml, "w") as f:
            yaml.dump(["a", "b", "c"], f)
        with pytest.raises(ConfigError, match="YAML mapping"):
            CerberusConfig(list_yaml)
        CerberusConfig.reset()

    def test_missing_section_raises(self, temp_dir: str) -> None:
        CerberusConfig.reset()
        incomplete: dict[str, Any] = {"system": {"name": "test"}}
        path: str = os.path.join(temp_dir, "incomplete.yaml")
        with open(path, "w") as f:
            yaml.dump(incomplete, f)
        with pytest.raises(ConfigError, match="Missing required"):
            CerberusConfig(path)
        CerberusConfig.reset()


class TestSingleton:
    """Singleton pattern behavior."""

    def test_same_instance(self, config: CerberusConfig) -> None:
        second: CerberusConfig = CerberusConfig()
        assert config is second

    def test_reset_clears_singleton(self, config: CerberusConfig) -> None:
        instance_id: int = id(config)
        CerberusConfig.reset()
        assert CerberusConfig._instance is None
        assert not CerberusConfig._initialized


class TestNestedAccess:
    """Safe nested value access with get() and require()."""

    def test_get_top_level(self, config: CerberusConfig) -> None:
        system: Any = config.get("system")
        assert isinstance(system, dict)
        assert "name" in system

    def test_get_nested_value(self, config: CerberusConfig) -> None:
        name: Any = config.get("system", "name")
        assert name == "cerberus-test"

    def test_get_deep_nested(self, config: CerberusConfig) -> None:
        addr: Any = config.get("sensors", "bme680", "i2c_address")
        assert addr == "0x77"

    def test_get_missing_returns_default(self, config: CerberusConfig) -> None:
        value: Any = config.get("nonexistent", "key", default="fallback")
        assert value == "fallback"

    def test_get_missing_returns_none(self, config: CerberusConfig) -> None:
        value: Any = config.get("nonexistent", "key")
        assert value is None

    def test_require_existing_value(self, config: CerberusConfig) -> None:
        name: Any = config.require("system", "name")
        assert name == "cerberus-test"

    def test_require_missing_raises(self, config: CerberusConfig) -> None:
        with pytest.raises(ConfigError, match="Required config"):
            config.require("nonexistent", "value")


class TestSectionAccess:
    """Section properties and section() method."""

    def test_section_returns_dict(self, config: CerberusConfig) -> None:
        system: dict[str, Any] = config.section("system")
        assert isinstance(system, dict)
        assert system["name"] == "cerberus-test"

    def test_missing_section_raises(self, config: CerberusConfig) -> None:
        with pytest.raises(ConfigError, match="not found"):
            config.section("nonexistent")

    def test_system_property(self, config: CerberusConfig) -> None:
        assert config.system["name"] == "cerberus-test"

    def test_database_property(self, config: CerberusConfig) -> None:
        assert "path" in config.database

    def test_mqtt_property(self, config: CerberusConfig) -> None:
        assert config.mqtt["port"] == 1883

    def test_safety_property(self, config: CerberusConfig) -> None:
        assert config.safety["battery_warn_pct"] == 25

    def test_motors_property(self, config: CerberusConfig) -> None:
        assert "left" in config.motors
        assert "right" in config.motors

    def test_navigation_property(self, config: CerberusConfig) -> None:
        assert config.navigation["home_lat"] == 36.1699

    def test_camera_property(self, config: CerberusConfig) -> None:
        assert config.camera["framerate"] == 30

    def test_sensors_property(self, config: CerberusConfig) -> None:
        assert "bme680" in config.sensors

    def test_heads_property(self, config: CerberusConfig) -> None:
        assert config.heads["active_head"] == "surveillance"

    def test_health_property(self, config: CerberusConfig) -> None:
        assert config.health["poll_interval"] == 5


class TestPathResolution:
    """Relative paths are resolved to absolute."""

    def test_database_path_absolute(self, config: CerberusConfig) -> None:
        db_path: str = config.get("database", "path")
        assert os.path.isabs(db_path)

    def test_log_dir_absolute(self, config: CerberusConfig) -> None:
        log_dir: str = config.get("system", "log_dir")
        assert os.path.isabs(log_dir)


class TestRepr:
    """String representation."""

    def test_repr_contains_path(self, config: CerberusConfig) -> None:
        r: str = repr(config)
        assert "CerberusConfig" in r
        assert "system" in r