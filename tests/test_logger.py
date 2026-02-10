"""
Tests for cerberus.core.logger — Logging system
Validates setup_logging, rotating file handlers, MQTT log handler,
log level configuration, and convenience functions.
"""

import os
import logging
import pytest
from typing import Any, Optional
from unittest.mock import MagicMock

from cerberus.core.config import CerberusConfig


def _reset_logging() -> None:
    """Reset the logger module state between tests."""
    import cerberus.core.logger as log_module
    log_module._logger_initialized = False
    log_module._mqtt_handler = None
    root: logging.Logger = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


class TestSetupLogging:
    """setup_logging initialization."""

    def test_initializes_logging(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["system"]["log_level"] = "DEBUG"

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        root: logging.Logger = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 2
        _reset_logging()

    def test_creates_log_directory(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        log_dir: str = os.path.join(temp_dir, "new_logs")
        config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        assert os.path.isdir(log_dir)
        _reset_logging()

    def test_creates_log_files(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        log_dir: str = os.path.join(temp_dir, "file_logs")
        config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        logger: logging.Logger = logging.getLogger("cerberus.test")
        logger.info("Test message")
        logger.error("Error message")

        assert os.path.exists(os.path.join(log_dir, "cerberus.log"))
        assert os.path.exists(os.path.join(log_dir, "cerberus_errors.log"))
        _reset_logging()

    def test_idempotent_setup(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging
        setup_logging(config)
        handler_count: int = len(logging.getLogger().handlers)

        setup_logging(config)
        assert len(logging.getLogger().handlers) == handler_count
        _reset_logging()

    def test_log_level_info(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["system"]["log_level"] = "INFO"

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        root: logging.Logger = logging.getLogger()
        assert root.level == logging.INFO
        _reset_logging()

    def test_log_level_warning(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["system"]["log_level"] = "WARNING"

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        root: logging.Logger = logging.getLogger()
        assert root.level == logging.WARNING
        _reset_logging()

    def test_invalid_log_level_defaults_info(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["system"]["log_level"] = "NONEXISTENT"

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        root: logging.Logger = logging.getLogger()
        assert root.level == logging.INFO
        _reset_logging()


class TestLogFormat:
    """Log output formatting."""

    def test_format_contains_expected_fields(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        log_dir: str = os.path.join(temp_dir, "fmt_logs")
        config._data["system"]["log_dir"] = log_dir
        config._data["system"]["log_level"] = "DEBUG"

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        logger: logging.Logger = logging.getLogger("cerberus.format_test")
        logger.info("Format validation message")

        log_file: str = os.path.join(log_dir, "cerberus.log")
        with open(log_file, "r", encoding="utf-8") as f:
            content: str = f.read()

        assert "cerberus.format_test" in content
        assert "INFO" in content
        assert "Format validation message" in content
        _reset_logging()


class TestErrorLog:
    """Error-level log file."""

    def test_errors_written_to_error_file(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        log_dir: str = os.path.join(temp_dir, "err_logs")
        config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        logger: logging.Logger = logging.getLogger("cerberus.error_test")
        logger.error("This is an error")

        error_file: str = os.path.join(log_dir, "cerberus_errors.log")
        with open(error_file, "r", encoding="utf-8") as f:
            content: str = f.read()

        assert "This is an error" in content
        _reset_logging()

    def test_info_not_in_error_file(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        log_dir: str = os.path.join(temp_dir, "err_logs2")
        config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging
        setup_logging(config)

        logger: logging.Logger = logging.getLogger("cerberus.info_test")
        logger.info("Just info")

        error_file: str = os.path.join(log_dir, "cerberus_errors.log")
        with open(error_file, "r", encoding="utf-8") as f:
            content: str = f.read()

        assert "Just info" not in content
        _reset_logging()


class TestMQTTLogHandler:
    """MQTTLogHandler for publishing alerts."""

    def test_handler_created(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler, MQTTLogHandler
        setup_logging(config)

        handler: Optional[MQTTLogHandler] = get_mqtt_log_handler()
        assert handler is not None
        _reset_logging()

    def test_handler_level_is_warning(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        assert handler.level == logging.WARNING
        _reset_logging()

    def test_no_publish_without_binding(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        record: logging.LogRecord = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="Alert test", args=(), exc_info=None
        )
        handler.emit(record)
        _reset_logging()

    def test_publish_after_binding(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        mock_client: MagicMock = MagicMock()
        handler.bind_mqtt(mock_client)

        logger: logging.Logger = logging.getLogger("cerberus.mqtt_test")
        logger.warning("MQTT alert test")

        mock_client.publish.assert_called()
        call_args = mock_client.publish.call_args
        assert "cerberus/" in call_args[0][0]
        assert "MQTT alert test" in call_args[0][1]
        _reset_logging()

    def test_info_not_published(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["system"]["log_level"] = "DEBUG"

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        mock_client: MagicMock = MagicMock()
        handler.bind_mqtt(mock_client)

        logger: logging.Logger = logging.getLogger("cerberus.info_mqtt")
        logger.info("Should not publish")

        mock_client.publish.assert_not_called()
        _reset_logging()

    def test_publish_error_handled(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        mock_client: MagicMock = MagicMock()
        mock_client.publish.side_effect = RuntimeError("network down")
        handler.bind_mqtt(mock_client)

        record: logging.LogRecord = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="Should handle error", args=(), exc_info=None
        )
        handler.emit(record)
        _reset_logging()

    def test_custom_alert_topic(self, config: CerberusConfig, temp_dir: str) -> None:
        _reset_logging()
        config._data["system"]["log_dir"] = os.path.join(temp_dir, "logs")
        config._data["mqtt"]["topics"]["alerts"] = "cerberus/custom/alerts"

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        handler = get_mqtt_log_handler()
        assert handler._topic == "cerberus/custom/alerts"
        _reset_logging()


class TestGetLogger:
    """get_logger convenience function."""

    def test_returns_named_logger(self) -> None:
        from cerberus.core.logger import get_logger
        logger: logging.Logger = get_logger("cerberus.test.module")
        assert logger.name == "cerberus.test.module"

    def test_returns_same_logger(self) -> None:
        from cerberus.core.logger import get_logger
        a: logging.Logger = get_logger("cerberus.same")
        b: logging.Logger = get_logger("cerberus.same")
        assert a is b


class TestGetMQTTHandler:
    """get_mqtt_log_handler retrieval."""

    def test_returns_none_before_setup(self) -> None:
        _reset_logging()
        from cerberus.core.logger import get_mqtt_log_handler
        handler = get_mqtt_log_handler()
        assert handler is None
        _reset_logging()