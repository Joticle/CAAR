"""
Cerberus TFLite Classifier
Loads and runs TensorFlow Lite models for image classification at the edge.
Supports multiple models for different heads (weed detection, threat
classification, pest identification). All inference happens on-device.
Gracefully degrades when TFLite runtime is not present (dev environment).
"""

import time
import logging
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of a single classification inference."""
    label: str = "unknown"
    confidence: float = 0.0
    class_index: int = -1
    top_k: list[dict[str, Any]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_name: str = ""
    timestamp: float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "class_index": self.class_index,
            "top_k": self.top_k,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_name": self.model_name,
            "timestamp": self.timestamp
        }


class TFLiteModel:
    """
    Wraps a single TFLite model file.
    Handles loading, input preparation, inference, and output parsing.
    """

    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        name: str = ""
    ) -> None:
        self._model_path: str = model_path
        self._labels_path: Optional[str] = labels_path
        self._name: str = name or Path(model_path).stem
        self._interpreter: Optional[Any] = None
        self._labels: list[str] = []
        self._input_details: Optional[list[dict]] = None
        self._output_details: Optional[list[dict]] = None
        self._input_shape: tuple[int, ...] = (0, 0, 0, 0)
        self._input_dtype: Any = np.uint8
        self._loaded: bool = False

        self._load_model()
        self._load_labels()

    def _load_model(self) -> None:
        """Load TFLite model file and allocate tensors."""
        if not Path(self._model_path).exists():
            logger.warning("Model file not found: %s", self._model_path)
            return

        try:
            from tflite_runtime.interpreter import Interpreter

            self._interpreter = Interpreter(model_path=self._model_path)
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shape = tuple(self._input_details[0]["shape"])
            self._input_dtype = self._input_details[0]["dtype"]

            self._loaded = True
            logger.info(
                "TFLite model loaded: %s — input_shape=%s, dtype=%s",
                self._name, self._input_shape, self._input_dtype
            )

        except ImportError:
            try:
                import tensorflow as tf

                self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
                self._interpreter.allocate_tensors()

                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._input_shape = tuple(self._input_details[0]["shape"])
                self._input_dtype = self._input_details[0]["dtype"]

                self._loaded = True
                logger.info(
                    "TFLite model loaded (via tensorflow): %s — input_shape=%s",
                    self._name, self._input_shape
                )

            except ImportError:
                logger.warning(
                    "No TFLite runtime available — model '%s' not loaded (dev mode)",
                    self._name
                )

        except Exception as e:
            logger.error("Failed to load model '%s': %s", self._name, e)

    def _load_labels(self) -> None:
        """Load label file (one label per line)."""
        if self._labels_path is None:
            auto_path: str = str(Path(self._model_path).with_suffix(".txt"))
            if Path(auto_path).exists():
                self._labels_path = auto_path

        if self._labels_path is None or not Path(self._labels_path).exists():
            logger.debug("No labels file for model '%s'", self._name)
            return

        try:
            with open(self._labels_path, "r", encoding="utf-8") as f:
                self._labels = [line.strip() for line in f if line.strip()]
            logger.info("Loaded %d labels for model '%s'", len(self._labels), self._name)
        except Exception as e:
            logger.error("Failed to load labels for '%s': %s", self._name, e)

    def predict(self, image: np.ndarray, top_k: int = 5) -> ClassificationResult:
        """
        Run inference on an image.
        image: numpy array (H, W, 3) in BGR or RGB format.
        Returns ClassificationResult with label, confidence, and top_k results.
        """
        if not self._loaded or self._interpreter is None:
            return self._simulated_result()

        try:
            start_time: float = time.time()

            input_data: np.ndarray = self._prepare_input(image)

            self._interpreter.set_tensor(
                self._input_details[0]["index"],
                input_data
            )
            self._interpreter.invoke()

            output_data: np.ndarray = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )

            elapsed_ms: float = (time.time() - start_time) * 1000

            return self._parse_output(output_data, top_k, elapsed_ms)

        except Exception as e:
            logger.error("Inference error on model '%s': %s", self._name, e)
            return ClassificationResult(
                model_name=self._name,
                timestamp=time.time()
            )

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Resize and format image for model input."""
        from PIL import Image

        target_h: int = self._input_shape[1]
        target_w: int = self._input_shape[2]

        pil_image: Image.Image = Image.fromarray(image)
        pil_image = pil_image.resize((target_w, target_h), Image.BILINEAR)
        input_array: np.ndarray = np.array(pil_image)

        if self._input_dtype == np.float32:
            input_array = input_array.astype(np.float32) / 255.0
        else:
            input_array = input_array.astype(self._input_dtype)

        return np.expand_dims(input_array, axis=0)

    def _parse_output(
        self,
        output: np.ndarray,
        top_k: int,
        inference_time_ms: float
    ) -> ClassificationResult:
        """Parse model output into ClassificationResult."""
        scores: np.ndarray = output[0]

        if scores.dtype == np.uint8:
            scores = scores.astype(np.float32) / 255.0
        elif scores.dtype == np.int8:
            scores = (scores.astype(np.float32) + 128) / 255.0

        top_indices: np.ndarray = np.argsort(scores)[::-1][:top_k]

        top_results: list[dict[str, Any]] = []
        for idx in top_indices:
            label: str = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
            top_results.append({
                "label": label,
                "confidence": round(float(scores[idx]), 4),
                "index": int(idx)
            })

        best_idx: int = int(top_indices[0])
        best_label: str = self._labels[best_idx] if best_idx < len(self._labels) else f"class_{best_idx}"
        best_confidence: float = float(scores[best_idx])

        return ClassificationResult(
            label=best_label,
            confidence=best_confidence,
            class_index=best_idx,
            top_k=top_results,
            inference_time_ms=inference_time_ms,
            model_name=self._name,
            timestamp=time.time()
        )

    def _simulated_result(self) -> ClassificationResult:
        """Return a simulated result when model is not loaded."""
        return ClassificationResult(
            label="simulated",
            confidence=0.0,
            class_index=0,
            top_k=[{"label": "simulated", "confidence": 0.0, "index": 0}],
            inference_time_ms=0.0,
            model_name=self._name,
            timestamp=time.time()
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def labels(self) -> list[str]:
        return self._labels

    def __repr__(self) -> str:
        return (
            f"TFLiteModel(name='{self._name}', "
            f"loaded={self._loaded}, "
            f"input={self._input_shape}, "
            f"labels={len(self._labels)})"
        )


class Classifier:
    """
    Multi-model classifier manager for Cerberus.
    Loads and manages multiple TFLite models for different tasks.
    Each payload head requests inference from its designated model.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._models_dir: str = config.get("intelligence", "models_dir", default="models")
        self._confidence_threshold: float = config.get(
            "intelligence", "confidence_threshold", default=0.5
        )
        self._models: dict[str, TFLiteModel] = {}

        logger.info("Classifier manager created — models_dir=%s", self._models_dir)

    def load_model(
        self,
        name: str,
        model_file: str,
        labels_file: Optional[str] = None
    ) -> bool:
        """
        Load a named TFLite model.
        name: unique identifier (e.g., "weed_detector", "threat_classifier")
        model_file: filename relative to models_dir
        labels_file: optional labels filename relative to models_dir
        """
        model_path: str = str(Path(self._models_dir) / model_file)
        labels_path: Optional[str] = None
        if labels_file:
            labels_path = str(Path(self._models_dir) / labels_file)

        model: TFLiteModel = TFLiteModel(
            model_path=model_path,
            labels_path=labels_path,
            name=name
        )

        self._models[name] = model

        if model.loaded:
            logger.info("Model '%s' loaded and ready", name)
            return True
        else:
            logger.warning("Model '%s' registered but not loaded (missing file or runtime)", name)
            return False

    def classify(
        self,
        model_name: str,
        image: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> ClassificationResult:
        """
        Run classification using a named model.
        Returns ClassificationResult. Uses configured threshold if not specified.
        """
        if model_name not in self._models:
            logger.error("Model '%s' not found — available: %s",
                         model_name, list(self._models.keys()))
            return ClassificationResult(model_name=model_name, timestamp=time.time())

        model: TFLiteModel = self._models[model_name]
        result: ClassificationResult = model.predict(image, top_k=top_k)

        effective_threshold: float = threshold if threshold is not None else self._confidence_threshold

        if result.confidence < effective_threshold:
            logger.debug(
                "Classification below threshold: %s=%.3f (threshold=%.3f)",
                result.label, result.confidence, effective_threshold
            )

        return result

    def get_model(self, name: str) -> Optional[TFLiteModel]:
        """Get a loaded model by name."""
        return self._models.get(name)

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def loaded_count(self) -> int:
        return sum(1 for m in self._models.values() if m.loaded)

    def __repr__(self) -> str:
        loaded: int = self.loaded_count
        total: int = len(self._models)
        return f"Classifier(models={loaded}/{total} loaded)"