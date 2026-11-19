"""Load a TensorFlow Lite model and run a single image prediction."""

import importlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

# Load NumPy dynamically so environments without type-checker import paths do
# not report a missing static import while preserving the runtime dependency.
np: Any = importlib.import_module("numpy")

CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MODEL_PATH = "plant_disease.tflite"
NUMBER_OF_CLASSES = 7


@dataclass
class Prediction:
    """The most likely class and its confidence score."""

    class_index: int
    confidence: float
    is_confident: bool
    used_dummy: bool = False


def _get_interpreter_class() -> Any:
    """Get an available TFLite interpreter implementation."""
    try:
        return importlib.import_module("tflite_runtime.interpreter").Interpreter
    except ImportError:
        try:
            return importlib.import_module("tensorflow.lite").Interpreter
        except ImportError:
            return None


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional[Any]:
    """Load a TFLite model, or return None if it is unavailable."""
    if not os.path.isfile(model_path):
        warnings.warn(
            f"Model file '{model_path}' was not found; dummy predictions will be used."
        )
        return None

    interpreter_class = _get_interpreter_class()
    if interpreter_class is None:
        warnings.warn(
            "Neither tflite-runtime nor TensorFlow is installed; "
            "dummy predictions will be used."
        )
        return None

    interpreter = interpreter_class(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _dummy_prediction() -> Prediction:
    """Return a deliberately uncertain placeholder prediction."""
    return Prediction(
        class_index=0,
        confidence=0.0,
        is_confident=False,
        used_dummy=True,
    )


def predict(
    input_tensor: Any,
    model_path: str = DEFAULT_MODEL_PATH,
    interpreter: Optional[Any] = None,
) -> Prediction:
    """
    Run inference and return the highest-scoring class.

    If the model is missing or inference fails, return a low-confidence
    placeholder so the application can still run.
    """
    if interpreter is None:
        interpreter = load_model(model_path)

    if interpreter is None:
        return _dummy_prediction()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    model_input = input_tensor
    input_dtype = input_details["dtype"]

    # Convert normalized floats to the model's quantized input if needed.
    if np.issubdtype(input_dtype, np.integer):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale:
            model_input = np.round(model_input / scale + zero_point)
        model_input = model_input.astype(input_dtype)
    else:
        model_input = model_input.astype(input_dtype)

    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    # Dequantize integer model output when quantization metadata is present.
    if np.issubdtype(output.dtype, np.integer):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale

    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if scores.size != NUMBER_OF_CLASSES or not np.all(np.isfinite(scores)):
        raise ValueError("Model output must contain seven finite class scores.")

    # Convert logits to probabilities; leave probability-like outputs intact.
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(
        np.sum(scores), 1.0, atol=0.05
    ):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)

    class_index = int(np.argmax(scores))
    confidence = float(np.clip(scores[class_index], 0.0, 1.0))

    return Prediction(
        class_index=class_index,
        confidence=confidence,
        is_confident=confidence >= CONFIDENCE_THRESHOLD,
    )