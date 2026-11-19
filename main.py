"""Run the plant disease detection pipeline."""

from camera import capture_image
from display import print_report
from model_loader import DEFAULT_MODEL_PATH, predict
from preprocessing import preprocess_image


def main() -> None:
    """Capture, preprocess, classify, and display one plant image."""
    image = capture_image()
    input_tensor = preprocess_image(image)

    result = predict(input_tensor, model_path=DEFAULT_MODEL_PATH)

    print_report(result)


if __name__ == "__main__":
    main()