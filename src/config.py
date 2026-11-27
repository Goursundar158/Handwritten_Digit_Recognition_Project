"""Configuration values shared by training and inference."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "digit_classifier.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
PIPELINE_FIGURE_PATH = RESULTS_DIR / "preprocessing_pipeline.png"
SEGMENTATION_FIGURE_PATH = RESULTS_DIR / "segmentation_boxes.png"
PREDICTIONS_PATH = RESULTS_DIR / "sample_predictions.csv"

IMAGE_SIZE = 32
PIXEL_SIZE = 16
HOG_CELL_SIZE = 8
HOG_BINS = 9
RANDOM_STATE = 42
TEST_SIZE = 0.20
