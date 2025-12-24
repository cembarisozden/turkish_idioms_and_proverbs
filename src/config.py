"""Configuration settings for the Turkish NLP pipeline."""
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 64  # Reduced from 128 for faster training
BATCH_SIZE = 32  # Increased from 16 for faster training
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2  # Reduced from 3 for faster training
EARLY_STOPPING_PATIENCE = 3  # ✅ Artırıldı: daha erken durma (overfitting önleme)
EARLY_STOPPING_THRESHOLD = 0.0001  # ✅ Düşürüldü: daha hassas durma

# Training configuration
RANDOM_SEED = 42
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Weak labeling configuration
NUM_POSITIVE_EXAMPLES = 5000  # ✅ Artırıldı: daha fazla veri = daha az overfitting
NUM_NEGATIVE_EXAMPLES = 5000  # ✅ Artırıldı: daha fazla veri = daha az overfitting
TEMPLATE_FILE = PROJECT_ROOT / "src" / "data" / "templates.txt"

# Inference configuration
DEFAULT_THRESHOLD = 0.8  # ✅ Artırıldı: 0.6 -> 0.8 (daha az false positive için)
TOKEN_WINDOW_SIZE = 5  # For n-gram matching

# Output paths
LEXICON_PATH = ARTIFACTS_DIR / "lexicon.json"
GENERATED_DATASET_PATH = ARTIFACTS_DIR / "generated_dataset.csv"
DETECTOR_MODEL_PATH = ARTIFACTS_DIR / "detector_model"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
    }

