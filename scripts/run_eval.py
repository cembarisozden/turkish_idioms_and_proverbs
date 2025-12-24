"""Script to evaluate the detector model."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate_detector import evaluate_detector, save_metrics
from src.config import GENERATED_DATASET_PATH, METRICS_PATH
from src.utils.io import load_csv
from src.utils.logging import setup_logging
import logging

logger = setup_logging()

def main():
    """Main function to evaluate model."""
    logger.info("Starting evaluation...")
    
    # Load test set
    logger.info(f"Loading dataset from {GENERATED_DATASET_PATH}")
    df = load_csv(GENERATED_DATASET_PATH)
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"Test samples: {len(test_df)}")
    
    # Evaluate
    metrics = evaluate_detector(test_df)
    
    # Save metrics
    save_metrics(metrics, METRICS_PATH)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()

