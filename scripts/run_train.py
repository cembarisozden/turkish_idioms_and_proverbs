"""Script to train the detector model."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_detector import train_detector
from src.config import GENERATED_DATASET_PATH, DETECTOR_MODEL_PATH
from src.utils.io import load_csv
from src.utils.logging import setup_logging
import logging

logger = setup_logging()

def main():
    """Main function to train model."""
    logger.info("Starting training...")
    
    # Load generated dataset
    logger.info(f"Loading dataset from {GENERATED_DATASET_PATH}")
    df = load_csv(GENERATED_DATASET_PATH)
    
    # Split into train and val
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Train
    train_detector(train_df, val_df, DETECTOR_MODEL_PATH)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

