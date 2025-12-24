"""Dataset splitting utilities."""
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
import logging

from src.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

logger = logging.getLogger(__name__)

def split_dataset(df: pd.DataFrame,
                  train_ratio: float = TRAIN_SPLIT,
                  val_ratio: float = VAL_SPLIT,
                  test_ratio: float = TEST_SPLIT,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.
    
    Args:
        df: DataFrame to split.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        random_state: Random seed.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )
    
    logger.info(f"Split dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df

