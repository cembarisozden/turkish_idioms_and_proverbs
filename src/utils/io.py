"""I/O utilities for file operations."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save.
        filepath: Path to save file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: Path) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Path to JSON file.
        
    Returns:
        Loaded data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save.
        filepath: Path to save file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8')

def load_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV file as DataFrame.
    
    Args:
        filepath: Path to CSV file.
        
    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(filepath, encoding='utf-8')

