"""Dataset loading utilities."""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def find_dataset_file(data_dir: Path) -> Optional[Path]:
    """Auto-detect dataset file in data directory.
    
    Args:
        data_dir: Directory to search.
        
    Returns:
        Path to dataset file or None if not found.
    """
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return None
    
    # Search for common data file extensions
    extensions = ['.csv', '.json', '.xlsx', '.xls']
    
    for ext in extensions:
        files = list(data_dir.glob(f'*{ext}'))
        if files:
            # Return the first matching file
            logger.info(f"Found dataset file: {files[0]}")
            return files[0]
    
    logger.error(f"No dataset file found in {data_dir}")
    return None

def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from file (auto-detect format).
    
    Args:
        filepath: Path to dataset file.
        
    Returns:
        Loaded DataFrame.
        
    Raises:
        ValueError: If file format is not supported or file cannot be loaded.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    try:
        if suffix == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8')
        elif suffix == '.json':
            df = pd.read_json(filepath, encoding='utf-8')
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def infer_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Infer column names for expression, definition, and type.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Tuple of (expression_col, definition_col, type_col).
        
    Raises:
        ValueError: If required columns cannot be inferred.
    """
    columns = [col.lower() for col in df.columns]
    
    # Try to find expression column
    expr_keywords = ['sozum', 'expression', 'expr', 'idiom', 'proverb', 'text', 'phrase', 'deyim', 'atasözü']
    expr_col = None
    for keyword in expr_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                expr_col = df.columns[i]
                break
        if expr_col:
            break
    
    # If not found, try first column
    if not expr_col:
        expr_col = df.columns[0]
        logger.warning(f"Could not infer expression column, using: {expr_col}")
    
    # Try to find definition column
    def_keywords = ['anlami', 'anlam', 'definition', 'def', 'meaning', 'açıklama', 'explanation']
    def_col = None
    for keyword in def_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                def_col = df.columns[i]
                break
        if def_col:
            break
    
    # If not found, try second column
    if not def_col:
        if len(df.columns) > 1:
            def_col = df.columns[1]
        else:
            def_col = expr_col  # Fallback
        logger.warning(f"Could not infer definition column, using: {def_col}")
    
    # Try to find type column (optional)
    type_keywords = ['turu2', 'turu', 'type', 'category', 'tür', 'kategori']
    type_col = None
    for keyword in type_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                type_col = df.columns[i]
                break
        if type_col:
            break
    
    logger.info(f"Inferred columns - Expression: {expr_col}, Definition: {def_col}, Type: {type_col}")
    
    return expr_col, def_col, type_col

def load_and_prepare_dataset(data_dir: Path) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    """Load dataset and infer columns.
    
    Args:
        data_dir: Directory containing dataset.
        
    Returns:
        Tuple of (DataFrame, expression_col, definition_col, type_col).
    """
    filepath = find_dataset_file(data_dir)
    if not filepath:
        raise FileNotFoundError(f"No dataset file found in {data_dir}")
    
    df = load_dataset(filepath)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    expr_col, def_col, type_col = infer_columns(df)
    
    return df, expr_col, def_col, type_col

