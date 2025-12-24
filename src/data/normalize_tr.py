"""Turkish text normalization utilities."""
import re
from typing import List

def turkish_lowercase(text: str) -> str:
    """Convert Turkish text to lowercase handling I/İ correctly.
    
    Args:
        text: Input text.
        
    Returns:
        Lowercased text with proper Turkish character handling.
    """
    # Turkish lowercase mapping: I -> ı, İ -> i
    text = text.replace('I', 'ı').replace('İ', 'i')
    return text.lower()

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Args:
        text: Input text.
        
    Returns:
        Text with normalized whitespace.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_punctuation(text: str) -> str:
    """Normalize punctuation marks.
    
    Args:
        text: Input text.
        
    Returns:
        Text with normalized punctuation.
    """
    # Remove extra punctuation, keep basic ones
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
    return text

def normalize_turkish_text(text: str, 
                          lowercase: bool = True,
                          normalize_ws: bool = True,
                          normalize_punct: bool = False) -> str:
    """Normalize Turkish text.
    
    Args:
        text: Input text.
        lowercase: Whether to lowercase.
        normalize_ws: Whether to normalize whitespace.
        normalize_punct: Whether to normalize punctuation.
        
    Returns:
        Normalized text.
    """
    if not text:
        return ""
    
    if lowercase:
        text = turkish_lowercase(text)
    
    if normalize_ws:
        text = normalize_whitespace(text)
    
    if normalize_punct:
        text = normalize_punctuation(text)
    
    return text

def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for Turkish text.
    
    Args:
        text: Input text.
        
    Returns:
        List of tokens.
    """
    # Simple word tokenization
    tokens = re.findall(r'\b\w+\b', normalize_turkish_text(text))
    return tokens

