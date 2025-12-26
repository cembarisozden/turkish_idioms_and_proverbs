"""Inference pipeline for idiom/proverb detection."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.config import MODEL_NAME, MAX_LENGTH, DETECTOR_MODEL_PATH, DEFAULT_THRESHOLD, TOKEN_WINDOW_SIZE
from src.lexicon.matcher import LexiconMatcher
from src.utils.io import load_json

logger = logging.getLogger(__name__)

class IdiomDetector:
    """End-to-end idiom/proverb detector combining rule-based and transformer."""
    
    def __init__(self,
                 model_path: Path = DETECTOR_MODEL_PATH,
                 lexicon_path: Optional[Path] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 use_token_window: bool = True):
        """Initialize detector.
        
        Args:
            model_path: Path to saved transformer model.
            lexicon_path: Path to lexicon JSON file.
            threshold: Classification threshold.
            use_token_window: Whether to use token window matching.
        """
        self.threshold = threshold
        self.use_token_window = use_token_window
        
        # Load lexicon
        if lexicon_path is None:
            from src.config import LEXICON_PATH
            lexicon_path = LEXICON_PATH
        
        lexicon = load_json(lexicon_path)
        self.matcher = LexiconMatcher(lexicon)
        
        # Load transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        self.device = device
    
    def classify(self, text: str, temperature: float = 1.0) -> float:
        """Classify text using transformer with temperature scaling.
        
        Args:
            text: Input text.
            temperature: Temperature for softmax scaling (>1.0 = softer scores).
            
        Returns:
            Probability score (0-1).
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits / temperature
            probs = torch.softmax(logits, dim=-1)
            score = probs[0][1].item()
        
        return score
    
    def detect(self, text: str, threshold: Optional[float] = None) -> Dict:
        """Detect idioms/proverbs in text.
        
        Args:
            text: Input text.
            threshold: Classification threshold (overrides default).
            
        Returns:
            Dictionary with has_idiom, score, and matches.
        """
        if threshold is None:
            threshold = self.threshold
        
        # Step 1: Rule-based matching (for additional information)
        matches = self.matcher.match(
            text,
            use_token_window=self.use_token_window,
            window_size=TOKEN_WINDOW_SIZE
        )
        
        # Step 2: Transformer classification
        classifier_score = self.classify(text)
        
        # Step 3: Final decision - Transformer skoruna göre karar ver
        has_idiom = classifier_score >= threshold
        
        # Format matches - Lexicon'da bulunan eşleşmeleri ek bilgi olarak göster
        formatted_matches = []
        if len(matches) > 0:
            for match in matches:
                formatted_matches.append({
                    'expr': match['expression'],
                    'definition': match['definition'],
                    'span': match['span']
                })
        
        return {
            'has_idiom': has_idiom,
            'score': classifier_score,
            'matches': formatted_matches,
            'lexicon_found': len(matches) > 0
        }
