"""Rule-based lexicon matcher for idioms/proverbs."""
import re
from typing import List, Dict, Tuple, Optional
import logging

from src.data.normalize_tr import normalize_turkish_text, tokenize_simple

logger = logging.getLogger(__name__)

class LexiconMatcher:
    """Rule-based matcher for Turkish idioms and proverbs."""
    
    def __init__(self, lexicon: Dict[str, Dict]):
        """Initialize matcher with lexicon.
        
        Args:
            lexicon: Dictionary mapping normalized expressions to metadata.
        """
        self.lexicon = lexicon
        self.normalized_expressions = list(lexicon.keys())
        
        # Pre-compile patterns for faster matching
        self.patterns = {}
        for expr in self.normalized_expressions:
            # Escape special regex characters
            pattern = re.escape(expr)
            self.patterns[expr] = re.compile(pattern, re.IGNORECASE)
    
    def exact_match(self, text: str) -> List[Dict]:
        """Find exact matches of expressions in text.
        
        Args:
            text: Input text to search.
            
        Returns:
            List of matches with span, expression, definition.
        """
        matches = []
        normalized_text = normalize_turkish_text(text)
        
        for expr in self.normalized_expressions:
            pattern = self.patterns[expr]
            
            # Find all occurrences
            for match in pattern.finditer(normalized_text):
                start, end = match.span()
                
                # Get original expression and definition
                expr_original = self.lexicon[expr].get('original', expr)
                definition = self.lexicon[expr].get('definition', '')
                
                matches.append({
                    'span': [start, end],
                    'expression': expr_original,
                    'definition': definition,
                    'normalized_expr': expr
                })
        
        # Remove overlapping matches (keep longer ones)
        matches = self._remove_overlaps(matches)
        
        return matches
    
    def token_window_match(self, text: str, window_size: int = 5) -> List[Dict]:
        """Find matches using token window (n-gram) approach.
        
        Args:
            text: Input text to search.
            window_size: Maximum size of token window for matching.
            
        Returns:
            List of matches with span, expression, definition.
        """
        matches = []
        tokens = tokenize_simple(text)  # ✅ tokenize_simple zaten normalize ediyor
        
        # ✅ Tüm olası deyim uzunluklarını kontrol et
        # Her deyim için metinde tüm pozisyonları kontrol et
        for expr in self.normalized_expressions:
            expr_tokens = tokenize_simple(expr)
            expr_len = len(expr_tokens)
            
            # ✅ Minimum 2 token olmalı, maksimum window_size kadar olabilir
            # Ama window_size'dan uzun deyimler için de kontrol et (sliding window ile)
            if expr_len < 2:
                continue
            
            # Eğer deyim window_size'dan uzunsa, metinde yeterli yer yoksa atla
            if expr_len > len(tokens):
                continue
            
            # Metindeki tüm pozisyonlarda bu deyimi ara
            for i in range(len(tokens) - expr_len + 1):
                window_tokens = tokens[i:i + expr_len]
                
                # Token'ları eşleştir (normalize edilmiş)
                if self._tokens_match(expr_tokens, window_tokens):
                    # Gerçek span'ı bul
                    span = self._find_token_span(text, i, i + expr_len)
                    
                    if span:
                        expr_original = self.lexicon[expr].get('original', expr)
                        definition = self.lexicon[expr].get('definition', '')
                        
                        # Aynı deyim için zaten eşleşme varsa atla (overlap kontrolü sonra yapılacak)
                        matches.append({
                            'span': span,
                            'expression': expr_original,
                            'definition': definition,
                            'normalized_expr': expr
                        })
                        break  # Bu deyim için bir eşleşme bulundu, diğer pozisyonlara bakma
        
        # Remove overlapping matches
        matches = self._remove_overlaps(matches)
        
        return matches
    
    def _tokens_match(self, expr_tokens: List[str], window_tokens: List[str]) -> bool:
        """Check if expression tokens match window tokens.
        
        Args:
            expr_tokens: Tokens from expression (normalized).
            window_tokens: Tokens from window (normalized).
            
        Returns:
            True if tokens match.
        """
        if len(expr_tokens) != len(window_tokens):
            return False
        
        # ✅ Normalize edilmiş token'ları karşılaştır
        # Her iki tarafı da normalize et ve karşılaştır
        expr_normalized = [normalize_turkish_text(t) for t in expr_tokens]
        window_normalized = [normalize_turkish_text(t) for t in window_tokens]
        
        return expr_normalized == window_normalized
    
    def _find_token_span(self, text: str, start_token_idx: int, end_token_idx: int) -> Optional[List[int]]:
        """Find character span for token indices.
        
        Args:
            text: Original text.
            start_token_idx: Start token index.
            end_token_idx: End token index.
            
        Returns:
            [start_char, end_char] or None.
        """
        tokens = tokenize_simple(text)
        if end_token_idx > len(tokens):
            return None
        
        # Find character positions
        pattern = r'\b\w+\b'
        matches = list(re.finditer(pattern, text))
        
        if start_token_idx >= len(matches) or end_token_idx > len(matches):
            return None
        
        start_char = matches[start_token_idx].start()
        end_char = matches[end_token_idx - 1].end()
        
        return [start_char, end_char]
    
    def _remove_overlaps(self, matches: List[Dict]) -> List[Dict]:
        """Remove overlapping matches, keeping longer ones.
        
        Args:
            matches: List of matches.
            
        Returns:
            Filtered list without overlaps.
        """
        if not matches:
            return []
        
        # Sort by start position
        matches = sorted(matches, key=lambda x: (x['span'][0], -x['span'][1]))
        
        filtered = []
        for match in matches:
            if not filtered:
                filtered.append(match)
            else:
                last = filtered[-1]
                # Check overlap
                if match['span'][0] >= last['span'][1]:
                    # No overlap
                    filtered.append(match)
                elif match['span'][1] - match['span'][0] > last['span'][1] - last['span'][0]:
                    # Overlap but current is longer
                    filtered[-1] = match
        
        return filtered
    
    def match(self, text: str, use_token_window: bool = False, window_size: int = 5) -> List[Dict]:
        """Match expressions in text.
        
        Args:
            text: Input text.
            use_token_window: Whether to use token window matching.
            window_size: Size of token window.
            
        Returns:
            List of matches.
        """
        if use_token_window:
            return self.token_window_match(text, window_size)
        else:
            return self.exact_match(text)

