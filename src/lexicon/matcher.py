"""Rule-based lexicon matcher for idioms/proverbs."""
import re
from typing import List, Dict, Tuple, Optional
import logging

from src.data import normalize_tr

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
        normalized_text = normalize_tr.normalize_turkish_text(text)
        
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
    
    def token_window_match(self, text: str, window_size: int = 8) -> List[Dict]:
        """Find matches using token window (n-gram) approach with partial matching.
        
        Args:
            text: Input text to search.
            window_size: Maximum size of token window for matching.
            
        Returns:
            List of matches with span, expression, definition.
        """
        matches = []
        tokens = normalize_tr.tokenize_simple(text)
        
        for expr in self.normalized_expressions:
            expr_tokens = normalize_tr.tokenize_simple(expr)
            expr_len = len(expr_tokens)
            
            if expr_len < 2:
                continue
            
            # Try exact match first
            if expr_len <= len(tokens):
                for i in range(len(tokens) - expr_len + 1):
                    window_tokens = tokens[i:i + expr_len]
                    
                    if self._tokens_match(expr_tokens, window_tokens):
                        span = self._find_token_span(text, i, i + expr_len)
                        
                        if span:
                            expr_original = self.lexicon[expr].get('original', expr)
                            definition = self.lexicon[expr].get('definition', '')
                            
                            matches.append({
                                'span': span,
                                'expression': expr_original,
                                'definition': definition,
                                'normalized_expr': expr
                            })
                            break
            
            # Try flexible partial match: match when expression tokens can be found in text
            # This handles cases like "açlıktan gözü gözleri dönmek" -> "açlıktan gözü dönmüştü"
            # Only for expressions with 3+ tokens
            if expr_len >= 3:
                # Try matching at least 2 consecutive tokens from expression
                for match_start in range(expr_len - 1):
                    for match_end in range(match_start + 2, expr_len + 1):
                        match_len = match_end - match_start
                        if match_len > len(tokens):
                            break
                        
                        expr_subset = expr_tokens[match_start:match_end]
                        
                        for i in range(len(tokens) - match_len + 1):
                            window_tokens = tokens[i:i + match_len]
                            
                            if self._tokens_match(expr_subset, window_tokens):
                                # Only accept if we matched at least 2 tokens and it's a significant portion
                                matched_ratio = match_len / expr_len
                                if matched_ratio >= 0.5 or match_len >= 2:
                                    span = self._find_token_span(text, i, i + match_len)
                                    
                                    if span:
                                        expr_original = self.lexicon[expr].get('original', expr)
                                        definition = self.lexicon[expr].get('definition', '')
                                        
                                        matches.append({
                                            'span': span,
                                            'expression': expr_original,
                                            'definition': definition,
                                            'normalized_expr': expr
                                        })
                                        break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
        
        matches = self._remove_overlaps(matches)
        
        return matches
    
    def _tokens_match(self, expr_tokens: List[str], window_tokens: List[str], allow_skip: bool = False) -> bool:
        """Check if expression tokens match window tokens using lemmatization.
        
        Args:
            expr_tokens: Tokens from expression (e.g., ["abuk", "sabuk", "konuşmak"]).
            window_tokens: Tokens from window (e.g., ["abuk", "sabuk", "konuştu"]).
            allow_skip: If True, allow skipping some tokens (for partial matching).
            
        Returns:
            True if tokens match (considering all possible lemmas).
        """
        if len(expr_tokens) != len(window_tokens) and not allow_skip:
            return False
        
        # Import fresh each time to avoid stale references
        from src.data.normalize_tr import normalize_turkish_text, get_all_lemmas
        
        # If lengths don't match and skip is allowed, try flexible matching
        if allow_skip and len(expr_tokens) != len(window_tokens):
            return self._flexible_tokens_match(expr_tokens, window_tokens)
        
        for expr_token, window_token in zip(expr_tokens, window_tokens):
            expr_norm = normalize_turkish_text(expr_token)
            window_norm = normalize_turkish_text(window_token)
            
            if expr_norm == window_norm:
                continue
            
            expr_lemmas = set(get_all_lemmas(expr_norm))
            window_lemmas = set(get_all_lemmas(window_norm))
            
            if not expr_lemmas.intersection(window_lemmas):
                return False
        
        return True
    
    def _flexible_tokens_match(self, expr_tokens: List[str], window_tokens: List[str]) -> bool:
        """Flexible token matching allowing some tokens to be skipped.
        
        Args:
            expr_tokens: Tokens from expression.
            window_tokens: Tokens from window.
            
        Returns:
            True if at least 70% of tokens match.
        """
        from src.data.normalize_tr import normalize_turkish_text, get_all_lemmas
        
        matches = 0
        expr_idx = 0
        window_idx = 0
        
        while expr_idx < len(expr_tokens) and window_idx < len(window_tokens):
            expr_token = expr_tokens[expr_idx]
            window_token = window_tokens[window_idx]
            
            expr_norm = normalize_turkish_text(expr_token)
            window_norm = normalize_turkish_text(window_token)
            
            if expr_norm == window_norm:
                matches += 1
                expr_idx += 1
                window_idx += 1
            else:
                expr_lemmas = set(get_all_lemmas(expr_norm))
                window_lemmas = set(get_all_lemmas(window_norm))
                
                if expr_lemmas.intersection(window_lemmas):
                    matches += 1
                    expr_idx += 1
                    window_idx += 1
                else:
                    # Try skipping one token from either side
                    if expr_idx + 1 < len(expr_tokens):
                        next_expr_norm = normalize_turkish_text(expr_tokens[expr_idx + 1])
                        if next_expr_norm == window_norm:
                            expr_idx += 1
                            continue
                    if window_idx + 1 < len(window_tokens):
                        next_window_norm = normalize_turkish_text(window_tokens[window_idx + 1])
                        if expr_norm == next_window_norm:
                            window_idx += 1
                            continue
                    # No match, advance both
                    expr_idx += 1
                    window_idx += 1
        
        # At least 70% of tokens must match
        min_matches = max(2, int(len(expr_tokens) * 0.7))
        return matches >= min_matches
    
    def _find_token_span(self, text: str, start_token_idx: int, end_token_idx: int) -> Optional[List[int]]:
        """Find character span for token indices.
        
        Args:
            text: Original text.
            start_token_idx: Start token index.
            end_token_idx: End token index.
            
        Returns:
            [start_char, end_char] or None.
        """
        tokens = normalize_tr.tokenize_simple(text)
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

