"""Turkish text normalization utilities."""
import re
from typing import List, Optional, Dict
import logging
import threading

# Zeyrek loglarını tamamen kapat
logging.getLogger('zeyrek').setLevel(logging.CRITICAL)

# Cache for lemmas
_lemma_cache: Dict[str, List[str]] = {}

# Zeyrek analyzer - lazy initialization
_analyzer = None

def _get_analyzer():
    """Lazy initialize Zeyrek MorphAnalyzer."""
    global _analyzer
    if _analyzer is None:
        try:
            from zeyrek import MorphAnalyzer
            _analyzer = MorphAnalyzer()
        except (ImportError, Exception):
            _analyzer = False
    return _analyzer if _analyzer else None


def get_lemma(word: str) -> str:
    """Get primary lemma (root form) of a Turkish word.
    
    Args:
        word: Input word (e.g., "konuştu", "geldim")
        
    Returns:
        Primary lemma form or original word if not found.
    """
    lemmas = get_all_lemmas(word)
    return lemmas[0] if lemmas else word


def _simple_stem(word: str) -> List[str]:
    """Simple rule-based Turkish verb stemmer as fallback.
    
    Args:
        word: Input word (e.g., "konuşuyor", "geldim")
        
    Returns:
        List of possible stems with 'mak/mek' suffix added.
    """
    if len(word) < 4:
        return [word]
    
    stems = set()
    
    # Common verb suffixes to remove (order matters - longer first)
    suffixes = [
        # -yor forms
        'ıyorum', 'iyorum', 'uyorum', 'üyorum',
        'ıyorsun', 'iyorsun', 'uyorsun', 'üyorsun', 
        'ıyor', 'iyor', 'uyor', 'üyor',
        'ıyoruz', 'iyoruz', 'uyoruz', 'üyoruz',
        'ıyorlar', 'iyorlar', 'uyorlar', 'üyorlar',
        # Past tense
        'dım', 'dim', 'dum', 'düm', 'tım', 'tim', 'tum', 'tüm',
        'dın', 'din', 'dun', 'dün', 'tın', 'tin', 'tun', 'tün',
        'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
        'dık', 'dik', 'duk', 'dük', 'tık', 'tik', 'tuk', 'tük',
        'dılar', 'diler', 'dular', 'düler', 'tılar', 'tiler', 'tular', 'tüler',
        # Future
        'acak', 'ecek', 'acağım', 'eceğim', 'acaksın', 'eceksin',
        # Miş forms
        'mış', 'miş', 'muş', 'müş',
        # Others
        'ır', 'ir', 'ur', 'ür', 'ar', 'er',
    ]
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            # Add mastar eki
            if stem[-1] in 'aıou':
                stems.add(stem + 'mak')
            else:
                stems.add(stem + 'mek')
    
    return list(stems) if stems else [word]


def _zeyrek_lemmatize_with_timeout(word: str, timeout: float = 0.3) -> Optional[List[str]]:
    """Lemmatize with timeout using threading.
    
    Args:
        word: Input word.
        timeout: Timeout in seconds.
        
    Returns:
        List of lemmas or None if timeout.
    """
    result = [None]
    exception = [None]
    
    def lemmatize():
        try:
            analyzer = _get_analyzer()
            if analyzer:
                results = analyzer.lemmatize(word)
                if results and len(results) > 0:
                    zeyrek_lemmas = results[0][1]
                    if zeyrek_lemmas:
                        result[0] = [l.lower() for l in zeyrek_lemmas]
        except Exception:
            pass
    
    thread = threading.Thread(target=lemmatize)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return None
    
    return result[0] if not exception[0] else None


def get_all_lemmas(word: str) -> List[str]:
    """Get all possible lemmas (root forms) of a Turkish word.
    
    Args:
        word: Input word (e.g., "konuştu", "geldim")
        
    Returns:
        List of possible lemmas (e.g., ["konmak", "konuşmak"]) or [word] if not found.
    """
    if not word or len(word) < 2:
        return [word]
    
    word_lower = word.lower()
    
    # Check cache first
    if word_lower in _lemma_cache:
        return _lemma_cache[word_lower]
    
    lemmas = []
    
    # Try Zeyrek with timeout
    zeyrek_result = _zeyrek_lemmatize_with_timeout(word_lower, timeout=0.3)
    if zeyrek_result:
        lemmas.extend(zeyrek_result)
    
    # If Zeyrek returned only the original word or nothing, try simple stemmer
    if not lemmas or lemmas == [word_lower]:
        simple_stems = _simple_stem(word_lower)
        lemmas.extend(simple_stems)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_lemmas = []
    for l in lemmas:
        if l not in seen:
            seen.add(l)
            unique_lemmas.append(l)
    
    # Cache result
    result = unique_lemmas if unique_lemmas else [word_lower]
    _lemma_cache[word_lower] = result
    
    return result


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

