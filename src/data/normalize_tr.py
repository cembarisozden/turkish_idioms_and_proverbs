"""Turkish text normalization utilities."""
import re
from typing import List, Optional, Dict
import logging
import threading
import time

# Logger setup
logger = logging.getLogger(__name__)

# Zeyrek loglarını tamamen kapat
logging.getLogger('zeyrek').setLevel(logging.CRITICAL)

# Cache for lemmas
_lemma_cache: Dict[str, List[str]] = {}

# Zeyrek analyzer - lazy initialization
_analyzer = None
_zeyrek_available = None  # None = not checked, True/False = checked

# Zeyrek istatistikleri
_zeyrek_stats = {
    'calls': 0,
    'successes': 0,
    'timeouts': 0,
    'errors': 0,
    'fallbacks': 0
}


def get_zeyrek_stats() -> Dict:
    """Zeyrek kullanım istatistiklerini döndür."""
    return _zeyrek_stats.copy()


def _get_analyzer():
    """Lazy initialize Zeyrek MorphAnalyzer."""
    global _analyzer, _zeyrek_available
    
    if _analyzer is None and _zeyrek_available is None:
        try:
            logger.info("Zeyrek MorphAnalyzer yükleniyor...")
            start_time = time.time()
            from zeyrek import MorphAnalyzer
            _analyzer = MorphAnalyzer()
            load_time = time.time() - start_time
            _zeyrek_available = True
            logger.info(f"✅ Zeyrek başarıyla yüklendi ({load_time:.2f}s)")
        except ImportError as e:
            _analyzer = False
            _zeyrek_available = False
            logger.warning(f"❌ Zeyrek import edilemedi: {e}")
        except Exception as e:
            _analyzer = False
            _zeyrek_available = False
            logger.error(f"❌ Zeyrek yüklenirken hata: {e}")
    
    return _analyzer if _analyzer else None


def check_zeyrek_status() -> Dict:
    """Zeyrek durumunu kontrol et ve detaylı bilgi döndür."""
    global _zeyrek_available
    
    result = {
        'available': False,
        'loaded': False,
        'test_result': None,
        'error': None
    }
    
    try:
        analyzer = _get_analyzer()
        if analyzer:
            result['available'] = True
            result['loaded'] = True
            
            # Test et
            test_word = "geldim"
            test_result = analyzer.lemmatize(test_word)
            if test_result and len(test_result) > 0:
                result['test_result'] = {
                    'input': test_word,
                    'lemmas': test_result[0][1] if test_result[0][1] else []
                }
                logger.info(f"✅ Zeyrek test başarılı: '{test_word}' -> {result['test_result']['lemmas']}")
            else:
                result['test_result'] = {'input': test_word, 'lemmas': []}
                logger.warning(f"⚠️ Zeyrek test sonucu boş: '{test_word}'")
        else:
            result['error'] = "Zeyrek yüklenemedi"
            logger.warning("❌ Zeyrek kullanılamıyor")
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Zeyrek test hatası: {e}")
    
    return result


def _simple_noun_stem(word: str) -> List[str]:
    """Simple rule-based Turkish noun stemmer.
    
    İsim eklerini çıkararak olası kökleri döndürür.
    
    Args:
        word: Input word (e.g., "gözleri", "evden", "taştan")
        
    Returns:
        List of possible noun stems.
    """
    if len(word) < 3:
        return [word]
    
    stems = set()
    original = word
    
    # İsim ekleri (uzundan kısaya sıralı - önemli!)
    noun_suffixes = [
        # Çoğul + hal ekleri kombinasyonları
        'larından', 'lerinden', 'larında', 'lerinde',
        'larına', 'lerine', 'larını', 'lerini',
        'lardan', 'lerden', 'larda', 'lerde',
        'lara', 'lere', 'ları', 'leri',
        
        # İyelik + hal ekleri kombinasyonları  
        'ından', 'inden', 'undan', 'ünden',
        'ımdan', 'imden', 'umdan', 'ümden',
        'ında', 'inde', 'unda', 'ünde',
        'ımda', 'imde', 'umda', 'ümde',
        'ına', 'ine', 'una', 'üne',
        'ıma', 'ime', 'uma', 'üme',
        'ını', 'ini', 'unu', 'ünü',
        'ımı', 'imi', 'umu', 'ümü',
        
        # Hal ekleri
        'dan', 'den', 'tan', 'ten',  # ayrılma hali
        'ndan', 'nden',  # ünlüyle biten kelimeler için
        'da', 'de', 'ta', 'te',      # bulunma hali
        'nda', 'nde',  # ünlüyle biten kelimeler için
        
        # Yönelme hali
        'na', 'ne',  # ünlüyle biten kelimeler için
        'ya', 'ye',  # ünlüyle biten kelimeler için
        'a', 'e',
        
        # Çoğul
        'lar', 'ler',
        
        # İyelik ekleri (tek başına)
        'ım', 'im', 'um', 'üm',  # 1. tekil
        'ın', 'in', 'un', 'ün',  # 2. tekil
        'ımız', 'imiz', 'umuz', 'ümüz',  # 1. çoğul
        'ınız', 'iniz', 'unuz', 'ünüz',  # 2. çoğul
        
        # Belirtme hali / 3. tekil iyelik
        'nı', 'ni', 'nu', 'nü',  # ünlüyle biten
        'yı', 'yi', 'yu', 'yü',  # ünlüyle biten
        'sı', 'si', 'su', 'sü',  # 3. tekil iyelik (ünlüyle biten)
        'ı', 'i', 'u', 'ü',
    ]
    
    for suffix in noun_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stem = word[:-len(suffix)]
            if len(stem) >= 2:
                stems.add(stem)
                # Bazı durumlarda ünlü düşmesi olabilir (oğul -> oğlu -> oğl)
                # Bu durumda son sessizi düşürüp ünlü ekleyebiliriz
    
    # Orijinal kelimeyi de ekle (ek bulunamadıysa)
    if not stems:
        stems.add(original)
    
    return list(stems)


def _simple_verb_stem(word: str) -> List[str]:
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
    verb_suffixes = [
        # -yor forms (geniş zaman + şimdiki)
        'ıyordum', 'iyordum', 'uyordum', 'üyordum',
        'ıyorsun', 'iyorsun', 'uyorsun', 'üyorsun',
        'ıyoruz', 'iyoruz', 'uyoruz', 'üyoruz',
        'ıyorlar', 'iyorlar', 'uyorlar', 'üyorlar',
        'ıyorum', 'iyorum', 'uyorum', 'üyorum',
        'ıyor', 'iyor', 'uyor', 'üyor',
        
        # Geçmiş zaman (-dı/-di)
        'dılar', 'diler', 'dular', 'düler',
        'tılar', 'tiler', 'tular', 'tüler',
        'dık', 'dik', 'duk', 'dük',
        'tık', 'tik', 'tuk', 'tük',
        'dım', 'dim', 'dum', 'düm',
        'tım', 'tim', 'tum', 'tüm',
        'dın', 'din', 'dun', 'dün',
        'tın', 'tin', 'tun', 'tün',
        'dı', 'di', 'du', 'dü',
        'tı', 'ti', 'tu', 'tü',
        
        # Gelecek zaman
        'acağım', 'eceğim', 'acaksın', 'eceksin',
        'acağız', 'eceğiz', 'acaklar', 'ecekler',
        'acak', 'ecek',
        
        # Miş geçmiş
        'mışlar', 'mişler', 'muşlar', 'müşler',
        'mışım', 'mişim', 'muşum', 'müşüm',
        'mışsın', 'mişsin', 'muşsun', 'müşsün',
        'mış', 'miş', 'muş', 'müş',
        
        # Geniş zaman (-r)
        'ırlar', 'irler', 'urlar', 'ürler',
        'arlar', 'erler',
        'ırım', 'irim', 'urum', 'ürüm',
        'arım', 'erim',
        'ırsın', 'irsin', 'ursun', 'ürsün',
        'arsın', 'ersin',
        'ır', 'ir', 'ur', 'ür',
        'ar', 'er',
        
        # İstek kipi
        'ayım', 'eyim', 'alım', 'elim',
        'asın', 'esin', 'alar', 'eler',
        
        # Emir kipi
        'sın', 'sin', 'sun', 'sün',
        'ınız', 'iniz', 'unuz', 'ünüz',
        'sınlar', 'sinler', 'sunlar', 'sünler',
        
        # Mastar ve isim-fiil
        'mak', 'mek',
        'ma', 'me',
        'ış', 'iş', 'uş', 'üş',
    ]
    
    for suffix in verb_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            if len(stem) >= 2:
                # Add mastar eki (kalın/ince ünlü uyumu)
                if stem[-1] in 'aıou':
                    stems.add(stem + 'mak')
                else:
                    stems.add(stem + 'mek')
                # Kök halini de ekle
                stems.add(stem)
    
    return list(stems) if stems else [word]


def _simple_stem(word: str) -> List[str]:
    """Simple rule-based Turkish stemmer combining noun and verb stemming.
    
    Args:
        word: Input word
        
    Returns:
        List of possible stems.
    """
    if len(word) < 3:
        return [word]
    
    stems = set()
    
    # Hem isim hem fiil olarak dene
    noun_stems = _simple_noun_stem(word)
    verb_stems = _simple_verb_stem(word)
    
    stems.update(noun_stems)
    stems.update(verb_stems)
    
    # Orijinal kelimeyi de ekle
    stems.add(word)
    
    return list(stems)


def _zeyrek_lemmatize_with_timeout(word: str, timeout: float = 2.0) -> Optional[List[str]]:
    """Lemmatize with timeout using threading.
    
    Args:
        word: Input word.
        timeout: Timeout in seconds (varsayılan 2.0 saniye).
        
    Returns:
        List of lemmas or None if timeout.
    """
    global _zeyrek_stats
    _zeyrek_stats['calls'] += 1
    
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
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=lemmatize)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        _zeyrek_stats['timeouts'] += 1
        logger.debug(f"⏱️ Zeyrek timeout: '{word}' ({timeout}s)")
        return None
    
    if exception[0]:
        _zeyrek_stats['errors'] += 1
        logger.debug(f"❌ Zeyrek hata: '{word}' -> {exception[0]}")
        return None
    
    if result[0]:
        _zeyrek_stats['successes'] += 1
        logger.debug(f"✅ Zeyrek başarılı: '{word}' -> {result[0]}")
    
    return result[0]


def get_all_lemmas(word: str) -> List[str]:
    """Get all possible lemmas (root forms) of a Turkish word.
    
    Args:
        word: Input word (e.g., "konuştu", "geldim", "gözleri")
        
    Returns:
        List of possible lemmas or [word] if not found.
    """
    global _zeyrek_stats
    
    if not word or len(word) < 2:
        return [word]
    
    word_lower = word.lower()
    
    # Check cache first
    if word_lower in _lemma_cache:
        return _lemma_cache[word_lower]
    
    lemmas = []
    used_fallback = False
    
    # Try Zeyrek with timeout (artık 2.0 saniye)
    zeyrek_result = _zeyrek_lemmatize_with_timeout(word_lower, timeout=2.0)
    
    if zeyrek_result:
        lemmas.extend(zeyrek_result)
        logger.debug(f"Zeyrek sonucu: '{word_lower}' -> {zeyrek_result}")
    
    # If Zeyrek returned only the original word or nothing, try simple stemmer
    if not lemmas or lemmas == [word_lower]:
        used_fallback = True
        _zeyrek_stats['fallbacks'] += 1
        
        simple_stems = _simple_stem(word_lower)
        lemmas.extend(simple_stems)
        logger.debug(f"Fallback stemmer: '{word_lower}' -> {simple_stems}")
    
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


def clear_lemma_cache():
    """Lemma cache'ini temizle."""
    global _lemma_cache
    _lemma_cache.clear()
    logger.info("Lemma cache temizlendi")


def reset_zeyrek_stats():
    """Zeyrek istatistiklerini sıfırla."""
    global _zeyrek_stats
    _zeyrek_stats = {
        'calls': 0,
        'successes': 0,
        'timeouts': 0,
        'errors': 0,
        'fallbacks': 0
    }
    logger.info("Zeyrek istatistikleri sıfırlandı")
