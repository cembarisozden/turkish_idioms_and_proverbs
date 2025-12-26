"""Test script for Turkish stemmer and Zeyrek integration."""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.normalize_tr import (
    get_all_lemmas,
    check_zeyrek_status,
    get_zeyrek_stats,
    reset_zeyrek_stats,
    _simple_noun_stem,
    _simple_verb_stem,
)

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_zeyrek_status():
    """Zeyrek durumunu kontrol et."""
    print("\n" + "="*60)
    print("🔍 ZEYREK DURUM KONTROLÜ")
    print("="*60)
    
    status = check_zeyrek_status()
    
    print(f"  Yüklü: {'✅ Evet' if status['available'] else '❌ Hayır'}")
    print(f"  Aktif: {'✅ Evet' if status['loaded'] else '❌ Hayır'}")
    
    if status['test_result']:
        print(f"  Test: '{status['test_result']['input']}' -> {status['test_result']['lemmas']}")
    
    if status['error']:
        print(f"  ❌ Hata: {status['error']}")
    
    return status['available']


def test_noun_stemming():
    """İsim ekleri testleri."""
    print("\n" + "="*60)
    print("📝 İSİM EKLERİ TESTİ")
    print("="*60)
    
    test_cases = [
        # (kelime, beklenen_kök_içermeli)
        ("gözü", "göz"),
        ("gözleri", "göz"),
        ("gözlerinden", "göz"),
        ("evden", "ev"),
        ("evde", "ev"),
        ("eve", "ev"),
        ("taştan", "taş"),
        ("adamdan", "adam"),
        ("elinden", "el"),
        ("sözünü", "söz"),
        ("yolunda", "yol"),
        ("başından", "baş"),
        ("ayaktan", "ayak"),
        ("karnı", "kar"),  # veya karn
        ("pabucunda", "pabuc"),  # veya pabuç
    ]
    
    success_count = 0
    for word, expected_root in test_cases:
        stems = _simple_noun_stem(word)
        found = any(expected_root in stem or stem in expected_root for stem in stems)
        status = "✅" if found else "❌"
        if found:
            success_count += 1
        print(f"  {status} '{word}' -> {stems} (beklenen: '{expected_root}')")
    
    print(f"\n  Başarı oranı: {success_count}/{len(test_cases)}")


def test_verb_stemming():
    """Fiil ekleri testleri."""
    print("\n" + "="*60)
    print("📝 FİİL EKLERİ TESTİ")
    print("="*60)
    
    test_cases = [
        ("döndü", "dön"),
        ("konuştu", "konuş"),
        ("geldi", "gel"),
        ("gidiyor", "git"),
        ("bakıyorum", "bak"),
        ("almış", "al"),
        ("verecek", "ver"),
        ("düşmek", "düş"),
    ]
    
    success_count = 0
    for word, expected_root in test_cases:
        stems = _simple_verb_stem(word)
        found = any(expected_root in stem for stem in stems)
        status = "✅" if found else "❌"
        if found:
            success_count += 1
        print(f"  {status} '{word}' -> {stems} (beklenen: '{expected_root}')")
    
    print(f"\n  Başarı oranı: {success_count}/{len(test_cases)}")


def test_full_lemmatization():
    """get_all_lemmas fonksiyonu testleri."""
    print("\n" + "="*60)
    print("📝 TAM LEMMATIZATION TESTİ (Zeyrek + Fallback)")
    print("="*60)
    
    reset_zeyrek_stats()
    
    test_words = [
        "gözü",
        "gözleri", 
        "döndü",
        "dönmüş",
        "açlıktan",
        "konuştu",
        "evden",
        "taştan",
        "yolunda",
        "başından",
        "kararmış",
        "bakıyorum",
    ]
    
    for word in test_words:
        lemmas = get_all_lemmas(word)
        print(f"  '{word}' -> {lemmas}")
    
    # İstatistikleri göster
    stats = get_zeyrek_stats()
    print("\n  📊 Zeyrek İstatistikleri:")
    print(f"     Toplam çağrı: {stats['calls']}")
    print(f"     Başarılı: {stats['successes']}")
    print(f"     Timeout: {stats['timeouts']}")
    print(f"     Hata: {stats['errors']}")
    print(f"     Fallback kullanıldı: {stats['fallbacks']}")


def test_matching_scenario():
    """Deyim eşleştirme senaryosu testi."""
    print("\n" + "="*60)
    print("🎯 DEYİM EŞLEŞTIRME SENARYO TESTİ")
    print("="*60)
    
    # Senaryo: "gözü dönmek" deyimi, metinde "gözleri döndü" geçiyor
    lexicon_expr = "gözü dönmek"
    text_phrase = "gözleri döndü"
    
    print(f"\n  Lexicon'daki deyim: '{lexicon_expr}'")
    print(f"  Metindeki ifade: '{text_phrase}'")
    
    lexicon_words = lexicon_expr.split()
    text_words = text_phrase.split()
    
    print(f"\n  Kelime kelime karşılaştırma:")
    
    for lex_word, txt_word in zip(lexicon_words, text_words):
        lex_lemmas = set(get_all_lemmas(lex_word))
        txt_lemmas = set(get_all_lemmas(txt_word))
        
        intersection = lex_lemmas.intersection(txt_lemmas)
        match = "✅ EŞLEŞME VAR" if intersection else "❌ EŞLEŞME YOK"
        
        print(f"\n    '{lex_word}' lemmaları: {lex_lemmas}")
        print(f"    '{txt_word}' lemmaları: {txt_lemmas}")
        print(f"    Ortak: {intersection if intersection else 'Yok'}")
        print(f"    Sonuç: {match}")


def main():
    print("\n" + "🇹🇷 TÜRKÇE STEMMER VE ZEYREK TEST ARACI 🇹🇷")
    print("="*60)
    
    # Zeyrek durumunu kontrol et
    zeyrek_ok = test_zeyrek_status()
    
    # İsim ekleri testi
    test_noun_stemming()
    
    # Fiil ekleri testi
    test_verb_stemming()
    
    # Tam lemmatization testi
    test_full_lemmatization()
    
    # Eşleştirme senaryosu
    test_matching_scenario()
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

