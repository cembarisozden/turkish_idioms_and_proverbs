"""Weak labeling for distant supervision."""
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import re

from src.data.normalize_tr import normalize_turkish_text
from src.config import NUM_POSITIVE_EXAMPLES, NUM_NEGATIVE_EXAMPLES

logger = logging.getLogger(__name__)

# Turkish sentence templates
TEMPLATES = [
    "Bugün yine {EXPR} ve kimse şaşırmadı.",
    "O an {EXPR} deyince ortam gerildi.",
    "Her zaman {EXPR} derdi büyükannem.",
    "{EXPR} sözü çok doğru bir söz.",
    "Bu durumda {EXPR} demek gerekiyor.",
    "O zaman {EXPR} dedi ve herkes güldü.",
    "Böyle durumlarda {EXPR} derler.",
    "{EXPR} diye bir söz vardır.",
    "O, {EXPR} sözünü çok severdi.",
    "Bazen {EXPR} demek yeterli olur.",
    "Bu konuda {EXPR} sözü geçerli.",
    "{EXPR} dediğinde herkes anladı.",
    "O gün {EXPR} demişti bana.",
    "Bu sözü duyunca {EXPR} geldi aklıma.",
    "{EXPR} sözü bu durumu çok iyi açıklıyor.",
    "Her zaman {EXPR} derdi annem.",
    "O an {EXPR} dedi ve herkes sustu.",
    "Bu tür durumlarda {EXPR} demek gerekir.",
    "{EXPR} sözü çok anlamlı bir söz.",
    "O zaman {EXPR} dedi ve herkes onayladı.",
    "Böyle zamanlarda {EXPR} derler.",
    "{EXPR} diye bilinen bir söz var.",
    "O, {EXPR} sözünü sık sık kullanırdı.",
    "Bazen {EXPR} demek en doğrusu olur.",
    "Bu konuda {EXPR} sözü çok uygun.",
    "{EXPR} dediğinde kimse itiraz etmedi.",
    "O gün {EXPR} demişti ve çok haklıydı.",
    "Bu sözü duyunca {EXPR} hatırladım.",
    "{EXPR} sözü bu durumu mükemmel açıklıyor.",
    "Her zaman {EXPR} derdi babam.",
    "O an {EXPR} dedi ve herkes düşündü.",
    "Bu tür durumlarda {EXPR} demek mantıklı.",
    "{EXPR} sözü çok değerli bir söz.",
    "O zaman {EXPR} dedi ve herkes beğendi.",
    "Böyle zamanlarda {EXPR} derler genelde.",
    "{EXPR} diye meşhur bir söz var.",
    "O, {EXPR} sözünü her fırsatta söylerdi.",
    "Bazen {EXPR} demek yeterli oluyor.",
    "Bu konuda {EXPR} sözü çok yerinde.",
    "{EXPR} dediğinde herkes başını salladı.",
    "O gün {EXPR} demişti ve çok doğruydu.",
    "Bu sözü duyunca {EXPR} aklıma geldi.",
    "{EXPR} sözü bu durumu harika açıklıyor.",
    "Her zaman {EXPR} derdi dedem.",
    "O an {EXPR} dedi ve herkes gülümsedi.",
    "Bu tür durumlarda {EXPR} demek doğru olur.",
    "{EXPR} sözü çok bilge bir söz.",
    "O zaman {EXPR} dedi ve herkes onayladı.",
    "Böyle zamanlarda {EXPR} derler bazen.",
    "{EXPR} diye ünlü bir söz var.",
    "O, {EXPR} sözünü çok sever ve kullanırdı.",
    "Bazen {EXPR} demek en iyisi olur.",
    "Bu konuda {EXPR} sözü çok uygun düşüyor.",
    "{EXPR} dediğinde kimse karşı çıkmadı.",
    "O gün {EXPR} demişti ve çok haklıydı.",
    "Bu sözü duyunca {EXPR} hatırladım hemen.",
    "{EXPR} sözü bu durumu çok güzel açıklıyor.",
    "Her zaman {EXPR} derdi ninem.",
    "O an {EXPR} dedi ve herkes dikkat kesildi.",
    "Bu tür durumlarda {EXPR} demek gerekir.",
    "{EXPR} sözü çok önemli bir söz.",
    "O zaman {EXPR} dedi ve herkes memnun oldu.",
    "Böyle zamanlarda {EXPR} derler genellikle.",
    "{EXPR} diye bilinen güzel bir söz var.",
    "O, {EXPR} sözünü sıkça kullanırdı.",
    "Bazen {EXPR} demek yeterli olabiliyor.",
    "Bu konuda {EXPR} sözü çok yerinde bir söz.",
    "{EXPR} dediğinde herkes anladı ne demek istediğini.",
    "O gün {EXPR} demişti ve çok doğru bir şey söylemişti.",
    "Bu sözü duyunca {EXPR} aklıma geldi hemen.",
    "{EXPR} sözü bu durumu mükemmel bir şekilde açıklıyor.",
    "Her zaman {EXPR} derdi büyükbabam.",
    "O an {EXPR} dedi ve herkes sessizleşti.",
    "Bu tür durumlarda {EXPR} demek mantıklı olur.",
    "{EXPR} sözü çok değerli ve anlamlı bir söz.",
    "O zaman {EXPR} dedi ve herkes beğendi sözünü.",
    "Böyle zamanlarda {EXPR} derler çoğu zaman.",
    "{EXPR} diye meşhur ve güzel bir söz var.",
    "O, {EXPR} sözünü her fırsatta söylemeyi severdi.",
    "Bazen {EXPR} demek yeterli oluyor böyle durumlarda.",
]

# Negative templates (without idiom) - Çeşitli doğal Türkçe cümleler
NEGATIVE_TEMPLATES = [
    # Günlük konuşma örnekleri
    "Bugün yine normal bir gün geçti ve kimse şaşırmadı.",
    "O an bir şey söyleyince ortam gerildi.",
    "Her zaman böyle derdi büyükannem.",
    "Bu söz çok doğru bir söz.",
    "Bu durumda bir şey demek gerekiyor.",
    "O zaman bir şey dedi ve herkes güldü.",
    "Böyle durumlarda böyle derler.",
    "Böyle bir söz vardır.",
    "O, bu sözü çok severdi.",
    "Bazen bir şey demek yeterli olur.",
    "Bu konuda bu söz geçerli.",
    "Bir şey dediğinde herkes anladı.",
    "O gün bir şey demişti bana.",
    "Bu sözü duyunca bir şey geldi aklıma.",
    "Bu söz bu durumu çok iyi açıklıyor.",
    "Her zaman böyle derdi annem.",
    "O an bir şey dedi ve herkes sustu.",
    "Bu tür durumlarda bir şey demek gerekir.",
    "Bu söz çok anlamlı bir söz.",
    "O zaman bir şey dedi ve herkes onayladı.",
    # Daha çeşitli doğal cümleler
    "Dün akşam markete gittim ve ekmek aldım.",
    "Yarın sabah erken kalkmam gerekiyor çünkü işe gitmem lazım.",
    "Bu kitabı okudum ve çok beğendim.",
    "Hava çok sıcak olduğu için dışarı çıkmadım.",
    "Arkadaşım bana telefon etti ve buluşmak istedi.",
    "Evde yemek yaptım ve ailemle birlikte yedik.",
    "Televizyonda güzel bir film izledim.",
    "Spor yapmak sağlık için çok önemlidir.",
    "Okula giderken otobüs kullandım.",
    "Hafta sonu sinemaya gitmeyi planlıyorum.",
    "Bu konuyu daha önce hiç duymamıştım.",
    "Kütüphanede sessizce kitap okudum.",
    "Yemek yaparken tuzu fazla koydum.",
    "Bahçede çiçek ektim ve suladım.",
    "Arkadaşlarımla parkta oyun oynadık.",
    "Geçen hafta tatil yaptım ve dinlendim.",
    "Bu soruyu çözmek için biraz düşünmem gerekiyor.",
    "Alışveriş yaparken liste hazırladım.",
    "Müzik dinlemeyi çok seviyorum.",
    "Yazın denize gitmek istiyorum.",
    "Kışın kar yağdığında çok mutlu oluyorum.",
    "Sabah kahvaltısında yumurta yedim.",
    "Akşam yemeğinden sonra yürüyüşe çıktım.",
    "Bu konuda daha fazla bilgi edinmem lazım.",
    "Projeyi tamamlamak için çok çalıştım.",
    "Toplantıda önemli kararlar alındı.",
    "Bu işi yapmak için zaman ayırmam gerekiyor.",
    "Hediye almak için mağazaya gittim.",
    "Çocuklar bahçede oyun oynuyorlar.",
    "Bu konuyu anlamak için daha fazla okumam lazım.",
    "Yeni bir dil öğrenmek istiyorum.",
    "Hafta içi her gün işe gidiyorum.",
    "Cumartesi günü temizlik yaptım.",
    "Pazar günü dinlenmeyi tercih ediyorum.",
    "Bu kitabı okumayı bitirdim.",
    "Yeni bir film izlemek istiyorum.",
    "Arkadaşım bana yardım etti.",
    "Bu konuda farklı düşünüyorum.",
    "Yemek yapmayı öğrenmek istiyorum.",
    "Spor salonuna gitmeye başladım.",
    "Bu şarkıyı çok seviyorum.",
    "Yaz tatilinde seyahat etmek istiyorum.",
    "Kışın sıcak içecekler içmeyi seviyorum.",
    "Bu konuda daha fazla araştırma yapmam gerekiyor.",
    "Projeyi zamanında bitirmek için çalışıyorum.",
    "Toplantıda önemli konular konuşuldu.",
    "Bu işi yapmak için hazırım.",
    "Hediye seçmek zor bir iş.",
    "Çocuklar okuldan geldiler.",
    "Bu konuyu anlamak kolay değil.",
    "Yeni bir hobi edinmek istiyorum.",
    "Hafta içi yoğun bir programım var.",
    "Cumartesi günü alışveriş yaptım.",
    "Pazar günü ailemle vakit geçirdim.",
]

def generate_positive_examples(lexicon: Dict[str, Dict], 
                               num_examples: int,
                               templates: List[str]) -> List[Dict]:
    """Generate positive examples by embedding idioms/proverbs into templates.
    
    Args:
        lexicon: Lexicon mapping normalized expressions to metadata.
        num_examples: Number of examples to generate.
        templates: List of sentence templates.
        
    Returns:
        List of dictionaries with text, label, expression, definition.
    """
    examples = []
    expressions = list(lexicon.keys())
    
    if not expressions:
        logger.warning("No expressions in lexicon for positive examples")
        return examples
    
    for _ in range(num_examples):
        # Random template
        template = random.choice(templates)
        
        # Random expression
        expr = random.choice(expressions)
        expr_original = lexicon[expr].get('original', expr)
        
        # Fill template
        text = template.format(EXPR=expr_original)
        
        examples.append({
            'text': text,
            'label': 1,
            'expression': expr_original,
            'definition': lexicon[expr].get('definition', '')
        })
    
    return examples

def extract_example_sentences_from_definition(definition: str) -> List[str]:
    """Extract example sentences from definition field.
    
    CSV'deki definition alanında HTML formatında örnek cümleler var:
    <i> 'Bunu başarmak için elinden geleni yapacaksın, dedi.' -</i>İ. O. Anar.
    
    Args:
        definition: Definition string that may contain example sentences.
        
    Returns:
        List of extracted example sentences.
    """
    if not definition or pd.isna(definition):
        return []
    
    sentences = []
    definition_str = str(definition)
    
    # Pattern 1: <i> 'cümle' -</i>Yazar. (en yaygın format)
    # Örnek: <i> 'Bunu başarmak için elinden geleni yapacaksın, dedi.' -</i>İ. O. Anar.
    pattern1 = r"<i>\s*['\"]([^'\"]+)['\"]\s*-</i>"
    matches1 = re.findall(pattern1, definition_str, re.IGNORECASE | re.DOTALL)
    sentences.extend(matches1)
    
    # Pattern 2: <i>cümle</i> (tırnak olmadan)
    # Örnek: <i> Elinden geleni geri koma!</i>
    pattern2 = r"<i>([^<]+)</i>"
    matches2 = re.findall(pattern2, definition_str, re.IGNORECASE | re.DOTALL)
    for match in matches2:
        # Remove author names and dashes (format: "cümle -Yazar.")
        cleaned = re.sub(r'\s*-\s*[A-ZİĞÜŞÇÖ][^.]*\.?\s*$', '', match.strip())
        cleaned = re.sub(r"^['\"]|['\"]$", '', cleaned)  # Remove quotes
        # Skip if it's already captured by pattern1 (contains quotes)
        if "'" not in cleaned and '"' not in cleaned and cleaned and len(cleaned) > 10:
            sentences.append(cleaned)
    
    # Pattern 3: 'cümle' format (without HTML)
    pattern3 = r"['\"]([^'\"]{15,})['\"]"  # At least 15 chars
    matches3 = re.findall(pattern3, definition_str)
    sentences.extend(matches3)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Remove common prefixes/suffixes
        sent = re.sub(r'^[:\-]\s*', '', sent)
        sent = re.sub(r'\s*[:\-]\s*$', '', sent)
        # Minimum length and basic quality checks
        if len(sent) >= 10 and len(sent) <= 200:
            # Should contain Turkish characters or be meaningful
            if any(char.isalpha() for char in sent):
                cleaned_sentences.append(sent)
    
    return cleaned_sentences

def generate_examples_from_csv_definitions(df: pd.DataFrame, 
                                         expr_col: str, 
                                         def_col: str) -> List[Dict]:
    """Generate training examples from CSV definition field example sentences.
    
    Args:
        df: DataFrame with expression and definition columns.
        expr_col: Name of expression column.
        def_col: Name of definition column.
        
    Returns:
        List of dictionaries with text, label, expression, definition.
    """
    examples = []
    total_extracted = 0
    
    for _, row in df.iterrows():
        expr = str(row[expr_col]) if pd.notna(row[expr_col]) else ""
        definition = str(row[def_col]) if pd.notna(row[def_col]) else ""
        
        if not expr or not definition:
            continue
        
        # Extract example sentences from definition
        example_sentences = extract_example_sentences_from_definition(definition)
        
        for example_sent in example_sentences:
            # ✅ Daha esnek eşleşme: Normalize edilmiş versiyonları kontrol et
            expr_normalized = normalize_turkish_text(expr)
            sent_normalized = normalize_turkish_text(example_sent)
            
            # Eşleşme kontrolü: 
            # 1. Orijinal deyim cümlede var mı?
            # 2. Normalize edilmiş deyim normalize edilmiş cümlede var mı?
            # 3. Değimin anahtar kelimeleri cümlede var mı? (daha esnek)
            expr_words = set(expr_normalized.split())
            sent_words = set(sent_normalized.split())
            
            # En az 2 kelime eşleşiyorsa veya tam eşleşme varsa kabul et
            common_words = expr_words.intersection(sent_words)
            word_match = len(common_words) >= min(2, len(expr_words) // 2)
            
            if (expr_normalized in sent_normalized or 
                expr in example_sent or 
                word_match):
                examples.append({
                    'text': example_sent,
                    'label': 1,
                    'expression': expr,
                    'definition': definition
                })
                total_extracted += 1
    
    logger.info(f"Extracted {total_extracted} example sentences from CSV definitions")
    return examples

def augment_with_turkish_inflections(expr: str) -> List[str]:
    """Generate Turkish inflected forms of an expression.
    
    Türkçe'de ekler çok önemli. Deyimler farklı çekimlerle kullanılabilir:
    - "elinden geleni yaptı" (geçmiş zaman)
    - "elinden geleni yapıyor" (şimdiki zaman)
    - "elinden geleni yapmış" (miş'li geçmiş)
    - "elinden geleni yapacak" (gelecek zaman)
    
    Args:
        expr: Expression to inflect.
        
    Returns:
        List of inflected forms.
    """
    inflected = [expr]  # Original
    
    # Eğer deyim bir fiil içeriyorsa, farklı çekimler dene
    # Basit bir yaklaşım: son kelimeyi değiştir
    
    # Geçmiş zaman ekleri
    past_suffixes = ['dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü']
    # Şimdiki zaman ekleri
    present_suffixes = ['yor', 'ıyor', 'iyor', 'uyor', 'üyor']
    # Gelecek zaman ekleri
    future_suffixes = ['acak', 'ecek']
    # Miş'li geçmiş ekleri
    misli_past_suffixes = ['mış', 'miş', 'muş', 'müş']
    
    # Basit bir yaklaşım: Eğer deyim "yapmak", "etmek" gibi fiillerle bitiyorsa
    # Farklı çekimler oluştur
    if expr.endswith('mak') or expr.endswith('mek'):
        base = expr[:-3]  # Remove "mak/mek"
        for suffix in past_suffixes[:2]:  # Sadece birkaç örnek
            inflected.append(base + suffix)
        for suffix in present_suffixes[:2]:
            inflected.append(base + suffix)
    
    return list(set(inflected))  # Remove duplicates

def generate_natural_positive_examples(lexicon: Dict[str, Dict],
                                      num_examples: int) -> List[Dict]:
    """✅ Generate positive examples using idioms in natural sentence contexts.
    
    Dataset'teki gerçek deyimleri doğal cümlelerde kullan.
    Farklı çekim ekleri ile çeşitlilik sağla.
    
    Args:
        lexicon: Lexicon mapping normalized expressions to metadata.
        num_examples: Number of examples to generate.
        
    Returns:
        List of dictionaries with text, label, expression, definition.
    """
    examples = []
    expressions = list(lexicon.keys())
    
    if not expressions:
        logger.warning("No expressions in lexicon for natural examples")
        return examples
    
    # Doğal cümle şablonları (deyimleri doğal bağlamda kullan)
    # ✅ İYİLEŞTİRİLDİ: Deyimleri cümle içinde doğal kullanım şablonları eklendi
    natural_contexts = [
        # Mevcut şablonlar (deyim tırnak içinde)
        "{EXPR} dedi ve herkes şaşırdı.",
        "O zaman {EXPR} demişti bana.",
        "Bu durumda {EXPR} demek gerekiyor.",
        "Her zaman {EXPR} derdi annem.",
        "O an {EXPR} dedi ve herkes güldü.",
        "Böyle durumlarda {EXPR} derler.",
        "{EXPR} sözü çok doğru bir söz.",
        "O, {EXPR} sözünü çok severdi.",
        "Bazen {EXPR} demek yeterli olur.",
        "Bu konuda {EXPR} sözü geçerli.",
        "{EXPR} dediğinde herkes anladı.",
        "O gün {EXPR} demişti ve çok haklıydı.",
        "Bu sözü duyunca {EXPR} hatırladım.",
        "{EXPR} sözü bu durumu çok iyi açıklıyor.",
        "Her zaman {EXPR} derdi büyükannem.",
        "O an {EXPR} dedi ve herkes sustu.",
        "Bu tür durumlarda {EXPR} demek gerekir.",
        "{EXPR} sözü çok anlamlı bir söz.",
        "O zaman {EXPR} dedi ve herkes onayladı.",
        "Böyle zamanlarda {EXPR} derler.",
        # Daha doğal cümleler
        "Dün {EXPR} dediğinde çok şaşırdım.",
        "Arkadaşım bana {EXPR} dedi ve güldük.",
        "Öğretmen {EXPR} dedi ve ders başladı.",
        "Büyükannem {EXPR} derdi her zaman.",
        "Bu konuda {EXPR} demek çok uygun.",
        "Toplantıda {EXPR} dedi ve herkes onayladı.",
        "Yemek yerken {EXPR} dedi ve güldük.",
        "Hava güzelken {EXPR} dedi ve dışarı çıktık.",
        "Kitap okurken {EXPR} aklıma geldi.",
        "Yürüyüş yaparken {EXPR} dedi arkadaşım.",
        # ✅ YENİ: Deyimleri cümle içinde doğal kullanım şablonları
        "İşi başarmak için {EXPR}.",
        "O {EXPR} ve başarılı oldu.",
        "Bu konuda {EXPR} gerekiyor.",
        "Her zaman {EXPR} ve sonuç aldı.",
        "Bunu yapmak için {EXPR}.",
        "O kişi {EXPR} ve işi halletti.",
        "Bu durumda {EXPR} şart.",
        "Başarılı olmak için {EXPR}.",
        "Sorunu çözmek için {EXPR}.",
        "Hedefe ulaşmak için {EXPR}.",
        "O {EXPR} çünkü çok istiyordu.",
        "Bu işi yapmak için {EXPR} çünkü önemli.",
        "Her zaman {EXPR} çünkü sorumluluk sahibi.",
        "O {EXPR} ve herkes memnun oldu.",
        "Bu projede {EXPR} gerekiyordu.",
        "Başarı için {EXPR} şarttı.",
        "O {EXPR} ve sonuç mükemmeldi.",
        "İşleri halletmek için {EXPR}.",
        "Hedeflere ulaşmak için {EXPR}.",
        "Sorunları çözmek için {EXPR}.",
        "Başarılı olmak istiyorsan {EXPR}.",
        "O {EXPR} ve başardı.",
        "Bu konuda {EXPR} lazım.",
        "Her zaman {EXPR} ve iyi sonuç aldı.",
        "Bunu yapabilmek için {EXPR}.",
        "O kişi {EXPR} ve başarılı oldu.",
        "Bu durumda {EXPR} gerekli.",
        "Başarılı olmak istiyorsan {EXPR} şart.",
        "Sorunları çözmek istiyorsan {EXPR}.",
        "Hedeflere ulaşmak istiyorsan {EXPR}.",
    ]
    
    for _ in range(num_examples):
        # Random expression
        expr = random.choice(expressions)
        expr_original = lexicon[expr].get('original', expr)
        
        # ✅ Ek çeşitliliği: Bazen farklı çekimler kullan
        # %70 orijinal, %30 çekimli versiyon
        if random.random() < 0.3:
            inflected_forms = augment_with_turkish_inflections(expr_original)
            expr_to_use = random.choice(inflected_forms)
        else:
            expr_to_use = expr_original
        
        # Random natural context
        context = random.choice(natural_contexts)
        
        # Fill context
        try:
            text = context.format(EXPR=expr_to_use)
        except KeyError:
            # Eğer şablon {EXPR} içermiyorsa, direkt kullan
            text = expr_to_use
        
        examples.append({
            'text': text,
            'label': 1,
            'expression': expr_original,  # Orijinal deyimi sakla
            'definition': lexicon[expr].get('definition', '')
        })
    
    return examples

def generate_negative_examples(num_examples: int,
                               templates: List[str]) -> List[Dict]:
    """Generate negative examples without idioms/proverbs.
    
    Args:
        num_examples: Number of examples to generate.
        templates: List of sentence templates.
        
    Returns:
        List of dictionaries with text, label, expression, definition.
    """
    examples = []
    
    for _ in range(num_examples):
        template = random.choice(templates)
        examples.append({
            'text': template,
            'label': 0,
            'expression': None,
            'definition': None
        })
    
    return examples

def generate_weak_labels(lexicon: Dict[str, Dict],
                        num_positive: int = NUM_POSITIVE_EXAMPLES,
                        num_negative: int = NUM_NEGATIVE_EXAMPLES,
                        use_natural_examples: bool = True) -> pd.DataFrame:
    """Generate weak labels for training using distant supervision.
    
    Args:
        lexicon: Lexicon mapping normalized expressions to metadata.
        num_positive: Number of positive examples.
        num_negative: Number of negative examples.
        use_natural_examples: Whether to include natural context examples from dataset.
        
    Returns:
        DataFrame with generated examples.
    """
    logger.info(f"Generating {num_positive} positive and {num_negative} negative examples")
    
    # ✅ Dataset'teki gerçek deyimleri kullan
    # %50 şablonlu, %50 doğal cümleler
    if use_natural_examples:
        template_count = num_positive // 2
        natural_count = num_positive - template_count
        
        logger.info(f"  - {template_count} template-based examples")
        logger.info(f"  - {natural_count} natural context examples")
        
        positive_template = generate_positive_examples(lexicon, template_count, TEMPLATES)
        positive_natural = generate_natural_positive_examples(lexicon, natural_count)
        positive = positive_template + positive_natural
    else:
        positive = generate_positive_examples(lexicon, num_positive, TEMPLATES)
    
    negative = generate_negative_examples(num_negative, NEGATIVE_TEMPLATES)
    
    all_examples = positive + negative
    random.shuffle(all_examples)
    
    df = pd.DataFrame(all_examples)
    logger.info(f"Generated {len(df)} examples (positive: {sum(df['label']==1)}, negative: {sum(df['label']==0)})")
    
    return df

