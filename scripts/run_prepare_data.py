"""Script to prepare data and build lexicon."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_dataset import load_and_prepare_dataset
from src.data.normalize_tr import normalize_turkish_text
from src.data.weak_labeling import generate_weak_labels, generate_examples_from_csv_definitions
from src.data.build_splits import split_dataset
from src.config import DATA_DIR, LEXICON_PATH, GENERATED_DATASET_PATH
from src.utils.io import save_json, save_csv
from src.utils.logging import setup_logging
import logging
import pandas as pd

logger = setup_logging()

def main():
    """Main function to prepare data."""
    logger.info("Starting data preparation...")
    
    # Load dataset
    logger.info("Loading dataset...")
    df, expr_col, def_col, type_col = load_and_prepare_dataset(DATA_DIR)
    
    # Build lexicon
    logger.info("Building lexicon...")
    lexicon = {}
    
    for _, row in df.iterrows():
        expr = str(row[expr_col])
        definition = str(row[def_col]) if pd.notna(row[def_col]) else ""
        expr_type = str(row[type_col]) if type_col and pd.notna(row[type_col]) else None
        
        # Normalize expression
        normalized = normalize_turkish_text(expr)
        
        if normalized and normalized not in lexicon:
            lexicon[normalized] = {
                'original': expr,
                'definition': definition,
                'type': expr_type
            }
    
    logger.info(f"Built lexicon with {len(lexicon)} expressions")
    
    # Save lexicon
    logger.info(f"Saving lexicon to {LEXICON_PATH}")
    save_json(lexicon, LEXICON_PATH)
    
    # ✅ Dataset'teki gerçek deyimleri doğrudan pozitif örnek olarak ekle
    logger.info("Adding real idioms/proverbs from dataset as positive examples...")
    real_examples = []
    seen_normalized = set()  # ✅ Duplicate kontrolü için
    
    for _, row in df.iterrows():
        expr = str(row[expr_col])
        definition = str(row[def_col]) if pd.notna(row[def_col]) else ""
        
        # ✅ Normalize edilmiş versiyonu kontrol et (duplicate önleme)
        normalized = normalize_turkish_text(expr)
        if normalized and normalized not in seen_normalized:
            seen_normalized.add(normalized)
            # Deyim/atasözünü doğrudan pozitif örnek olarak ekle
            real_examples.append({
                'text': expr,
                'label': 1,
                'expression': expr,
                'definition': definition
            })
    
    logger.info(f"Added {len(real_examples)} unique real examples from dataset")
    
    # ✅ CSV'deki definition alanından örnek cümleleri çıkar ve ekle
    logger.info("Extracting example sentences from CSV definitions...")
    csv_examples = generate_examples_from_csv_definitions(df, expr_col, def_col)
    logger.info(f"Extracted {len(csv_examples)} example sentences from CSV")
    
    # Generate weak labels (şablonlu ve doğal cümleler)
    logger.info("Generating weak labels...")
    generated_df = generate_weak_labels(lexicon, use_natural_examples=True)
    
    # ✅ Gerçek örnekleri ve CSV'den çıkarılan örnekleri de ekle
    real_df = pd.DataFrame(real_examples)
    csv_examples_df = pd.DataFrame(csv_examples)
    
    # Tüm pozitif örnekleri birleştir
    all_positive_df = pd.concat([generated_df[generated_df['label'] == 1], 
                                  real_df, 
                                  csv_examples_df], ignore_index=True)
    
    # Negatif örnekleri ekle
    negative_df = generated_df[generated_df['label'] == 0]
    generated_df = pd.concat([all_positive_df, negative_df], ignore_index=True)
    
    logger.info(f"Total examples: {len(generated_df)} (positive: {sum(generated_df['label']==1)}, negative: {sum(generated_df['label']==0)})")
    
    # ✅ Split dataset - ÖNCE split yap, SONRA split column ekle
    logger.info("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(generated_df)
    
    # Split column ekle ve birleştir
    train_df = train_df.copy()
    train_df['split'] = 'train'
    val_df = val_df.copy()
    val_df['split'] = 'val'
    test_df = test_df.copy()
    test_df['split'] = 'test'
    
    # Tüm split'leri birleştir
    generated_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save generated dataset
    logger.info(f"Saving generated dataset to {GENERATED_DATASET_PATH}")
    save_csv(generated_df, GENERATED_DATASET_PATH)
    
    logger.info("Data preparation completed!")

if __name__ == "__main__":
    main()

