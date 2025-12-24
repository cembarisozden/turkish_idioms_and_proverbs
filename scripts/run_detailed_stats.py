"""Script to calculate detailed statistics on test data."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.inference import IdiomDetector
from src.config import GENERATED_DATASET_PATH, DEFAULT_THRESHOLD
from src.utils.io import load_csv
from src.utils.logging import setup_logging
import logging

logger = setup_logging()

def calculate_detailed_stats(test_df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    """Calculate detailed statistics on test data.
    
    Args:
        test_df: Test DataFrame with 'text' and 'label' columns.
        threshold: Classification threshold.
        
    Returns:
        Dictionary with detailed statistics.
    """
    logger.info(f"Calculating detailed statistics with threshold={threshold}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Initialize detector
    detector = IdiomDetector(threshold=threshold)
    
    # Get predictions and scores
    all_predictions = []
    all_labels = []
    all_scores = []
    all_lexicon_found = []
    
    logger.info("Processing test samples...")
    for idx, row in test_df.iterrows():
        text = str(row['text'])
        label = int(row['label'])
        
        result = detector.detect(text, threshold=threshold)
        
        all_predictions.append(1 if result['has_idiom'] else 0)
        all_labels.append(label)
        all_scores.append(result['score'])
        all_lexicon_found.append(result['lexicon_found'])
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_lexicon_found = np.array(all_lexicon_found)
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC AUC (if binary classification)
    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
    except:
        roc_auc = None
    
    # Per-class statistics
    class_stats = {
        'No Idiom (Class 0)': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1': float(f1[0]),
            'support': int(support[0])
        },
        'Idiom (Class 1)': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1': float(f1[1]),
            'support': int(support[1])
        }
    }
    
    # Score statistics
    score_stats = {
        'mean': float(np.mean(all_scores)),
        'std': float(np.std(all_scores)),
        'min': float(np.min(all_scores)),
        'max': float(np.max(all_scores)),
        'median': float(np.median(all_scores)),
        'q25': float(np.percentile(all_scores, 25)),
        'q75': float(np.percentile(all_scores, 75))
    }
    
    # Score distribution by label
    scores_by_label = {
        'No Idiom (label=0)': {
            'mean': float(np.mean(all_scores[all_labels == 0])),
            'std': float(np.std(all_scores[all_labels == 0])),
            'min': float(np.min(all_scores[all_labels == 0])),
            'max': float(np.max(all_scores[all_labels == 0]))
        },
        'Idiom (label=1)': {
            'mean': float(np.mean(all_scores[all_labels == 1])),
            'std': float(np.std(all_scores[all_labels == 1])),
            'min': float(np.min(all_scores[all_labels == 1])),
            'max': float(np.max(all_scores[all_labels == 1]))
        }
    }
    
    # Lexicon statistics
    lexicon_stats = {
        'total_found': int(np.sum(all_lexicon_found)),
        'percentage_found': float(np.mean(all_lexicon_found) * 100),
        'found_in_true_positives': int(np.sum(all_lexicon_found[all_labels == 1])),
        'found_in_true_negatives': int(np.sum(all_lexicon_found[all_labels == 0]))
    }
    
    # Compile results
    stats = {
        'threshold': threshold,
        'total_samples': len(test_df),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'roc_auc': float(roc_auc) if roc_auc is not None else None
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'per_class_metrics': class_stats,
        'score_statistics': score_stats,
        'score_distribution_by_label': scores_by_label,
        'lexicon_statistics': lexicon_stats
    }
    
    return stats, all_labels, all_predictions, all_scores

def print_statistics(stats: dict):
    """Print statistics in a readable format."""
    print("\n" + "="*60)
    print("DETAYLI İSTATİSTİKLER")
    print("="*60)
    
    print(f"\n📊 Genel Metrikler (Threshold: {stats['threshold']})")
    print("-" * 60)
    overall = stats['overall_metrics']
    print(f"Accuracy:        {overall['accuracy']:.4f}")
    print(f"Precision (Macro): {overall['precision_macro']:.4f}")
    print(f"Recall (Macro):    {overall['recall_macro']:.4f}")
    print(f"F1 (Macro):        {overall['f1_macro']:.4f}")
    if overall['roc_auc']:
        print(f"ROC AUC:          {overall['roc_auc']:.4f}")
    
    print(f"\n📈 Confusion Matrix")
    print("-" * 60)
    cm = stats['confusion_matrix']
    print(f"True Negative (TN):  {cm['true_negative']:4d}")
    print(f"False Positive (FP): {cm['false_positive']:4d}")
    print(f"False Negative (FN): {cm['false_negative']:4d}")
    print(f"True Positive (TP):  {cm['true_positive']:4d}")
    
    print(f"\n📋 Sınıf Bazında Metrikler")
    print("-" * 60)
    for class_name, metrics in stats['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Support:   {metrics['support']}")
    
    print(f"\n📊 Skor İstatistikleri")
    print("-" * 60)
    score_stats = stats['score_statistics']
    print(f"Ortalama:  {score_stats['mean']:.4f}")
    print(f"Std Dev:   {score_stats['std']:.4f}")
    print(f"Min:       {score_stats['min']:.4f}")
    print(f"Max:       {score_stats['max']:.4f}")
    print(f"Median:    {score_stats['median']:.4f}")
    print(f"Q25:       {score_stats['q25']:.4f}")
    print(f"Q75:       {score_stats['q75']:.4f}")
    
    print(f"\n📊 Etiket Bazında Skor Dağılımı")
    print("-" * 60)
    for label_name, dist in stats['score_distribution_by_label'].items():
        print(f"\n{label_name}:")
        print(f"  Ortalama: {dist['mean']:.4f}")
        print(f"  Std Dev:  {dist['std']:.4f}")
        print(f"  Min:      {dist['min']:.4f}")
        print(f"  Max:      {dist['max']:.4f}")
    
    print(f"\n📚 Lexicon İstatistikleri")
    print("-" * 60)
    lex = stats['lexicon_statistics']
    print(f"Toplam Bulunan:           {lex['total_found']}")
    print(f"Yüzde Bulunan:            {lex['percentage_found']:.2f}%")
    print(f"True Positive'lerde:      {lex['found_in_true_positives']}")
    print(f"True Negative'lerde:      {lex['found_in_true_negatives']}")
    
    print("\n" + "="*60)

def main():
    """Main function to calculate detailed statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate detailed statistics on test data')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                       help=f'Classification threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    logger.info("Starting detailed statistics calculation...")
    
    # Load test set
    logger.info(f"Loading dataset from {GENERATED_DATASET_PATH}")
    df = load_csv(GENERATED_DATASET_PATH)
    test_df = df[df['split'] == 'test'].copy()
    
    # Calculate statistics
    stats, labels, predictions, scores = calculate_detailed_stats(test_df, args.threshold)
    
    # Print statistics
    print_statistics(stats)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {output_path}")
    else:
        # Save to default location
        output_path = Path(__file__).parent.parent / "artifacts" / "detailed_stats.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {output_path}")
    
    logger.info("Detailed statistics calculation completed!")

if __name__ == "__main__":
    main()

