"""Evaluation script for transformer-based detector."""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
from pathlib import Path
from typing import Dict
import json
import logging

from src.config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, METRICS_PATH, DETECTOR_MODEL_PATH
from src.models.train_detector import IdiomDataset

logger = logging.getLogger(__name__)

def evaluate_detector(test_df: pd.DataFrame,
                    model_path: Path = DETECTOR_MODEL_PATH) -> Dict[str, float]:
    """Evaluate detector on test set.
    
    Args:
        test_df: Test DataFrame with 'text' and 'label' columns.
        model_path: Path to saved model.
        
    Returns:
        Dictionary of metrics.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Create dataset
    test_dataset = IdiomDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluate
    logger.info("Evaluating...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    logger.info("Evaluation metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_predictions, target_names=['No Idiom', 'Idiom']))
    
    return metrics

def save_metrics(metrics: Dict[str, float], filepath: Path = METRICS_PATH) -> None:
    """Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics.
        filepath: Path to save file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {filepath}")

