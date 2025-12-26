"""Training script for transformer-based idiom/proverb detector."""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from src.config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD,
    DETECTOR_MODEL_PATH, RANDOM_SEED
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

set_seed(RANDOM_SEED)

class IdiomDataset(Dataset):
    """Dataset for idiom/proverb detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = MAX_LENGTH):
        """Initialize dataset.
        
        Args:
            texts: List of input texts.
            labels: List of labels (0 or 1).
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        eval_pred: Evaluation predictions tuple (predictions, labels).
        
    Returns:
        Dictionary of metrics.
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_detector(train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  output_dir: Path = DETECTOR_MODEL_PATH) -> None:
    """Train transformer-based detector.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns.
        val_df: Validation DataFrame with 'text' and 'label' columns.
        output_dir: Directory to save model.
    """
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=0.3,  # ✅ Artırıldı: 0.1 -> 0.3 (overfitting önleme)
        attention_probs_dropout_prob=0.2  # ✅ Artırıldı: 0.1 -> 0.2 (overfitting önleme)
    )
    
    # ✅ GPU kontrolü ve bilgilendirme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gpu = device.type == 'cuda'
    
    if is_gpu:
        logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  GPU not available, using CPU (will be slow)")
    
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = IdiomDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = IdiomDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    # ✅ GPU için optimize edilmiş batch size
    # GPU'da daha büyük batch size kullanabiliriz
    effective_batch_size = BATCH_SIZE
    if is_gpu:
        # GPU bellek durumuna göre batch size artırılabilir
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 8:  # 8GB+
            effective_batch_size = min(BATCH_SIZE * 2, 64)  # 2x veya max 64
            logger.info(f"   Using larger batch size for GPU: {effective_batch_size}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        learning_rate=LEARNING_RATE * 0.8,  # ✅ Biraz düşürüldü: overfitting önleme
        weight_decay=0.1,  # ✅ Artırıldı: 0.01 -> 0.1 (daha güçlü regularization)
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,
        # ✅ GPU için mixed precision (fp16)
        fp16=is_gpu,  # GPU varsa True, CPU'da False
        # ✅ GPU için pin memory (hızlı veri transferi)
        dataloader_pin_memory=is_gpu,  # GPU varsa True
        lr_scheduler_type='cosine',  # ✅ Daha yumuşak schedule
        warmup_ratio=0.1,  # ✅ Warmup ekle (%10 warmup)
        max_grad_norm=1.0,  # ✅ Gradient clipping: overfitting önleme
        # ✅ GPU için ek optimizasyonlar
        dataloader_num_workers=4 if is_gpu else 0,  # GPU'da paralel data loading
        report_to=None,  # TensorBoard kullanmıyorsak kapat
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )]
    )
    
    # Train
    logger.info("Starting training...")
    if is_gpu:
        logger.info("🚀 Training on GPU - Expected speedup: 50-100x faster!")
    train_result = trainer.train()
    
    # ✅ Overfitting kontrolü - train vs validation loss karşılaştırması
    if hasattr(train_result, 'train_loss') and hasattr(train_result, 'metrics'):
        train_loss = train_result.train_loss
        eval_loss = train_result.metrics.get('eval_loss', None)
        
        logger.info("=" * 60)
        logger.info("Training Results:")
        logger.info(f"Final Train Loss: {train_loss:.4f}")
        if eval_loss:
            logger.info(f"Final Validation Loss: {eval_loss:.4f}")
            loss_diff = abs(train_loss - eval_loss)
            logger.info(f"Loss Difference: {loss_diff:.4f}")
            
            # Overfitting uyarısı
            if loss_diff > 0.5:
                logger.warning("⚠️  OVERFITTING UYARISI: Train ve validation loss arasında büyük fark var!")
                logger.warning("   Model eğitim verilerini ezberlemiş olabilir.")
            elif loss_diff > 0.2:
                logger.warning("⚠️  Dikkat: Train ve validation loss arasında orta seviye fark var.")
            else:
                logger.info("✅ Train ve validation loss arasındaki fark normal görünüyor.")
        logger.info("=" * 60)
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed!")

