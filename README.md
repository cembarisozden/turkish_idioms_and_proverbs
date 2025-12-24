# Turkish Idioms and Proverbs Detection Pipeline

## Setup

```bash
pip install torch transformers accelerate scikit-learn pandas numpy openpyxl
```

## Run Commands

### 1. Prepare Data
```bash
python scripts/run_prepare_data.py
```

### 2. Train Model
```bash
python scripts/run_train.py
```

### 3. Evaluate Model
```bash
python scripts/run_eval.py
```

### 4. Inference
```bash
python scripts/run_infer.py --text "Bugün yine eli kulağında ve kimse şaşırmadı."
```

Optional flags:
- `--threshold 0.6`: Set classification threshold (default: 0.6)
- `--token-window`: Use token window matching instead of exact matching

