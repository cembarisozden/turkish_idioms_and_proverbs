"""Script for inference on new text."""
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.inference import IdiomDetector
from src.config import DEFAULT_THRESHOLD
from src.utils.logging import setup_logging
import logging

logger = setup_logging()

def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description='Detect Turkish idioms/proverbs in text')
    parser.add_argument('--text', type=str, required=True, help='Input text to analyze')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                       help=f'Classification threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--no-token-window', action='store_true',
                       help='Disable token window matching (use exact matching instead)')
    
    args = parser.parse_args()
    
    # Initialize detector
    # ✅ Varsayılan olarak token window kullan (daha iyi eşleşme için)
    use_token_window = not args.no_token_window
    logger.info("Initializing detector...")
    detector = IdiomDetector(threshold=args.threshold, use_token_window=use_token_window)
    
    # Detect
    logger.info(f"Analyzing text: {args.text}")
    result = detector.detect(args.text, threshold=args.threshold)
    
    # Print JSON output
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

