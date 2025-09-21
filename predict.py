"""Prediction script for SMS spam detection."""

import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List, Union

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sms_spam_detector import TextPreprocessor, SpamClassifier


def setup_logging():
    """Configure logging for predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class SMSSpamPredictor:
    """Easy-to-use interface for SMS spam prediction."""

    def __init__(self, model_path: str = None):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model file. If None, uses default location.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        if model_path is None:
            # Use path relative to script location
            script_dir = Path(__file__).parent
            model_path = script_dir / "models" / "spam_classifier.joblib"
        
        # Verify model file exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
        # Disable internal preprocessing since we'll preprocess externally
        self.classifier = SpamClassifier(preprocessing_enabled=False)

        # Load trained model
        try:
            self.classifier.load_model(str(model_path))
            self.logger.info(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict_single(self, message: str) -> dict:
        """Predict spam/ham for a single message.

        Args:
            message: SMS message to classify.

        Returns:
            dict: Prediction result with confidence scores.
        """
        # Validate input
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")
        
        # Preprocess message
        processed_msg = self.preprocessor.preprocess_text(message)

        # Make prediction
        msg_series = pd.Series([processed_msg])
        prediction = self.classifier.predict(msg_series)[0]
        probabilities = self.classifier.predict_proba(msg_series)[0]

        return {
            "message": message,
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence": max(probabilities),
            "spam_probability": probabilities[1],
            "ham_probability": probabilities[0]
        }

    def predict_batch(self, messages: List[str]) -> List[dict]:
        """Predict spam/ham for multiple messages efficiently.

        Args:
            messages: List of SMS messages to classify.

        Returns:
            List[dict]: List of prediction results.
        """
        if not messages:
            return []
        
        # Validate and filter empty messages
        valid_messages = []
        for i, msg in enumerate(messages):
            if not msg or not msg.strip():
                self.logger.warning(f"Skipping empty message at index {i}")
                continue
            valid_messages.append(msg)
        
        if not valid_messages:
            return []
        
        # Preprocess all messages at once
        processed_messages = [
            self.preprocessor.preprocess_text(msg) for msg in valid_messages
        ]
        
        # Batch prediction
        msg_series = pd.Series(processed_messages)
        predictions = self.classifier.predict(msg_series)
        probabilities = self.classifier.predict_proba(msg_series)
        
        # Build results
        results = []
        for i, msg in enumerate(valid_messages):
            pred = predictions[i]
            probs = probabilities[i]
            results.append({
                "message": msg,
                "prediction": "spam" if pred == 1 else "ham",
                "confidence": max(probs),
                "spam_probability": probs[1],
                "ham_probability": probs[0]
            })
        
        return results

    def print_prediction(self, result: dict) -> None:
        """Print formatted prediction result.

        Args:
            result: Prediction result dictionary.
        """
        print(f"Message: {result['message'][:80]}{'...' if len(result['message']) > 80 else ''}")
        print(f"Prediction: {result['prediction'].upper()} "
              f"(Confidence: {result['confidence']:.4f})")
        print(f"Spam Probability: {result['spam_probability']:.4f}")
        print(f"Ham Probability: {result['ham_probability']:.4f}")
        print("-" * 60)

    def is_model_loaded(self) -> bool:
        """Check if the model is properly loaded."""
        return hasattr(self.classifier, 'best_model') and self.classifier.best_model is not None


def main():
    """Interactive prediction interface."""
    setup_logging()

    print("SMS Spam Detection System")
    print("=" * 50)

    # Initialize predictor
    try:
        predictor = SMSSpamPredictor()
        print("Model loaded successfully!")
        
        # Verify model is properly loaded
        if not predictor.is_model_loaded():
            print("Warning: Model appears to be empty. Please retrain the model.")
            return
            
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please run train.py first to train the model.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model file and try again.")
        return

    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Predict single message")
        print("2. Test sample messages")
        print("3. Batch predict from input")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            message = input("\nEnter SMS message: ").strip()
            if message:
                try:
                    result = predictor.predict_single(message)
                    print("\nPrediction Result:")
                    predictor.print_prediction(result)
                except Exception as e:
                    print(f"Error during prediction: {e}")
            else:
                print("Please enter a valid message.")

        elif choice == "2":
            sample_messages = [
                "Congratulations! You've won $1000! Click here to claim now.",
                "Hey, what time should we meet for dinner?",
                "URGENT: Your bank account will be closed. Verify now!",
                "Thanks for the meeting today. See you next week.",
                "WIN BIG! Play our casino games and get free spins!",
                "Can you pick up some milk on your way home?",
                "FINAL NOTICE: Your subscription expires today. Renew now!",
                "Happy birthday! Hope you have a great day!"
            ]

            print("\nTesting sample messages:")
            print("=" * 60)

            try:
                results = predictor.predict_batch(sample_messages)
                for result in results:
                    predictor.print_prediction(result)
            except Exception as e:
                print(f"Error during batch prediction: {e}")

        elif choice == "3":
            print("\nEnter multiple messages (one per line, empty line to finish):")
            messages = []
            while True:
                msg = input().strip()
                if not msg:
                    break
                messages.append(msg)
            
            if messages:
                try:
                    results = predictor.predict_batch(messages)
                    print(f"\nBatch Prediction Results ({len(results)} messages):")
                    print("=" * 60)
                    for result in results:
                        predictor.print_prediction(result)
                except Exception as e:
                    print(f"Error during batch prediction: {e}")
            else:
                print("No messages entered.")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
