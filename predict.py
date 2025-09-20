"""Prediction script for SMS spam detection."""

import sys
import logging
from pathlib import Path
from typing import List

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sms_spam_detector import TextPreprocessor, SpamClassifier, ModelEvaluator


def setup_logging():
    """Configure logging for predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class SMSSpamPredictor:
    """Easy-to-use interface for SMS spam prediction."""

    def __init__(self, model_path: str = "models/spam_classifier.joblib"):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model file.
        """
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
        self.classifier = SpamClassifier()
        self.evaluator = ModelEvaluator(self.preprocessor)

        # Load trained model
        self.classifier.load_model(model_path)

    def predict_single(self, message: str) -> dict:
        """Predict spam/ham for a single message.

        Args:
            message: SMS message to classify.

        Returns:
            dict: Prediction result with confidence scores.
        """
        # Preprocess message
        processed_msg = self.preprocessor.preprocess_text(message)

        # Make prediction
        import pandas as pd
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
        """Predict spam/ham for multiple messages.

        Args:
            messages: List of SMS messages to classify.

        Returns:
            List[dict]: List of prediction results.
        """
        results = []
        for message in messages:
            result = self.predict_single(message)
            results.append(result)

        return results

    def print_prediction(self, result: dict) -> None:
        """Print formatted prediction result.

        Args:
            result: Prediction result dictionary.
        """
        print(f"Message: {result['message']}")
        print(f"Prediction: {result['prediction'].upper()} "
              f"(Confidence: {result['confidence']:.4f})")
        print(f"Spam Probability: {result['spam_probability']:.4f}")
        print("-" * 60)


def main():
    """Interactive prediction interface."""
    setup_logging()

    print("SMS Spam Detection System")
    print("=" * 50)

    # Initialize predictor
    try:
        predictor = SMSSpamPredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train.py first to train the model.")
        return

    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Predict single message")
        print("2. Test sample messages")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            message = input("\nEnter SMS message: ").strip()
            if message:
                result = predictor.predict_single(message)
                print("\nPrediction Result:")
                predictor.print_prediction(result)

        elif choice == "2":
            sample_messages = [
                "Congratulations! You've won $1000! Click here to claim now.",
                "Hey, what time should we meet for dinner?",
                "URGENT: Your bank account will be closed. Verify now!",
                "Thanks for the meeting today. See you next week.",
                "WIN BIG! Play our casino games and get free spins!"
            ]

            print("\nTesting sample messages:")
            print("=" * 50)

            results = predictor.predict_batch(sample_messages)
            for result in results:
                predictor.print_prediction(result)

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
