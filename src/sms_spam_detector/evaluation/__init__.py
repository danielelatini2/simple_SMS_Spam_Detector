"""Model evaluation and testing module for SMS spam detection."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing import TextPreprocessor
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and testing utilities."""

    def __init__(self, preprocessor: TextPreprocessor = None):
        """Initialize the model evaluator.

        Args:
            preprocessor: Text preprocessor for handling new messages.
        """
        self.preprocessor = preprocessor or TextPreprocessor()

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Prediction probabilities (optional).

        Returns:
            Dict: Comprehensive evaluation metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics["auc_score"] = roc_auc_score(y_true, y_pred_proba[:, 1])

        # Generate classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=["ham", "spam"]
        )

        logger.info(f"Evaluation metrics calculated - F1: {metrics['f1_score']:.4f}, "
                   f"Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            title: str = "Confusion Matrix") -> None:
        """Plot confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix array.
            title: Plot title.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      title: str = "ROC Curve") -> None:
        """Plot ROC curve.

        Args:
            y_true: True labels.
            y_pred_proba: Prediction probabilities.
            title: Plot title.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2,
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def test_sample_messages(self, model, sample_messages: List[str]) -> Dict[str, Any]:
        """Test model on sample messages and return detailed results.

        Args:
            model: Trained spam classifier.
            sample_messages: List of test messages.

        Returns:
            Dict: Test results with predictions and probabilities.
        """
        # Preprocess messages
        processed_messages = [
            self.preprocessor.preprocess_text(msg) for msg in sample_messages
        ]

        # Convert to pandas Series for model compatibility
        messages_series = pd.Series(processed_messages)

        # Make predictions
        predictions = model.predict(messages_series)
        probabilities = model.predict_proba(messages_series)

        # Prepare results
        results = []
        for i, (original_msg, processed_msg, pred, prob) in enumerate(
            zip(sample_messages, processed_messages, predictions, probabilities)
        ):
            result = {
                "message_id": i + 1,
                "original_message": original_msg,
                "processed_message": processed_msg,
                "prediction": "spam" if pred == 1 else "ham",
                "confidence": max(prob),
                "spam_probability": prob[1],
                "ham_probability": prob[0]
            }
            results.append(result)

        return {
            "test_results": results,
            "summary": {
                "total_messages": len(sample_messages),
                "spam_detected": sum(predictions),
                "ham_detected": len(predictions) - sum(predictions)
            }
        }

    def print_test_results(self, test_results: Dict[str, Any]) -> None:
        """Print formatted test results.

        Args:
            test_results: Results from test_sample_messages.
        """
        print("=" * 80)
        print("SMS SPAM DETECTION TEST RESULTS")
        print("=" * 80)

        for result in test_results["test_results"]:
            print(f"\nMessage {result['message_id']}:")
            print(f"Original: {result['original_message']}")
            print(f"Prediction: {result['prediction'].upper()} "
                  f"(Confidence: {result['confidence']:.4f})")
            print(f"Spam Probability: {result['spam_probability']:.4f}")
            print("-" * 80)

        summary = test_results["summary"]
        print(f"\nSUMMARY:")
        print(f"Total Messages: {summary['total_messages']}")
        print(f"Spam Detected: {summary['spam_detected']}")
        print(f"Ham Detected: {summary['ham_detected']}")

    def generate_evaluation_report(self, model, X_test: pd.Series, y_test: np.ndarray) -> str:
        """Generate a comprehensive evaluation report.

        Args:
            model: Trained spam classifier.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            str: Formatted evaluation report.
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba)

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("SMS SPAM DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Test Set Size: {len(X_test)}")
        report.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall: {metrics['recall']:.4f}")
        report.append(f"F1-Score: {metrics['f1_score']:.4f}")

        if "auc_score" in metrics:
            report.append(f"AUC Score: {metrics['auc_score']:.4f}")

        report.append("\nConfusion Matrix:")
        report.append(str(metrics['confusion_matrix']))

        report.append("\nDetailed Classification Report:")
        report.append(metrics['classification_report'])

        return "\n".join(report)
