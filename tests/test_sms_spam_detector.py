"""Unit tests for SMS spam detection system."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.sms_spam_detector import DataLoader, TextPreprocessor, SpamClassifier, ModelEvaluator


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def setUp(self):
        self.data_loader = DataLoader()

    def test_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsInstance(self.data_loader, DataLoader)
        self.assertEqual(self.data_loader.data_url, DataLoader.DEFAULT_URL)

    def test_prepare_labels(self):
        """Test label preparation."""
        sample_data = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'],
            'message': ['test1', 'test2', 'test3', 'test4']
        })

        info = self.data_loader.get_dataset_info(sample_data)
        self.assertEqual(info['shape'], (4, 2))
        self.assertEqual(info['label_distribution']['ham'], 2)
        self.assertEqual(info['label_distribution']['spam'], 2)

    def test_clean_dataset(self):
        """Test dataset cleaning."""
        # Create sample data with duplicates
        sample_data = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'],
            'message': ['test1', 'test2', 'test1', 'test4']
        })

        cleaned_data = self.data_loader.clean_dataset(sample_data)
        self.assertEqual(len(cleaned_data), 3)  # One duplicate should be removed


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(download_nltk_data=False)

    def test_to_lowercase(self):
        """Test text lowercasing."""
        text = "HELLO WORLD!"
        result = self.preprocessor.to_lowercase(text)
        self.assertEqual(result, "hello world!")

    def test_remove_punctuation_and_numbers(self):
        """Test punctuation and number removal."""
        text = "hello world! call 123-456-7890 for $100"
        result = self.preprocessor.remove_punctuation_and_numbers(text)
        self.assertEqual(result, "hello world! call  for $")

    def test_join_tokens(self):
        """Test token joining."""
        tokens = ["hello", "world", "test"]
        result = self.preprocessor.join_tokens(tokens)
        self.assertEqual(result, "hello world test")


class TestSpamClassifier(unittest.TestCase):
    """Test cases for SpamClassifier class."""

    def setUp(self):
        self.classifier = SpamClassifier()

    def test_initialization(self):
        """Test SpamClassifier initialization."""
        self.assertIsInstance(self.classifier, SpamClassifier)
        self.assertFalse(self.classifier.is_trained)

    def test_prepare_labels(self):
        """Test label preparation."""
        labels = pd.Series(['ham', 'spam', 'ham', 'spam'])
        result = self.classifier.prepare_labels(labels)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = self.classifier.create_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0][0], 'vectorizer')
        self.assertEqual(pipeline.steps[1][0], 'classifier')


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""

    def setUp(self):
        self.evaluator = ModelEvaluator()

    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        self.assertIsInstance(self.evaluator, ModelEvaluator)
        self.assertIsInstance(self.evaluator.preprocessor, TextPreprocessor)

    def test_evaluate_predictions(self):
        """Test prediction evaluation."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = self.evaluator.evaluate_predictions(y_true, y_pred)

        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)

        # Check accuracy calculation
        expected_accuracy = 0.75  # 3 correct out of 4
        self.assertEqual(metrics['accuracy'], expected_accuracy)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_complete_pipeline(self):
        """Test the complete training pipeline with sample data."""
        # Create sample data
        sample_data = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'] * 10,  # 40 samples
            'message': [
                'hello how are you',
                'win money now call free',
                'meeting tomorrow at office',
                'urgent click link now'
            ] * 10
        })

        # Initialize components
        preprocessor = TextPreprocessor(download_nltk_data=False)
        classifier = SpamClassifier()

        # Preprocess data
        processed_messages = preprocessor.preprocess_text(sample_data['message'])

        # Train model (simple training for test)
        classifier.train_simple(processed_messages, sample_data['label'])

        # Verify model is trained
        self.assertTrue(classifier.is_trained)

        # Make predictions
        predictions = classifier.predict(processed_messages.head(5))
        self.assertEqual(len(predictions), 5)

        # Verify predictions are binary (0 or 1)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))


if __name__ == '__main__':
    unittest.main()
