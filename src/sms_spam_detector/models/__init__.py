"""Machine learning models module for SMS spam detection."""

import pickle
import joblib
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import logging

logger = logging.getLogger(__name__)


class SpamClassifier:
    """SMS spam classification model with training and prediction capabilities."""

    def __init__(self, vectorizer_params: Optional[Dict[str, Any]] = None):
        """Initialize the spam classifier.

        Args:
            vectorizer_params: Parameters for the CountVectorizer.
        """
        self.vectorizer_params = vectorizer_params or {
            "min_df": 1,
            "max_df": 0.9,
            "ngram_range": (1, 2)
        }

        self.pipeline = None
        self.best_model = None
        self.is_trained = False

    def create_pipeline(self) -> Pipeline:
        """Create the machine learning pipeline.

        Returns:
            Pipeline: Scikit-learn pipeline with vectorizer and classifier.
        """
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(**self.vectorizer_params)),
            ("classifier", MultinomialNB())
        ])

        logger.info("Pipeline created successfully")
        return pipeline

    def prepare_labels(self, labels: pd.Series) -> np.ndarray:
        """Convert string labels to binary format.

        Args:
            labels: Series containing 'spam' and 'ham' labels.

        Returns:
            np.ndarray: Binary labels (1 for spam, 0 for ham).
        """
        return labels.apply(lambda x: 1 if x == "spam" else 0).values

    def train_with_grid_search(
        self,
        X: pd.Series,
        y: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        scoring: str = "f1"
    ) -> Dict[str, Any]:
        """Train the model using grid search for hyperparameter tuning.

        Args:
            X: Feature data (text messages).
            y: Target labels.
            param_grid: Parameter grid for grid search.
            cv: Number of cross-validation folds.
            scoring: Scoring metric for grid search.

        Returns:
            Dict: Training results and best parameters.
        """
        if param_grid is None:
            param_grid = {
                "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
            }

        # Create pipeline
        self.pipeline = self.create_pipeline()

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Perform grid search
        logger.info("Starting grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        # Fit the grid search
        grid_search.fit(X, y_binary)

        # Store the best model
        self.best_model = grid_search.best_estimator_
        self.is_trained = True

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

        logger.info(f"Training completed. Best parameters: {results['best_params']}")
        logger.info(f"Best cross-validation score: {results['best_score']:.4f}")

        return results

    def train_simple(self, X: pd.Series, y: pd.Series) -> None:
        """Train the model without grid search (faster training).

        Args:
            X: Feature data (text messages).
            y: Target labels.
        """
        # Create pipeline
        self.pipeline = self.create_pipeline()

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Fit the pipeline
        logger.info("Starting simple training...")
        self.pipeline.fit(X, y_binary)

        self.best_model = self.pipeline
        self.is_trained = True

        logger.info("Simple training completed")

    def predict(self, messages: pd.Series) -> np.ndarray:
        """Predict spam/ham for new messages.

        Args:
            messages: Text messages to classify.

        Returns:
            np.ndarray: Predictions (1 for spam, 0 for ham).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.best_model.predict(messages)

    def predict_proba(self, messages: pd.Series) -> np.ndarray:
        """Get prediction probabilities for new messages.

        Args:
            messages: Text messages to classify.

        Returns:
            np.ndarray: Prediction probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.best_model.predict_proba(messages)

    def evaluate_on_split(self, X: pd.Series, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Evaluate the model on a train-test split.

        Args:
            X: Feature data (text messages).
            y: Target labels.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.

        Returns:
            Dict: Evaluation metrics and results.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )

        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

        results = {
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "test_size": len(X_test),
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }

        logger.info(f"Evaluation completed. F1-score: {f1:.4f}")

        return results

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model.
        """
        self.best_model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

    def get_feature_names(self) -> list:
        """Get feature names from the vectorizer.

        Returns:
            list: Feature names.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature names")

        vectorizer = self.best_model.named_steps["vectorizer"]
        return vectorizer.get_feature_names_out().tolist()
