"""Machine learning models module for SMS spam detection."""

import joblib
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import logging
import string

logger = logging.getLogger(__name__)


def preprocess_text(text):
    """Basic text preprocessing function.

    Args:
        text: Input text to preprocess

    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


class SpamClassifier:
    """SMS spam classification model with training and prediction capabilities."""

    def __init__(
        self,
        classifier=None,
        classifier_name: str = "classifier",
        vectorizer_params: Optional[Dict[str, Any]] = None,
        preprocessing_enabled: bool = True,
        vectorizer_type: str = "count"
    ):
        """Initialize the spam classifier.

        Args:
            classifier: Custom classifier instance. Defaults to MultinomialNB.
            classifier_name: Name for the classifier in the pipeline.
            vectorizer_params: Parameters for the vectorizer.
            preprocessing_enabled: Whether to include text preprocessing in pipeline.
            vectorizer_type: Type of vectorizer ('count' or 'tfidf').
        """
        # Set default classifier
        self.classifier = classifier if classifier is not None else MultinomialNB()
        self.classifier_name = classifier_name
        self.preprocessing_enabled = preprocessing_enabled
        self.vectorizer_type = vectorizer_type

        # Default vectorizer parameters
        self.vectorizer_params = vectorizer_params or {
            "min_df": 1,
            "max_df": 0.9,
            "ngram_range": (1, 2),
            "stop_words": "english"
        }

        self.pipeline = None
        self.best_model = None
        self.is_trained = False
        self.class_weights = None

    def create_pipeline(self) -> Pipeline:
        """Create the machine learning pipeline with optional preprocessing.

        Returns:
            Pipeline: Scikit-learn pipeline with optional preprocessing, vectorizer and classifier.
        """
        steps = []

        # Add preprocessing step if enabled
        if self.preprocessing_enabled:
            steps.append(("preprocessor", FunctionTransformer(
                lambda x: [preprocess_text(text) for text in x],
                validate=False
            )))

        # Add vectorizer
        if self.vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(**self.vectorizer_params)
        else:
            vectorizer = CountVectorizer(**self.vectorizer_params)

        steps.append(("vectorizer", vectorizer))

        # Add classifier
        steps.append((self.classifier_name, self.classifier))

        pipeline = Pipeline(steps)

        logger.info(f"Pipeline created successfully with steps: {[step[0] for step in steps]}")
        return pipeline

    def prepare_labels(self, labels: pd.Series) -> np.ndarray:
        """Convert string labels to binary format.

        Args:
            labels: Series containing 'spam' and 'ham' labels.

        Returns:
            np.ndarray: Binary labels (1 for spam, 0 for ham).
        """
        return labels.apply(lambda x: 1 if x == "spam" else 0).values

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets.

        Args:
            y: Binary labels array

        Returns:
            Dict[int, float]: Class weights dictionary
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))

        logger.info(f"Computed class weights: {class_weights}")
        return class_weights

    def train_with_grid_search(
        self,
        X: pd.Series,
        y: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int = 5,
        scoring: str = "f1",
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Tuple[float, Dict[str, Any]]:
        """Train the model using grid search with cross-validation.

        Args:
            X: Text data for training.
            y: Labels for training.
            param_grid: Grid of parameters for hyperparameter tuning.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.
            cv: Number of cross-validation folds.
            scoring: Scoring metric for grid search.
            n_jobs: Number of parallel jobs.
            verbose: Verbosity level.

        Returns:
            Tuple[float, Dict[str, Any]]: Test F1 score and best parameters.
        """
        logger.info("Starting grid search training")

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )

        # Create pipeline
        self.pipeline = self.create_pipeline()

        # Default expanded parameter grid
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )

        logger.info(f"Fitting grid search with {len(param_grid)} parameter combinations")
        grid_search.fit(X_train, y_train)

        # Store best model
        self.best_model = grid_search.best_estimator_
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)

        logger.info(f"Grid search completed. Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"Test F1 score: {test_f1:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

        return test_f1, grid_search.best_params_

    def train_with_class_weights(
        self,
        X: pd.Series,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        auto_weight: bool = True,
        custom_weights: Optional[Dict[int, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Train the model with class weight handling for imbalanced datasets.

        Args:
            X: Text data for training.
            y: Labels for training.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.
            auto_weight: Whether to automatically compute class weights.
            custom_weights: Custom class weights dictionary.

        Returns:
            Tuple[float, Dict[str, Any]]: Test F1 score and training metrics.
        """
        logger.info("Starting training with class weight handling")

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )

        # Compute or use provided class weights
        if auto_weight and hasattr(self.classifier, 'class_weight'):
            if custom_weights is None:
                self.class_weights = self.compute_class_weights(y_train)
            else:
                self.class_weights = custom_weights

            # Create a new classifier instance with class weights
            classifier_params = self.classifier.get_params()
            classifier_params['class_weight'] = self.class_weights
            self.classifier.set_params(**classifier_params)

        elif auto_weight:
            logger.warning(f"Classifier {type(self.classifier).__name__} does not support class_weight parameter")

        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.best_model = self.pipeline
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"Training with class weights completed. Test F1 score: {test_f1:.4f}")

        training_metrics = {
            'test_f1': test_f1,
            'classification_report': report,
            'class_weights': self.class_weights
        }

        return test_f1, training_metrics

    def generate_validation_curves(
        self,
        X: pd.Series,
        y: pd.Series,
        param_name: str,
        param_range: list,
        cv: int = 5,
        scoring: str = "f1",
        n_jobs: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate validation curves for hyperparameter analysis.

        Args:
            X: Text data for training.
            y: Labels for training.
            param_name: Name of parameter to vary.
            param_range: Range of parameter values to test.
            cv: Number of cross-validation folds.
            scoring: Scoring metric.
            n_jobs: Number of parallel jobs.

        Returns:
            Tuple: Train scores, validation scores, train scores std, validation scores std.
        """
        logger.info(f"Generating validation curves for parameter: {param_name}")

        # Prepare labels and pipeline
        y_binary = self.prepare_labels(y)
        pipeline = self.create_pipeline()

        # Generate validation curves
        train_scores, validation_scores = validation_curve(
            pipeline, X, y_binary, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=n_jobs
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)

        logger.info("Validation curves generated successfully")

        return train_scores_mean, validation_scores_mean, train_scores_std, validation_scores_std

    def train_simple(
        self,
        X: pd.Series,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> float:
        """Simple training method for testing purposes.

        Args:
            X: Text data for training.
            y: Labels for training.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.

        Returns:
            float: Test F1 score.
        """
        logger.info("Starting simple training")

        # Prepare labels
        y_binary = self.prepare_labels(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )

        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.best_model = self.pipeline
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)

        logger.info(f"Simple training completed. Test F1 score: {test_f1:.4f}")

        return test_f1

    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Get default expanded parameter grid based on classifier type.

        Returns:
            Dict[str, Any]: Parameter grid for grid search.
        """
        # Base vectorizer parameters
        base_params = {
            "vectorizer__min_df": [1, 2, 5],
            "vectorizer__max_df": [0.8, 0.9, 0.95],
            "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]
        }

        # Classifier-specific parameters
        classifier_type = type(self.classifier).__name__

        if classifier_type == "MultinomialNB":
            base_params[f"{self.classifier_name}__alpha"] = [0.1, 0.5, 1.0, 2.0]
        elif classifier_type == "LogisticRegression":
            base_params[f"{self.classifier_name}__C"] = [0.1, 1.0, 10.0, 100.0]
            base_params[f"{self.classifier_name}__penalty"] = ['l1', 'l2']
        elif classifier_type == "SVC":
            base_params[f"{self.classifier_name}__C"] = [0.1, 1.0, 10.0]
            base_params[f"{self.classifier_name}__kernel"] = ['linear', 'rbf']
        elif classifier_type == "RandomForestClassifier":
            base_params[f"{self.classifier_name}__n_estimators"] = [50, 100, 200]
            base_params[f"{self.classifier_name}__max_depth"] = [None, 10, 20]

        return base_params

    def predict(self, texts: Union[str, pd.Series, list]) -> np.ndarray:
        """Predict spam/ham for given texts.

        Args:
            texts: Text(s) to classify.

        Returns:
            np.ndarray: Binary predictions (1 for spam, 0 for ham).

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model must be trained before making predictions")

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()

        predictions = self.best_model.predict(texts)
        logger.info(f"Made predictions for {len(texts)} texts")

        return predictions

    def predict_proba(self, texts: Union[str, pd.Series, list]) -> np.ndarray:
        """Predict probabilities for given texts.

        Args:
            texts: Text(s) to classify.

        Returns:
            np.ndarray: Prediction probabilities.

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model must be trained before making predictions")

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()

        probabilities = self.best_model.predict_proba(texts)
        logger.info(f"Made probability predictions for {len(texts)} texts")

        return probabilities

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: Path to save the model.

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("No trained model to save")

        joblib.dump({
            'model': self.best_model,
            'classifier_name': self.classifier_name,
            'vectorizer_params': self.vectorizer_params,
            'preprocessing_enabled': self.preprocessing_enabled,
            'vectorizer_type': self.vectorizer_type,
            'class_weights': self.class_weights
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model.
        """
        model_data = joblib.load(filepath)

        self.best_model = model_data['model']
        self.classifier_name = model_data.get('classifier_name', 'classifier')
        self.vectorizer_params = model_data.get('vectorizer_params', {})
        self.preprocessing_enabled = model_data.get('preprocessing_enabled', True)
        self.vectorizer_type = model_data.get('vectorizer_type', 'count')
        self.class_weights = model_data.get('class_weights', None)
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Get feature importance for the trained model.

        Args:
            top_n: Number of top features to return.

        Returns:
            pd.DataFrame: Feature importance dataframe or None if not available.
        """
        if not self.is_trained or self.best_model is None:
            logger.warning("Model must be trained to get feature importance")
            return None

        try:
            # Get feature names from vectorizer
            vectorizer = self.best_model.named_steps['vectorizer']
            feature_names = vectorizer.get_feature_names_out()

            # Get feature importance based on classifier type
            classifier = self.best_model.named_steps[self.classifier_name]

            if hasattr(classifier, 'feature_log_prob_'):
                # For Naive Bayes
                importance = np.exp(classifier.feature_log_prob_[1]) - np.exp(classifier.feature_log_prob_[0])
            elif hasattr(classifier, 'coef_'):
                # For linear models
                importance = classifier.coef_[0]
            else:
                logger.warning(f"Feature importance not available for {type(classifier).__name__}")
                return None

            # Create dataframe with top features
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', key=abs, ascending=False).head(top_n)

            logger.info(f"Retrieved top {top_n} features")
            return feature_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
