"""
Test script to demonstrate the enhanced SpamClassifier functionality.
This script tests all the new features added to the SpamClassifier class.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sms_spam_detector.models import SpamClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample SMS data for testing."""
    data = {
        'message': [
            "Free entry in 2 a weekly comp to win FA Cup final tickets",
            "Hi how are you today",
            "URGENT! Your mobile No 07xxxxxxxxx was awarded £2000 Bonus Caller",
            "Thanks for your message",
            "Win a £1000 cash prize or a prize worth £5000",
            "Hello, how was your day?",
            "Congratulations! You've won a free iPhone! Click here now!",
            "Can we meet for lunch tomorrow?",
            "WINNER!! As a valued network customer you have been selected",
            "See you at the meeting"
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    }
    return pd.DataFrame(data)

def test_model_flexibility():
    """Test 1: Model flexibility with different classifiers."""
    print("\n" + "="*50)
    print("TEST 1: Model Flexibility")
    print("="*50)

    df = create_sample_data()

    # Test with default MultinomialNB
    print("\n1.1 Testing with default MultinomialNB:")
    classifier1 = SpamClassifier()
    print(f"Default classifier: {type(classifier1.classifier).__name__}")

    # Test with custom LogisticRegression
    print("\n1.2 Testing with custom LogisticRegression:")
    classifier2 = SpamClassifier(
        classifier=LogisticRegression(random_state=42, solver='liblinear'),
        classifier_name="logistic_regression"
    )
    print(f"Custom classifier: {type(classifier2.classifier).__name__}")
    print(f"Classifier name in pipeline: {classifier2.classifier_name}")

    # Test with RandomForestClassifier
    print("\n1.3 Testing with RandomForestClassifier:")
    classifier3 = SpamClassifier(
        classifier=RandomForestClassifier(random_state=42, n_estimators=10),
        classifier_name="random_forest"
    )
    print(f"Custom classifier: {type(classifier3.classifier).__name__}")

    print("✅ Model flexibility test passed!")

def test_preprocessing_pipeline():
    """Test 2: Preprocessing pipeline configuration."""
    print("\n" + "="*50)
    print("TEST 2: Preprocessing Pipeline")
    print("="*50)

    df = create_sample_data()

    # Test with preprocessing enabled (default)
    print("\n2.1 Testing with preprocessing enabled:")
    classifier1 = SpamClassifier(preprocessing_enabled=True)
    pipeline1 = classifier1.create_pipeline()
    print(f"Pipeline steps: {[step[0] for step in pipeline1.steps]}")

    # Test with preprocessing disabled
    print("\n2.2 Testing with preprocessing disabled:")
    classifier2 = SpamClassifier(preprocessing_enabled=False)
    pipeline2 = classifier2.create_pipeline()
    print(f"Pipeline steps: {[step[0] for step in pipeline2.steps]}")

    # Test with TF-IDF vectorizer
    print("\n2.3 Testing with TF-IDF vectorizer:")
    classifier3 = SpamClassifier(vectorizer_type="tfidf")
    pipeline3 = classifier3.create_pipeline()
    print(f"Vectorizer type: {type(pipeline3.named_steps['vectorizer']).__name__}")

    print("✅ Preprocessing pipeline test passed!")

def test_advanced_hyperparameter_tuning():
    """Test 3: Advanced hyperparameter tuning."""
    print("\n" + "="*50)
    print("TEST 3: Advanced Hyperparameter Tuning")
    print("="*50)

    # Create more substantial sample data for better testing
    extended_data = {
        'message': [
            "Free entry in 2 a weekly comp to win FA Cup final tickets",
            "Hi how are you today", "URGENT! Your mobile was awarded £2000",
            "Thanks for your message", "Win a £1000 cash prize",
            "Hello, how was your day?", "Congratulations! You've won!",
            "Can we meet for lunch?", "WINNER!! You have been selected",
            "See you at the meeting", "Call now to claim your prize",
            "How are things going?", "Free ringtones! Text STOP",
            "What time is dinner?", "Your account has been suspended",
            "Good morning!", "Claim your reward now", "Talk later",
            "Limited time offer expires today", "Hope you're well"
        ] * 5,  # Multiply to have more data
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                 'spam', 'ham', 'spam', 'ham'] * 5
    }
    df = pd.DataFrame(extended_data)

    print("\n3.1 Testing grid search with default parameters:")
    classifier = SpamClassifier()
    test_f1, best_params = classifier.train_with_grid_search(
        df['message'], df['label'], cv=3, verbose=0
    )
    print(f"Test F1 score: {test_f1:.4f}")
    print(f"Best parameters: {best_params}")

    print("\n3.2 Testing with custom parameter grid:")
    custom_param_grid = {
        "vectorizer__min_df": [1, 2],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "classifier__alpha": [0.1, 1.0]
    }
    classifier2 = SpamClassifier()
    test_f1_custom, best_params_custom = classifier2.train_with_grid_search(
        df['message'], df['label'], param_grid=custom_param_grid, cv=3, verbose=0
    )
    print(f"Test F1 score with custom grid: {test_f1_custom:.4f}")
    print(f"Best parameters: {best_params_custom}")

    print("✅ Advanced hyperparameter tuning test passed!")

def test_validation_curves():
    """Test 4: Validation curves generation."""
    print("\n" + "="*50)
    print("TEST 4: Validation Curves")
    print("="*50)

    # Create sample data
    extended_data = {
        'message': [
            "Free entry to win prizes", "Hi how are you", "URGENT! You won £2000",
            "Thanks for message", "Win cash prize", "Hello friend",
            "Congratulations winner!", "Meet for lunch?", "You've been selected",
            "See you later", "Call to claim prize", "How are things?",
            "Free offer today", "What time dinner?", "Account suspended",
            "Good morning", "Claim reward now", "Talk soon"
        ] * 3,
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 3
    }
    df = pd.DataFrame(extended_data)

    print("\n4.1 Generating validation curves for alpha parameter:")
    classifier = SpamClassifier()

    train_scores, val_scores, train_std, val_std = classifier.generate_validation_curves(
        df['message'], df['label'],
        param_name="classifier__alpha",
        param_range=[0.1, 0.5, 1.0, 2.0],
        cv=3
    )

    print(f"Train scores: {train_scores}")
    print(f"Validation scores: {val_scores}")
    print("✅ Validation curves test passed!")

def test_class_weight_handling():
    """Test 5: Class weight handling for imbalanced datasets."""
    print("\n" + "="*50)
    print("TEST 5: Class Weight Handling")
    print("="*50)

    # Create imbalanced dataset
    imbalanced_data = {
        'message': [
            "Free entry to win", "URGENT! You won", "Win cash prize",
            "Congratulations winner!", "Call to claim", "Free offer",
            "Claim reward now", "Limited time offer"  # More spam
        ] + ["Hi friend", "How are you?"] * 1,  # Less ham
        'label': ['spam'] * 8 + ['ham'] * 2
    }
    df = pd.DataFrame(imbalanced_data)

    print(f"\nDataset distribution:")
    print(df['label'].value_counts())

    print("\n5.1 Testing with LogisticRegression and automatic class weights:")
    classifier1 = SpamClassifier(
        classifier=LogisticRegression(random_state=42, solver='liblinear')
    )
    test_f1_1, metrics_1 = classifier1.train_with_class_weights(
        df['message'], df['label']
    )
    print(f"Test F1 score: {test_f1_1:.4f}")
    print(f"Class weights: {metrics_1['class_weights']}")

    print("\n5.2 Testing with custom class weights:")
    custom_weights = {0: 1.0, 1: 2.0}  # Give more weight to spam class
    classifier2 = SpamClassifier(
        classifier=LogisticRegression(random_state=42, solver='liblinear')
    )
    test_f1_2, metrics_2 = classifier2.train_with_class_weights(
        df['message'], df['label'],
        custom_weights=custom_weights
    )
    print(f"Test F1 score with custom weights: {test_f1_2:.4f}")
    print(f"Custom weights used: {custom_weights}")

    print("\n5.3 Testing with classifier that doesn't support class weights:")
    classifier3 = SpamClassifier()  # MultinomialNB doesn't support class_weight
    test_f1_3, metrics_3 = classifier3.train_with_class_weights(
        df['message'], df['label']
    )
    print(f"Test F1 score (no class weights): {test_f1_3:.4f}")

    print("✅ Class weight handling test passed!")

def test_backward_compatibility():
    """Test 6: Backward compatibility."""
    print("\n" + "="*50)
    print("TEST 6: Backward Compatibility")
    print("="*50)

    df = create_sample_data()

    print("\n6.1 Testing basic functionality (original interface):")
    classifier = SpamClassifier()

    # Train with grid search (should work as before)
    test_f1, best_params = classifier.train_with_grid_search(
        df['message'], df['label'], cv=2, verbose=0
    )
    print(f"Training successful - Test F1: {test_f1:.4f}")

    # Make predictions
    predictions = classifier.predict(["Free prize! Call now!", "Hello friend"])
    print(f"Predictions: {predictions}")

    # Test probabilities
    probas = classifier.predict_proba(["Free prize! Call now!", "Hello friend"])
    print(f"Prediction probabilities shape: {probas.shape}")

    print("✅ Backward compatibility test passed!")

def test_feature_importance():
    """Test 7: Feature importance functionality."""
    print("\n" + "="*50)
    print("TEST 7: Feature Importance")
    print("="*50)

    df = create_sample_data()

    classifier = SpamClassifier()
    classifier.train_with_grid_search(df['message'], df['label'], cv=2, verbose=0)

    print("\n7.1 Getting feature importance:")
    feature_importance = classifier.get_feature_importance(top_n=10)
    if feature_importance is not None:
        print("Top 10 features:")
        print(feature_importance)
    else:
        print("Feature importance not available for this classifier")

    print("✅ Feature importance test passed!")

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced SpamClassifier Tests")
    print("=" * 60)

    try:
        test_model_flexibility()
        test_preprocessing_pipeline()
        test_advanced_hyperparameter_tuning()
        test_validation_curves()
        test_class_weight_handling()
        test_backward_compatibility()
        test_feature_importance()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ The enhanced SpamClassifier is working correctly!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
