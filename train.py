"""Main training script for SMS spam detection."""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sms_spam_detector import DataLoader, TextPreprocessor, SpamClassifier, ModelEvaluator


def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training pipeline."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting SMS spam detection training pipeline")

    # Step 1: Data Loading
    logger.info("Step 1: Loading data")
    data_loader = DataLoader()

    # Download dataset if not exists (with error handling)
    try:
        dataset_file = data_loader.extract_path / "SMSSpamCollection"
        if not dataset_file.exists():
            logger.info("Dataset not found, downloading...")
            data_loader.download_dataset()
    except Exception as e:
        logger.warning(f"Error checking dataset file: {e}")
        logger.info("Attempting to download dataset...")
        data_loader.download_dataset()

    # Load dataset
    df = data_loader.load_dataset()

    # Get dataset info
    dataset_info = data_loader.get_dataset_info(df)
    logger.info(f"Dataset info: {dataset_info}")

    # Clean dataset
    df_clean = data_loader.clean_dataset(df)

    # Step 2: Text Preprocessing
    logger.info("Step 2: Text preprocessing")
    preprocessor = TextPreprocessor()

    # Preprocess messages
    df_clean["processed_message"] = preprocessor.preprocess_text(
        df_clean["message"], verbose=True
    )

    # Step 3: Model Training  
    logger.info("Step 3: Model training")
    # Disable internal preprocessing since we already preprocessed
    classifier = SpamClassifier(preprocessing_enabled=False)

    # Train with grid search - properly handle return tuple
    f1_score, best_params = classifier.train_with_grid_search(
        df_clean["processed_message"],
        df_clean["label"]
    )

    logger.info(f"Training completed!")
    logger.info(f"Best F1 Score: {f1_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")

    # Step 4: Model Evaluation
    logger.info("Step 4: Model evaluation")
    evaluator = ModelEvaluator(preprocessor)

    # Generate evaluation report using test split data
    from sklearn.model_selection import train_test_split
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean["processed_message"],
        classifier.prepare_labels(df_clean["label"]),
        test_size=0.2,
        random_state=42,
        stratify=classifier.prepare_labels(df_clean["label"])
    )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(classifier, X_test, y_test)
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(report)

    # Step 5: Feature Importance Analysis
    logger.info("Step 5: Analyzing feature importance")
    try:
        top_features = classifier.get_feature_importance(top_n=15)
        if top_features is not None:
            print("\n" + "="*50)
            print("TOP 15 MOST IMPORTANT FEATURES")
            print("="*50)
            print(top_features.to_string())
        else:
            logger.info("Feature importance not available for this classifier")
    except Exception as e:
        logger.warning(f"Could not analyze feature importance: {e}")

    # Step 6: Test on sample messages
    logger.info("Step 6: Testing on sample messages")
    sample_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
        "Hi mom, I'll be home late tonight. Don't wait up for dinner.",
        "WINNER! You have been selected to receive a $5000 cash prize. Call 1-800-WINNER now!",
        "Meeting moved to 3pm in conference room B."
    ]

    test_results = evaluator.test_sample_messages(classifier, sample_messages)
    print("\n" + "="*50)
    print("SAMPLE MESSAGE PREDICTIONS")
    print("="*50)
    evaluator.print_test_results(test_results)

    # Step 7: Save model
    logger.info("Step 7: Saving model")
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "spam_classifier.joblib"
    classifier.save_model(str(model_path))
    
    logger.info(f"Model saved to: {model_path}")
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
