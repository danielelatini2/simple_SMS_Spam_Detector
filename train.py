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

    # Download dataset if not exists
    dataset_file = data_loader.extract_path / "SMSSpamCollection"
    if not dataset_file.exists():
        logger.info("Dataset not found, downloading...")
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
    classifier = SpamClassifier()

    # Train with grid search
    training_results = classifier.train_with_grid_search(
        df_clean["processed_message"],
        df_clean["label"]
    )

    logger.info(f"Training results: {training_results}")

    # Step 4: Model Evaluation
    logger.info("Step 4: Model evaluation")
    evaluator = ModelEvaluator(preprocessor)

    # Evaluate on split
    evaluation_results = classifier.evaluate_on_split(
        df_clean["processed_message"],
        df_clean["label"]
    )

    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        classifier,
        df_clean["processed_message"],
        classifier.prepare_labels(df_clean["label"])
    )

    print(report)

    # Step 5: Test on sample messages
    logger.info("Step 5: Testing on sample messages")
    sample_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
    ]

    test_results = evaluator.test_sample_messages(classifier, sample_messages)
    evaluator.print_test_results(test_results)

    # Step 6: Save model
    logger.info("Step 6: Saving model")
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "spam_classifier.joblib"
    classifier.save_model(str(model_path))

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
