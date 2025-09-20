# SMS Spam Detection System

A professional, modular SMS spam detection system using machine learning techniques for text classification.

## Project Structure

```
SMSpamDetection/
│
├── src/
│   └── sms_spam_detector/
│       ├── __init__.py
│       ├── data/
│       │   └── __init__.py          # Data loading and handling
│       ├── preprocessing/
│       │   └── __init__.py          # Text preprocessing utilities
│       ├── models/
│       │   └── __init__.py          # ML models and training
│       └── evaluation/
│           └── __init__.py          # Model evaluation and testing
│
├── config/
│   └── settings.py                  # Configuration settings
│
├── data/
│   ├── raw/                        # Raw datasets
│   └── processed/                  # Processed datasets
│
├── models/                         # Saved trained models
├── logs/                           # Application logs
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks for exploration
│
├── train.py                        # Main training script
├── predict.py                      # Prediction interface
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Professional Logging**: Comprehensive logging throughout the application
- **Configuration Management**: Centralized configuration settings
- **Data Pipeline**: Automated data downloading, loading, and preprocessing
- **Model Training**: Grid search hyperparameter optimization
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Interactive Prediction**: Easy-to-use prediction interface
- **Extensible Design**: Easy to add new features and models

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the complete training pipeline:

```bash
python train.py
```

This will:
- Download the SMS spam dataset
- Preprocess the text data
- Train the model with hyperparameter tuning
- Evaluate the model performance
- Save the trained model

### Making Predictions

Use the interactive prediction interface:

```bash
python predict.py
```

Or use the predictor programmatically:

```python
from predict import SMSSpamPredictor

predictor = SMSSpamPredictor()
result = predictor.predict_single("Congratulations! You won $1000!")
print(result)
```

### Using Individual Components

```python
from src.sms_spam_detector import DataLoader, TextPreprocessor, SpamClassifier

# Load data
loader = DataLoader()
df = loader.load_dataset()

# Preprocess text
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess_text(df['message'])

# Train model
classifier = SpamClassifier()
results = classifier.train_with_grid_search(clean_text, df['label'])
```

## Model Performance

The system uses a Naive Bayes classifier with the following features:
- Text preprocessing (lowercasing, punctuation removal, tokenization, stemming)
- N-gram features (unigrams and bigrams)
- TF-IDF vectorization
- Hyperparameter optimization via grid search

## Configuration

Modify `config/settings.py` to customize:
- Dataset sources
- Model parameters
- Preprocessing options
- File paths
- Logging settings

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Dependencies

- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- nltk: Natural language processing
- requests: HTTP requests for data download
- numpy: Numerical computations

## License

This project is open source and available under the MIT License.
