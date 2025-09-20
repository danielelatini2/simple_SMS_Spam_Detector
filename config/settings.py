"""Configuration settings for SMS spam detection."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data settings
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_EXTRACT_PATH = RAW_DATA_DIR / "sms_spam_collection"
DATASET_FILE = DATASET_EXTRACT_PATH / "SMSSpamCollection"

# Model settings
MODEL_CONFIG = {
    "vectorizer": {
        "min_df": 1,
        "max_df": 0.9,
        "ngram_range": (1, 2)
    },
    "classifier": {
        "alpha_range": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    },
    "training": {
        "test_size": 0.2,
        "cv_folds": 5,
        "scoring": "f1",
        "random_state": 42
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": str(LOGS_DIR / "sms_spam_detection.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    "keep_symbols": ["$", "!"],
    "remove_pattern": r"[^a-z\s$!]",
    "nltk_downloads": ["punkt", "punkt_tab", "stopwords"],
    "language": "english"
}

# Create directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
