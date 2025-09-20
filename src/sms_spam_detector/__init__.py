"""SMS Spam Detection Package

A professional SMS spam detection system with modular architecture.
"""

__version__ = "1.0.0"
__author__ = "SMS Spam Detection Team"

from .data import DataLoader
from .preprocessing import TextPreprocessor
from .models import SpamClassifier
from .evaluation import ModelEvaluator

__all__ = ["DataLoader", "TextPreprocessor", "SpamClassifier", "ModelEvaluator"]
