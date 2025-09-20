"""Text preprocessing module for SMS spam detection."""

import re
import nltk
from typing import List, Union
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles all text preprocessing operations."""

    def __init__(self, download_nltk_data: bool = True):
        """Initialize the text preprocessor.

        Args:
            download_nltk_data: Whether to download required NLTK data.
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        if download_nltk_data:
            self._download_nltk_requirements()

    def _download_nltk_requirements(self) -> None:
        """Download necessary NLTK data files."""
        required_data = ["punkt", "punkt_tab", "stopwords"]

        for data_name in required_data:
            try:
                nltk.download(data_name, quiet=True)
                logger.info(f"Downloaded NLTK data: {data_name}")
            except Exception as e:
                logger.warning(f"Failed to download {data_name}: {str(e)}")

    def to_lowercase(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """Convert text to lowercase.

        Args:
            text: Text to convert (string or pandas Series).

        Returns:
            Lowercase text.
        """
        if isinstance(text, str):
            return text.lower()
        return text.str.lower()

    def remove_punctuation_and_numbers(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """Remove punctuation and numbers, keeping useful symbols like $ and !.

        Args:
            text: Text to clean (string or pandas Series).

        Returns:
            Cleaned text.
        """
        pattern = r"[^a-z\s$!]"

        if isinstance(text, str):
            return re.sub(pattern, "", text)
        return text.apply(lambda x: re.sub(pattern, "", x))

    def tokenize_text(self, text: Union[str, pd.Series]) -> Union[List[str], pd.Series]:
        """Tokenize text into individual words.

        Args:
            text: Text to tokenize (string or pandas Series).

        Returns:
            Tokenized text.
        """
        if isinstance(text, str):
            return word_tokenize(text)
        return text.apply(word_tokenize)

    def remove_stopwords(self, tokens: Union[List[str], pd.Series]) -> Union[List[str], pd.Series]:
        """Remove stopwords from tokenized text.

        Args:
            tokens: Tokenized text (list of strings or pandas Series).

        Returns:
            Text with stopwords removed.
        """
        if isinstance(tokens, list):
            return [word for word in tokens if word not in self.stop_words]
        return tokens.apply(lambda x: [word for word in x if word not in self.stop_words])

    def stem_tokens(self, tokens: Union[List[str], pd.Series]) -> Union[List[str], pd.Series]:
        """Apply stemming to reduce words to their base form.

        Args:
            tokens: Tokenized text (list of strings or pandas Series).

        Returns:
            Stemmed tokens.
        """
        if isinstance(tokens, list):
            return [self.stemmer.stem(word) for word in tokens]
        return tokens.apply(lambda x: [self.stemmer.stem(word) for word in x])

    def join_tokens(self, tokens: Union[List[str], pd.Series]) -> Union[str, pd.Series]:
        """Join tokens back into a single string.

        Args:
            tokens: Tokenized text (list of strings or pandas Series).

        Returns:
            Joined text string.
        """
        if isinstance(tokens, list):
            return " ".join(tokens)
        return tokens.apply(lambda x: " ".join(x))

    def preprocess_text(self, text: Union[str, pd.Series], verbose: bool = False) -> Union[str, pd.Series]:
        """Apply complete text preprocessing pipeline.

        Args:
            text: Text to preprocess (string or pandas Series).
            verbose: Whether to print intermediate steps.

        Returns:
            Fully preprocessed text.
        """
        if verbose and isinstance(text, pd.Series):
            logger.info("=== BEFORE PREPROCESSING ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 1: Lowercase
        text = self.to_lowercase(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER LOWERCASING ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 2: Remove punctuation and numbers
        text = self.remove_punctuation_and_numbers(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER REMOVING PUNCTUATION & NUMBERS ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 3: Tokenize
        text = self.tokenize_text(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER TOKENIZATION ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 4: Remove stopwords
        text = self.remove_stopwords(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER REMOVING STOP WORDS ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 5: Stem tokens
        text = self.stem_tokens(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER STEMMING ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        # Step 6: Join tokens back
        text = self.join_tokens(text)
        if verbose and isinstance(text, pd.Series):
            logger.info("=== AFTER JOINING TOKENS ===")
            logger.info(f"Sample: {text.head(3).tolist()}")

        return text
