"""Data loading and handling module for SMS spam detection."""

import os
import requests
import zipfile
import io
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data downloading, loading, and basic preprocessing."""

    DEFAULT_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    DEFAULT_EXTRACT_PATH = "data/raw/sms_spam_collection"

    def __init__(self, data_url: str = DEFAULT_URL, extract_path: str = DEFAULT_EXTRACT_PATH):
        self.data_url = data_url
        self.extract_path = extract_path

    def download_dataset(self) -> bool:
        """Download the SMS spam dataset from UCI repository.

        Returns:
            bool: True if download successful, False otherwise.
        """
        try:
            logger.info(f"Downloading dataset from {self.data_url}")
            response = requests.get(self.data_url)

            if response.status_code == 200:
                logger.info("Download successful")
                self._extract_dataset(response.content)
                return True
            else:
                logger.error(f"Failed to download dataset. Status code: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False

    def _extract_dataset(self, content: bytes) -> None:
        """Extract the downloaded dataset.

        Args:
            content: Downloaded zip file content.
        """
        try:
            os.makedirs(self.extract_path, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(self.extract_path)
                logger.info("Dataset extraction successful")
        except Exception as e:
            logger.error(f"Error extracting dataset: {str(e)}")
            raise

    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load the SMS spam dataset into a pandas DataFrame.

        Args:
            file_path: Optional custom path to the dataset file.

        Returns:
            pd.DataFrame: Loaded dataset with 'label' and 'message' columns.
        """
        if file_path is None:
            file_path = os.path.join(self.extract_path, "SMSSpamCollection")

        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8"
            )

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset.

        Args:
            df: The dataset DataFrame.

        Returns:
            dict: Dataset statistics and information.
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "label_distribution": df["label"].value_counts().to_dict()
        }

        return info

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and handle missing values.

        Args:
            df: The dataset DataFrame.

        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        initial_shape = df.shape

        # Remove duplicates
        df_clean = df.drop_duplicates()

        # Handle missing values (if any)
        df_clean = df_clean.dropna()

        logger.info(f"Dataset cleaned. Shape changed from {initial_shape} to {df_clean.shape}")

        return df_clean
