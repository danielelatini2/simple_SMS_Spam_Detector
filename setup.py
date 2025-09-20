#!/usr/bin/env python3
"""Setup script for SMS Spam Detection System."""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install Python package requirements."""
    print("Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Error installing packages")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False
    return True

def run_tests():
    """Run unit tests to verify installation."""
    print("Running unit tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("✓ All tests passed")
    except subprocess.CalledProcessError:
        print("✗ Some tests failed")
        return False
    return True

def main():
    """Main setup function."""
    print("SMS Spam Detection System - Setup")
    print("=" * 40)

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        print("Setup failed during package installation")
        return

    # Download NLTK data
    if not download_nltk_data():
        print("Setup failed during NLTK data download")
        return

    # Run tests
    if not run_tests():
        print("Setup completed but some tests failed")
    else:
        print("\n" + "=" * 40)
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python train.py' to train the model")
        print("2. Run 'python predict.py' for interactive predictions")
        print("3. Explore 'notebooks/sms_spam_eda.ipynb' for analysis")

if __name__ == "__main__":
    main()
