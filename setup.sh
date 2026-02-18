#!/usr/bin/env bash

# This script is used to train the phishing URL detection model.
# It uses a virtual environment to manage dependencies.

# Check if a virtual environment exists, if not, create one.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install required libraries, including prompt_toolkit
echo "Installing required libraries..."
pip install pandas scikit-learn numpy tldextract requests rich prompt_toolkit

# Run the training script
echo "Training the model..."
python train.py "$@"

echo "Training complete. Model saved to model/logistic_regression_model.joblib (or random_forest_model.joblib)"
