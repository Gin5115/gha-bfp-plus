#!/bin/bash
set -e
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Retraining model..."
python model/train.py

echo "Build complete."
