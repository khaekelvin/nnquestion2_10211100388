#!/bin/bash
set -e  # Exit on error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating directories..."
mkdir -p data/raw models/saved_models static/images

echo "Copying dataset..."
cp Loandataset.csv data/raw/

echo "Training model..."
python train.py

echo "Build completed!"