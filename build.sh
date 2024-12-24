#!/bin/bash
set -e

echo "Setting up environment..."
export RENDER=true

echo "Creating directories..."
mkdir -p data/raw models/saved_models static/images

echo "Copying dataset..."
cp Loandataset.csv data/raw/

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model..."
python train.py