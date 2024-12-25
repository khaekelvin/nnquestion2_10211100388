#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up directories..."
mkdir -p data/raw
mkdir -p models/saved_models
mkdir -p static/images

echo "Setting permissions..."
chmod +x start.sh

echo "Training model..."
python train.py