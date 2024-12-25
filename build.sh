#!/bin/bash
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up directories..."
mkdir -p data/raw
mkdir -p models/saved_models
mkdir -p static/images

python train.py