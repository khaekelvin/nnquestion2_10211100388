#!/bin/bash
# filepath: build.sh
set -e

echo "Setting up environment..."
PROJECT_ROOT=$(pwd)

echo "Creating directories..."
mkdir -p ${PROJECT_ROOT}/data/raw
mkdir -p ${PROJECT_ROOT}/models/saved_models
mkdir -p ${PROJECT_ROOT}/static/images

echo "Copying dataset..."
if [ -f "${PROJECT_ROOT}/data/sample/Loandataset.csv" ]; then
    cp "${PROJECT_ROOT}/data/sample/Loandataset.csv" "${PROJECT_ROOT}/data/raw/"
else
    echo "Error: Sample dataset not found!"
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model..."
python train.py