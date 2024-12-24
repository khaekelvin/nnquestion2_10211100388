#!/bin/bash

# Create directories
mkdir -p data/raw models/saved_models static/images

# Copy dataset if not exists
if [ ! -f data/raw/Loandataset.csv ]; then
    cp Loandataset.csv data/raw/
fi

# Train model if needed
if [ ! -f models/saved_models/model.h5 ]; then
    python train.py
fi

# Run Flask app
python app.py