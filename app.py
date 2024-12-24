import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import pandas as pd
from models.loan_predictor import LoanPredictor
from utils.data_preprocessing import preprocess_input
from utils.model_interpretation import generate_feature_importance_plot
import joblib
from config import FlaskConfig, ModelConfig

app = Flask(__name__)
app.config.from_object(FlaskConfig)
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Ensure required directories exist
os.makedirs('static/images', exist_ok=True)

def ensure_model_exists():
    """Ensure model and required files exist"""
    if not os.path.exists(ModelConfig.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {ModelConfig.DATA_PATH}")
    
    if not os.path.exists(ModelConfig.MODEL_PATH):
        logger.info("Training new model...")
        from train import main as train_model
        train_model()

try:
    ensure_model_exists()
    model = LoanPredictor.load_model(ModelConfig.MODEL_PATH)
    scaler = joblib.load(ModelConfig.SCALER_PATH)
except Exception as e:
    logger.error(f"Error initializing app: {str(e)}")
    raise

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()
        
        # Preprocess input
        scaled_input = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        # Return prediction
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

def calculate_monthly_payment(loan_amount, interest_rate, term_months):
    """Calculate monthly loan payment"""
    monthly_rate = interest_rate / (12 * 100)
    return loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error="Internal server error"), 500

def initialize_app():
    """Initialize application and required directories"""
    # Create required directories
    os.makedirs(os.path.dirname(ModelConfig.DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ModelConfig.MODEL_PATH), exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Copy dataset if not exists
    if not os.path.exists(ModelConfig.DATA_PATH):
        source_dataset = os.path.join(os.path.dirname(__file__), 'Loandataset.csv')
        if os.path.exists(source_dataset):
            import shutil
            shutil.copy(source_dataset, ModelConfig.DATA_PATH)
        else:
            raise FileNotFoundError("Dataset not found!")

# Initialize app
initialize_app()

if __name__ == '__main__':
    # Check if model and scaler exist
    if not os.path.exists(ModelConfig.MODEL_PATH) or not os.path.exists(ModelConfig.SCALER_PATH):
        logger.error("Model or scaler not found. Please run train.py first.")
        print("Error: Model or scaler not found. Please run train.py first.")
        exit(1)
        
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=FlaskConfig.DEBUG)