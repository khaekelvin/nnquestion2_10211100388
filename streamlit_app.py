import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Configure page
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

def initialize_app():
    """Initialize application and load model"""
    try:
        # Create directories if they don't exist
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        
        # Check if model exists
        model_path = 'models/saved_models/model.h5'
        scaler_path = 'models/saved_models/scaler.pkl'
        
        if not os.path.exists(model_path):
            st.error("Model file not found. Training new model...")
            from train import main as train_model
            train_model()
        
        # Load model and scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        return None, None

def main():
    st.title("🏦 Loan Default Prediction System")
    
    # Initialize app and load model
    model, scaler = initialize_app()
    
    if model is None:
        st.stop()
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0)
            property_value = st.number_input("Property Value ($)", min_value=0.0)
            income = st.number_input("Annual Income ($)", min_value=0.0)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
            
        with col2:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0)
            term_months = st.number_input("Loan Term (months)", min_value=12, max_value=360)
            upfront_charges = st.number_input("Upfront Charges ($)", min_value=0.0)
            rate_spread = st.number_input("Rate Spread (%)", min_value=0.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                input_data = {
                    'loan_amount': loan_amount,
                    'property_value': property_value,
                    'income': income,
                    'credit_score': credit_score,
                    'rate_of_interest': interest_rate,
                    'term_in_months': term_months,
                    'upfront_charges': upfront_charges,
                    'interest_rate_spread': rate_spread
                }
                
                prediction = model.predict(np.array([list(input_data.values())]))
                
                risk_level = "High Risk" if prediction[0][0] > 0.5 else "Low Risk"
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    st.metric(
                        label="Default Risk",
                        value=f"{prediction[0][0]:.2%}",
                        delta="High Risk" if prediction[0][0] > 0.5 else "Low Risk"
                    )

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()