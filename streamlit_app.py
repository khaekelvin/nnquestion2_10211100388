import streamlit as st
import pandas as pd
import numpy as np
from models.loan_predictor import LoanPredictor
from utils.data_preprocessing import preprocess_input
from utils.model_interpretation import generate_feature_importance_plot
import joblib
from config import ModelConfig
import os

def initialize_app():
    if not os.path.exists(ModelConfig.MODEL_PATH):
        st.error("Model not found. Please train the model first.")
        return False
    return True

def main():
    st.title("Loan Default Prediction System")
    
    if not initialize_app():
        return

    # Load model and scaler
    model = LoanPredictor.load_model(ModelConfig.MODEL_PATH)
    scaler = joblib.load(ModelConfig.SCALER_PATH)

    # Input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount", min_value=0.0)
            property_value = st.number_input("Property Value", min_value=0.0)
            income = st.number_input("Income", min_value=0.0)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
            
        with col2:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0)
            term_months = st.number_input("Term (months)", min_value=12, max_value=360)
            upfront_charges = st.number_input("Upfront Charges", min_value=0.0)
            rate_spread = st.number_input("Interest Rate Spread", min_value=0.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
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

            # Make prediction
            try:
                scaled_input = preprocess_input(input_data)
                prediction = model.predict(scaled_input)[0][0]

                # Display results
                st.header("Prediction Results")
                risk_level = "High Risk" if prediction > 0.5 else "Low Risk"
                color = "red" if prediction > 0.5 else "green"
                
                st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
                st.progress(float(prediction))
                st.write(f"Default Probability: {prediction:.2%}")

                # Feature importance
                st.header("Feature Importance")
                fig = generate_feature_importance_plot(scaled_input, model)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()