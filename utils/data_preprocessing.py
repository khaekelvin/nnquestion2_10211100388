import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the loan dataset
    """
    # Read data
    print("Reading data from:", filepath)
    df = pd.read_csv(filepath)
    print("Columns in dataset:", df.columns.tolist())
    
    # Map expected column names to actual column names in dataset
    column_mappings = {
        'loan_amount': 'loan_amount',
        'rate_of_interest': 'rate_of_interest',
        'interest_rate_spread': 'Interest_rate_spread',
        'upfront_charges': 'Upfront_charges',
        'term_in_months': 'term_in_months',
        'property_value': 'property_value',
        'income': 'income',
        'credit_score': 'Credit_Score',
        'status': 'Status'  # Changed from 'status' to 'Status'
    }
    
    # Select features for model
    features = [
        'loan_amount',
        'rate_of_interest',
        'Interest_rate_spread',
        'Upfront_charges',
        'term_in_months',
        'property_value',
        'income',
        'Credit_Score'
    ]
    
    print("Using features:", features)
    
    # Select features and target
    X = df[features]
    y = df['Status']  # Changed from status to Status
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, features

def preprocess_input(input_data):
    """
    Preprocess single input for prediction
    """
    # Map form field names to model feature names
    feature_mapping = {
        'loan_amount': 'loan_amount',
        'property_value': 'property_value',
        'income': 'income',
        'credit_score': 'Credit_Score',
        'interest_rate_spread': 'Interest_rate_spread',
        'upfront_charges': 'Upfront_charges',
        'loan_term': 'term_in_months',
        'debt_to_income': 'rate_of_interest'  # Using this field for rate_of_interest
    }
    
    # Create DataFrame with correct column names
    transformed_data = {}
    for form_name, feature_name in feature_mapping.items():
        transformed_data[feature_name] = float(input_data[form_name])
    
    # Create DataFrame with the correct order of features
    features = [
        'loan_amount',
        'rate_of_interest',
        'Interest_rate_spread',
        'Upfront_charges',
        'term_in_months',
        'property_value',
        'income',
        'Credit_Score'
    ]
    
    input_df = pd.DataFrame([transformed_data])[features]
    
    # Load scaler
    scaler = joblib.load('models/saved_models/scaler.pkl')
    
    # Scale input
    scaled_input = scaler.transform(input_df)
    
    return scaled_input