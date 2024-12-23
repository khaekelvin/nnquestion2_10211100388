import unittest
import numpy as np
import pandas as pd
from utils.data_preprocessing import load_and_preprocess_data, preprocess_input
from sklearn.preprocessing import StandardScaler
from config import TestConfig

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create sample data for testing"""
        cls.sample_data = pd.DataFrame({
            'loan_amount': [100000, 200000],
            'rate_of_interest': [5.5, 6.0],
            'interest_rate_spread': [2.1, 2.5],
            'upfront_charges': [1000, 2000],
            'term_in_months': [360, 240],
            'property_value': [250000, 300000],
            'income': [80000, 90000],
            'credit_score': [720, 680],
            'status': [0, 1]
        })
        cls.sample_data.to_csv(TestConfig.TEST_DATA_PATH, index=False)
    
    def test_load_and_preprocess_data(self):
        """Test data loading and preprocessing"""
        X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data(
            TestConfig.TEST_DATA_PATH
        )
        
        # Check shapes
        self.assertEqual(len(X_train.shape), 2)
        self.assertEqual(X_train.shape[1], 8)
        
        # Check if scaler is fitted
        self.assertIsInstance(scaler, StandardScaler)
        self.assertTrue(hasattr(scaler, 'mean_'))
        
        # Check features list
        self.assertEqual(len(features), 8)
        self.assertIn('loan_amount', features)
    
    def test_preprocess_input(self):
        """Test single input preprocessing"""
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(self.sample_data.drop('status', axis=1))
        
        # Test input
        input_data = {
            'loan_amount': 150000,
            'rate_of_interest': 5.8,
            'interest_rate_spread': 2.3,
            'upfront_charges': 1500,
            'term_in_months': 300,
            'property_value': 275000,
            'income': 85000,
            'credit_score': 700
        }
        
        processed = preprocess_input(input_data, scaler)
        
        # Check output shape and type
        self.assertEqual(processed.shape, (1, 8))
        self.assertTrue(isinstance(processed, np.ndarray))

if __name__ == '__main__':
    unittest.main()