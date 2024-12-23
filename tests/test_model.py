import unittest
import numpy as np
import pandas as pd
from models import LoanPredictor
from utils.data_preprocessing import preprocess_input
import joblib
from config import TestConfig

class TestLoanPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        cls.input_dim = 8
        cls.model = LoanPredictor(cls.input_dim)
        
        # Create sample data
        cls.X_train = np.random.random((100, cls.input_dim))
        cls.y_train = np.random.randint(0, 2, 100)
        cls.X_val = np.random.random((20, cls.input_dim))
        cls.y_val = np.random.randint(0, 2, 20)
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model.input_shape[1], self.input_dim)
    
    def test_model_training(self):
        """Test if model trains without errors"""
        history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=2,
            batch_size=32
        )
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)
    
    def test_model_prediction(self):
        """Test if model makes predictions correctly"""
        sample_input = np.random.random((1, self.input_dim))
        prediction = self.model.predict(sample_input)
        self.assertTrue(0 <= prediction <= 1)
        self.assertEqual(prediction.shape, (1, 1))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Save model
        self.model.save_model(TestConfig.TEST_MODEL_PATH)
        
        # Load model
        loaded_model = LoanPredictor.load_model(TestConfig.TEST_MODEL_PATH)
        
        # Compare predictions
        sample_input = np.random.random((1, self.input_dim))
        original_pred = self.model.predict(sample_input)
        loaded_pred = loaded_model.predict(sample_input)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

if __name__ == '__main__':
    unittest.main()