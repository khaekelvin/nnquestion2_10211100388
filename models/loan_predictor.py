import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class LoanPredictor:
    def __init__(self, input_dim):
        self.model = self._build_model(input_dim)
        
    def _build_model(self, input_dim):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    @staticmethod
    def load_model(filepath):
        model = models.load_model(filepath)
        predictor = LoanPredictor(model.input_shape[1])
        predictor.model = model
        return predictor