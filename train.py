import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_preprocessing import load_and_preprocess_data
from models import LoanPredictor
import joblib
from config import ModelConfig

def plot_training_history(history, save_path='static/images/training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='static/images/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred.round())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path='models/saved_models/metrics.json'):
    """Save evaluation metrics"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Create necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data(
        ModelConfig.DATA_PATH
    )
    
    # Initialize and train model
    print("Training model...")
    model = LoanPredictor(input_dim=X_train.shape[1])
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=ModelConfig.EPOCHS,
        batch_size=ModelConfig.BATCH_SIZE
    )
    
    # Save model and scaler
    print("Saving model and scaler...")
    os.makedirs('models/saved_models', exist_ok=True)
    model.save_model(ModelConfig.MODEL_PATH)
    joblib.dump(scaler, ModelConfig.SCALER_PATH)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = classification_report(y_test, y_pred.round(), output_dict=True)
    
    # Plot and save visualizations
    print("Generating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Save metrics
    save_metrics(metrics)
    
    print("Training complete! Model and artifacts saved.")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()