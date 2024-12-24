import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelConfig:
    # Data settings
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Loandataset.csv')
    
    # Model parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Model paths
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'model.h5')
    SCALER_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'scaler.pkl')
    
    # These will be updated based on actual columns in the dataset
    FEATURE_NAMES = [
        'loan_amount',
        'interest_rate',
        'term',
        'property_value',
        'income'
    ]

class FlaskConfig:
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = 'uploads'
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

class TestConfig:
    TEST_DATA_PATH = 'tests/test_data.csv'
    TEST_MODEL_PATH = 'tests/test_model.h5'
    TEST_SCALER_PATH = 'tests/test_scaler.pkl'