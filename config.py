import os

class ModelConfig:
    # Detect environment
    IS_PRODUCTION = os.environ.get('RENDER', False)
    
    # Set base directory based on environment
    if IS_PRODUCTION:
        BASE_DIR = "/opt/render/project/src"
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Data paths
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
    
    DATA_PATH = os.path.join(DATA_DIR, "Loandataset.csv")
    MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    
    # Model parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
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