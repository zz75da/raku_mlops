import pytest
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock environment variables
os.environ["SECRET_KEY"] = "test_secret_key"
os.environ["MLFLOW_TRACKING_URI"] = "http://test-mlflow:5000"
os.environ["PREPROCESS_API_URL"] = "http://test-preprocess:5001"

@pytest.fixture
def mock_jwt_token():
    return "mock_jwt_token"

@pytest.fixture
def sample_image_data():
    # Create a simple test image
    from PIL import Image
    import io
    import base64
    
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

@pytest.fixture
def sample_text_data():
    return ["High quality leather handbag", "Men's athletic shoes"]

@pytest.fixture
def sample_image_paths(tmp_path):
    # Create test image files
    from PIL import Image
    image_paths = []
    
    for i in range(3):
        img_path = tmp_path / f"test_image_{i}.jpg"
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        img.save(img_path)
        image_paths.append(str(img_path))
    
    return image_paths

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.tracking.MlflowClient') as mock:
        yield mock

@pytest.fixture
def mock_requests():
    with patch('requests.post') as mock:
        yield mock