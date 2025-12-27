import os
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
DEVICE = 'cuda'  # or 'cpu'
DTYPE = 'bfloat16'  # or 'float16'

# Image processing
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# Server configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Upload configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'gif', 'bmp', 'webp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

