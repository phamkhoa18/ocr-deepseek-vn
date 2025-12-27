import os
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Auto-detect device and optimize settings
def get_optimal_config():
    """Tự động phát hiện và tối ưu cấu hình theo hệ thống"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        # PyTorch chưa được cài đặt, dùng CPU mặc định
        has_cuda = False
    
    if has_cuda:
        # Kiểm tra VRAM
        try:
            import torch
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory >= 10:
                # High-end GPU (10GB+)
                return {
                    'device': 'cuda',
                    'dtype': 'bfloat16',
                    'base_size': 1024,
                    'image_size': 1280,
                    'crop_mode': True
                }
            elif gpu_memory >= 8:
                # Mid-range GPU (8-10GB)
                return {
                    'device': 'cuda',
                    'dtype': 'bfloat16',
                    'base_size': 1024,
                    'image_size': 640,
                    'crop_mode': True
                }
            elif gpu_memory >= 6:
                # Entry-level GPU (6-8GB)
                return {
                    'device': 'cuda',
                    'dtype': 'bfloat16',
                    'base_size': 1024,
                    'image_size': 640,
                    'crop_mode': True
                }
            else:
                # Low VRAM GPU (<6GB) - giảm kích thước
                return {
                    'device': 'cuda',
                    'dtype': 'float16',
                    'base_size': 768,
                    'image_size': 512,
                    'crop_mode': True
                }
        except:
            # Fallback nếu không kiểm tra được
            return {
                'device': 'cuda',
                'dtype': 'bfloat16',
                'base_size': 1024,
                'image_size': 640,
                'crop_mode': True
            }
    else:
        # CPU only - tối ưu cho CPU
        return {
            'device': 'cpu',
            'dtype': 'float32',  # CPU thường không hỗ trợ bfloat16 tốt
            'base_size': 768,
            'image_size': 512,
            'crop_mode': True
        }

# Lấy cấu hình tối ưu
optimal_config = get_optimal_config()

# Có thể override bằng environment variables hoặc giữ nguyên auto-detect
DEVICE = os.getenv('DEVICE', optimal_config['device'])
DTYPE = os.getenv('DTYPE', optimal_config['dtype'])
BASE_SIZE = int(os.getenv('BASE_SIZE', optimal_config['base_size']))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', optimal_config['image_size']))
CROP_MODE = os.getenv('CROP_MODE', str(optimal_config['crop_mode'])).lower() == 'true'

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

