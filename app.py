import os
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from config import *
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model and tokenizer
model = None
tokenizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """Initialize the DeepSeek-OCR model"""
    global model, tokenizer
    try:
        # Hiển thị thông tin cấu hình
        print("=" * 60)
        print("Cấu hình hệ thống:")
        print(f"  - Device: {DEVICE}")
        print(f"  - Dtype: {DTYPE}")
        print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"  - Base size: {BASE_SIZE}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  - GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        else:
            print("  - GPU: Không có (sử dụng CPU)")
        print("=" * 60)
        
        print("\nĐang tải tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        print("Đang tải model (có thể mất vài phút lần đầu)...")
        print("Lưu ý: Model sẽ được tải từ Hugging Face (~20-30GB)")
        
        # Thử dùng flash attention nếu có GPU, nếu không thì dùng default
        # Không dùng flash_attention_2 vì có thể gây lỗi với một số version transformers
        try:
            # Thử load model với trust_remote_code (bắt buộc cho DeepSeek-OCR)
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16 if DTYPE == 'bfloat16' else (torch.float16 if DTYPE == 'float16' else torch.float32)
            )
        except Exception as e:
            print(f"Warning: Lỗi khi tải model với dtype: {e}")
            print("Đang thử lại với dtype mặc định...")
            try:
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            except Exception as e2:
                raise Exception(f"Không thể tải model: {str(e2)}")
        
        # Move to device and set dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        dtype = dtype_map.get(DTYPE, torch.bfloat16)
        
        model = model.eval()
        if torch.cuda.is_available() and DEVICE == 'cuda':
            print(f"Đang chuyển model lên GPU với dtype={DTYPE}...")
            model = model.cuda().to(dtype)
        else:
            print(f"Đang chuyển model lên CPU với dtype={DTYPE}...")
            model = model.to(dtype)
        
        print("\n✅ Model đã được tải thành công!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Lỗi khi tải model: {str(e)}")
        print("\nGợi ý khắc phục:")
        print("  1. Kiểm tra kết nối internet")
        print("  2. Đảm bảo có đủ dung lượng ổ cứng (50GB+)")
        print("  3. Thử đổi DEVICE='cpu' trong config.py nếu GPU có vấn đề")
        print("  4. Giảm IMAGE_SIZE trong config.py nếu thiếu RAM/VRAM")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/api/ocr', methods=['POST'])
def ocr():
    """Process OCR request"""
    try:
        if model is None or tokenizer is None:
            return jsonify({
                'success': False,
                'error': 'Model chưa được tải. Vui lòng đợi...'
            }), 503
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy file ảnh'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Chưa chọn file'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Định dạng file không được hỗ trợ. Chỉ chấp nhận: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get prompt from form
        prompt_text = request.form.get('prompt', '<image>\nFree OCR.')
        if not prompt_text.strip():
            prompt_text = '<image>\nFree OCR.'
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        # Process OCR
        output_path = os.path.join(OUTPUT_FOLDER, f"result_{filename}")
        
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt_text,
                image_file=filepath,
                output_path=output_path,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
                save_results=True,
                test_compress=True
            )
        except Exception as e:
            raise Exception(f"Lỗi khi xử lý OCR: {str(e)}")
        
        # Read result - try multiple methods
        result_text = ""
        
        # Method 1: Try to read from output file first (most reliable)
        txt_file = f"{output_path}.txt"
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    result_text = f.read().strip()
            except Exception as e:
                print(f"Warning: Không thể đọc file {txt_file}: {e}")
        
        # Method 2: If no file, try to get from result object
        if not result_text:
            if isinstance(result, dict):
                result_text = result.get('text', result.get('result', str(result)))
            elif isinstance(result, str):
                result_text = result
            elif hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        
        # Clean up result text
        if result_text:
            result_text = result_text.strip()
        
        return jsonify({
            'success': True,
            'text': result_text,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500

@app.route('/api/ocr-base64', methods=['POST'])
def ocr_base64():
    """Process OCR from base64 image"""
    try:
        if model is None or tokenizer is None:
            return jsonify({
                'success': False,
                'error': 'Model chưa được tải. Vui lòng đợi...'
            }), 503
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy dữ liệu ảnh'
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get prompt
        prompt_text = data.get('prompt', '<image>\nFree OCR.')
        if not prompt_text.strip():
            prompt_text = '<image>\nFree OCR.'
        
        # Save temporary file
        import uuid
        temp_filename = f"temp_{uuid.uuid4().hex}.png"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_filepath)
        
        # Process OCR
        output_path = os.path.join(OUTPUT_FOLDER, f"result_{temp_filename}")
        
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt_text,
                image_file=temp_filepath,
                output_path=output_path,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
                save_results=True,
                test_compress=True
            )
        except Exception as e:
            raise Exception(f"Lỗi khi xử lý OCR: {str(e)}")
        
        # Read result - try multiple methods
        result_text = ""
        
        # Method 1: Try to read from output file first (most reliable)
        txt_file = f"{output_path}.txt"
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    result_text = f.read().strip()
            except Exception as e:
                print(f"Warning: Không thể đọc file {txt_file}: {e}")
        
        # Method 2: If no file, try to get from result object
        if not result_text:
            if isinstance(result, dict):
                result_text = result.get('text', result.get('result', str(result)))
            elif isinstance(result, str):
                result_text = result
            elif hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        
        # Clean up result text
        if result_text:
            result_text = result_text.strip()
        
        # Clean up temp file
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'text': result_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Đang khởi tạo DeepSeek-OCR Web Application...")
    print("=" * 50)
    
    # Initialize model
    if init_model():
        print(f"\nServer đang chạy tại: http://{HOST}:{PORT}")
        print("Nhấn Ctrl+C để dừng server\n")
        app.run(host=HOST, port=PORT, debug=DEBUG)
    else:
        print("Không thể khởi tạo model. Vui lòng kiểm tra lại cấu hình.")

