# DeepSeek-OCR Web Application

Ứng dụng web với giao diện đẹp để sử dụng DeepSeek-OCR - công cụ nhận dạng văn bản từ ảnh sử dụng AI.

## ✨ Tính năng

- 🖼️ **Upload ảnh**: Kéo thả hoặc chọn file ảnh (PNG, JPG, JPEG, PDF, GIF, BMP, WEBP)
- 📝 **Nhập prompt tùy chỉnh**: Hỗ trợ nhiều loại prompt khác nhau
- 🎨 **Giao diện đẹp**: UI hiện đại, responsive
- 📋 **Sao chép kết quả**: Copy kết quả OCR dễ dàng
- 💾 **Tải xuống**: Lưu kết quả dưới dạng file .txt
- ⚡ **Xử lý nhanh**: Sử dụng GPU để tăng tốc độ xử lý

## 🚀 Cài đặt

### Yêu cầu hệ thống

#### ⚠️ Cấu hình Tối thiểu (CPU)
- **Python**: 3.8+
- **RAM**: 8GB (tối thiểu) - 16GB (khuyến nghị)
- **Ổ cứng**: 50GB trống
- **GPU**: Không cần (chạy trên CPU, chậm ~30-60s/ảnh)

#### ✅ Cấu hình Khuyến nghị (GPU)
- **Python**: 3.8+
- **GPU**: NVIDIA với 6GB+ VRAM (GTX 1660, RTX 2060, RTX 3060...)
- **CUDA**: 11.8+
- **RAM**: 16GB
- **Ổ cứng**: 50GB trống
- **Hiệu suất**: 5-15 giây/ảnh

#### 🔥 Cấu hình Tối ưu (GPU High-end)
- **GPU**: NVIDIA với 10GB+ VRAM (RTX 3080, RTX 4090, A100...)
- **RAM**: 32GB+
- **Hiệu suất**: 1-3 giây/ảnh

> 📋 **Xem chi tiết**: Đọc file [CONFIG_GUIDE.md](CONFIG_GUIDE.md) để biết thêm về các mức cấu hình và cách tối ưu.

> 🔍 **Kiểm tra hệ thống**: Chạy `python check_system.py` để kiểm tra cấu hình và nhận khuyến nghị.

### Bước 1: Clone repository

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
```

### Bước 2: Tạo môi trường ảo

**🐧 Linux/Mac (Khuyến nghị - Không cần Conda):**
```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt
source venv/bin/activate
```

**Hoặc dùng script tự động:**
```bash
chmod +x install.sh
./install.sh
```

**🪟 Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**🐍 Conda (Nếu đã cài):**
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### Bước 3: Cài đặt PyTorch

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### Bước 4: Cài đặt các package khác

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

**Lưu ý**: Nếu gặp lỗi khi cài `flash-attn`, bạn có thể bỏ qua bước này. Model vẫn hoạt động nhưng có thể chậm hơn.

### Bước 5: Cấu hình (Tùy chọn)

Chỉnh sửa file `config.py` nếu cần thay đổi:
- `DEVICE`: 'cuda' hoặc 'cpu'
- `HOST`: Địa chỉ host (mặc định: '0.0.0.0')
- `PORT`: Cổng server (mặc định: 5000)

## 🎯 Sử dụng

### Khởi chạy ứng dụng

```bash
python app.py
```

Sau khi khởi động thành công, mở trình duyệt và truy cập:
```
http://localhost:5000
```

### Cách sử dụng

1. **Upload ảnh**: 
   - Kéo thả ảnh vào vùng upload, hoặc
   - Click vào vùng upload để chọn file

2. **Nhập prompt** (Tùy chọn):
   - Để trống sẽ sử dụng "Free OCR"
   - Hoặc chọn một trong các prompt mẫu có sẵn
   - Hoặc nhập prompt tùy chỉnh của bạn

3. **Xử lý OCR**: Click nút "Xử lý OCR" và đợi kết quả

4. **Sao chép/Tải xuống**: Sử dụng các nút để sao chép hoặc tải xuống kết quả

### Các loại prompt hỗ trợ

- **Free OCR**: `<image>\nFree OCR.` - Nhận dạng văn bản tự do
- **Convert to Markdown**: `<image>\n<|grounding|>Convert the document to markdown.` - Chuyển đổi tài liệu sang Markdown
- **OCR Image**: `<image>\n<|grounding|>OCR this image.` - OCR ảnh với layout
- **Parse Figure**: `<image>\nParse the figure.` - Phân tích hình ảnh/biểu đồ
- **Describe Image**: `<image>\nDescribe this image in detail.` - Mô tả chi tiết ảnh

## 📁 Cấu trúc dự án

```
OCR-DEEPSEEK-VN/
├── app.py                 # Backend Flask application
├── config.py             # Cấu hình ứng dụng
├── requirements.txt      # Dependencies
├── README.md            # File hướng dẫn này
├── templates/           # HTML templates
│   └── index.html       # Giao diện chính
├── static/              # Static files
│   ├── css/
│   │   └── style.css    # Stylesheet
│   └── js/
│       └── main.js      # JavaScript logic
├── uploads/             # Thư mục lưu file upload (tự động tạo)
└── outputs/             # Thư mục lưu kết quả (tự động tạo)
```

## 🔧 Troubleshooting

### Lỗi: Model không tải được

- Kiểm tra kết nối internet (model sẽ tự động tải từ Hugging Face)
- Đảm bảo có đủ dung lượng ổ cứng (model khá lớn)
- Kiểm tra CUDA và PyTorch đã được cài đặt đúng

### Lỗi: Out of Memory

- Giảm `IMAGE_SIZE` trong `config.py`
- Sử dụng CPU thay vì GPU (đổi `DEVICE = 'cpu'` trong `config.py`)
- Xử lý ảnh nhỏ hơn

### Lỗi: Flash Attention

- Nếu không cài được `flash-attn`, có thể bỏ qua
- Model vẫn hoạt động nhưng sẽ chậm hơn

## 📝 API Endpoints

### GET `/`
Trang chủ - Giao diện web

### GET `/health`
Kiểm tra trạng thái server và model

### POST `/api/ocr`
Xử lý OCR từ file upload

**Request:**
- `image`: File ảnh (multipart/form-data)
- `prompt`: Prompt text (optional)

**Response:**
```json
{
    "success": true,
    "text": "Kết quả OCR...",
    "filename": "image.jpg"
}
```

### POST `/api/ocr-base64`
Xử lý OCR từ base64 image

**Request:**
```json
{
    "image": "data:image/png;base64,...",
    "prompt": "<image>\nFree OCR."
}
```

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🙏 Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - Model OCR chính
- Flask - Web framework
- Transformers - Hugging Face library

## 📧 Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub.

---

**Lưu ý**: Ứng dụng này sử dụng model DeepSeek-OCR từ Hugging Face. Lần đầu chạy sẽ mất thời gian để tải model (có thể vài GB).

