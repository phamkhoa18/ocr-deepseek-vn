# 🚀 Hướng dẫn nhanh

## Cài đặt nhanh

### 1. Tạo môi trường và cài đặt

```bash
# Tạo môi trường
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Cài PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Cài các package khác
pip install -r requirements.txt

# Cài flash-attn (tùy chọn, có thể bỏ qua nếu lỗi)
pip install flash-attn==2.7.3 --no-build-isolation
```

### 2. Chạy ứng dụng

```bash
python app.py
```

### 3. Mở trình duyệt

Truy cập: **http://localhost:5000**

## Sử dụng

1. **Upload ảnh**: Kéo thả hoặc click để chọn file
2. **Nhập prompt** (tùy chọn): Để trống hoặc chọn prompt mẫu
3. **Click "Xử lý OCR"**: Đợi kết quả
4. **Sao chép/Tải xuống**: Sử dụng các nút để lưu kết quả

## Lưu ý

- Lần đầu chạy sẽ mất thời gian để tải model (có thể vài GB)
- Cần có GPU để xử lý nhanh (không bắt buộc)
- Hỗ trợ các định dạng: PNG, JPG, JPEG, PDF, GIF, BMP, WEBP

## Troubleshooting

**Lỗi cài flash-attn?** → Bỏ qua, không ảnh hưởng nhiều

**Out of Memory?** → Đổi `DEVICE = 'cpu'` trong `config.py`

**Model không tải?** → Kiểm tra internet và dung lượng ổ cứng

