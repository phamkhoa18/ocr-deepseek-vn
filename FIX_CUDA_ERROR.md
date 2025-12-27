# 🔧 Khắc phục lỗi CUDA masked_scatter_size_check

## Vấn đề:

Lỗi `masked_scatter_size_check: Assertion 'totalElements <= srcSize' failed` xảy ra khi:
- Kích thước ảnh quá lớn (1280x1280)
- Model code có bug với tensor size lớn
- CUDA kernel không xử lý được tensor quá lớn

## Giải pháp đã áp dụng:

### 1. Giảm image_size mặc định

- Từ 1280 → 640 (cho GPU 24GB)
- Tránh lỗi CUDA kernel

### 2. Tự động resize ảnh lớn

- Nếu ảnh > 2048px, tự động resize
- Giữ tỷ lệ khung hình

### 3. Fallback khi lỗi

- Nếu lỗi CUDA, tự động thử lại với image_size=512
- Giảm base_size xuống 768

### 4. Tắt test_compress

- `test_compress=False` để tránh lỗi thêm

---

## Cách sử dụng:

1. **Upload ảnh bình thường** - Code sẽ tự xử lý
2. **Nếu ảnh quá lớn** - Tự động resize
3. **Nếu gặp lỗi CUDA** - Tự động thử lại với kích thước nhỏ hơn

---

## Cấu hình tối ưu:

Trong `config.py`, image_size đã được giảm:
- GPU 24GB: 640 (thay vì 1280)
- GPU 8-10GB: 640
- GPU 6GB: 640
- GPU <6GB: 512

---

## Nếu vẫn lỗi:

1. **Giảm image_size thủ công:**
   ```python
   # Trong config.py
   IMAGE_SIZE = 512
   BASE_SIZE = 768
   ```

2. **Xử lý ảnh nhỏ hơn:**
   - Resize ảnh trước khi upload
   - Giới hạn kích thước tối đa 1024x1024

3. **Kiểm tra CUDA:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## Lưu ý:

- Image_size=640 vẫn cho chất lượng tốt
- Không cần 1280 trừ khi cần độ chi tiết cực cao
- Tốc độ xử lý nhanh hơn với 640

