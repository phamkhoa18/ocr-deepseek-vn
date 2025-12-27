# 🔧 Khắc phục lỗi "data did not match any variant of untagged enum ModelWrapper"

## Vấn đề:

Lỗi này xảy ra khi:
- Transformers version quá cũ (4.40.0) không tương thích với model mới
- Cache model bị corrupt
- Safetensors index file bị lỗi

## Giải pháp:

### Bước 1: Xóa cache và cài transformers version phù hợp

```bash
source venv/bin/activate

# Xóa cache model cũ
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR

# Cài transformers version tương thích (4.46-4.47)
pip install transformers==4.46.0 accelerate

# Hoặc cài từ requirements.txt (đã được cập nhật)
pip install -r requirements.txt
```

### Bước 2: Chạy lại

```bash
python app.py
```

---

## Nếu vẫn lỗi:

### Cách 1: Xóa toàn bộ cache Hugging Face

```bash
rm -rf ~/.cache/huggingface
pip install transformers==4.46.0 accelerate
python app.py
```

### Cách 2: Dùng transformers 4.47.0

```bash
pip install transformers==4.47.0 accelerate
python app.py
```

### Cách 3: Kiểm tra và sửa safetensors

```bash
# Kiểm tra file model
ls -lh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/*/model.safetensors.index.json

# Nếu file bị lỗi, xóa và tải lại
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
python app.py
```

---

## Version transformers khuyến nghị:

- ✅ **4.46.0** - Tương thích tốt nhất
- ✅ **4.47.0** - Cũng tốt
- ❌ **4.40.0** - Quá cũ, không tương thích
- ❌ **4.57.x** - Quá mới, có thể gây lỗi seen_tokens

---

## Code đã được cập nhật:

- ✅ Patch DynamicCache tự động
- ✅ Requirements.txt đã cập nhật transformers 4.46-4.47
- ✅ Xử lý lỗi tốt hơn

---

## Sau khi fix:

1. Model sẽ tải lại từ đầu (mất vài phút)
2. Cache mới sẽ được tạo
3. Lỗi sẽ được fix

