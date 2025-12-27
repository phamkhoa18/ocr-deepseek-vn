# 🔧 Khắc phục lỗi DynamicCache.seen_tokens

## Vấn đề:

Lỗi `'DynamicCache' object has no attribute 'seen_tokens'` xảy ra vì:
- Transformers >= 4.41 đã loại bỏ `seen_tokens`
- Thay bằng `cache_position`
- Model code từ Hugging Face vẫn dùng `seen_tokens`

## Giải pháp:

Code đã được cập nhật để tự động patch lỗi này. Nếu vẫn lỗi, thử:

### Cách 1: Hạ cấp transformers (Nhanh nhất)

```bash
source venv/bin/activate
pip install transformers==4.40.0
python app.py
```

### Cách 2: Cập nhật model code (Nếu có quyền)

Nếu bạn có quyền chỉnh sửa model code từ Hugging Face cache:

1. Tìm file: `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/*/modeling_deepseekv2.py`

2. Tìm và thay thế:
   ```python
   # Tìm: past_key_values.seen_tokens
   # Thay bằng: len(past_key_values.cache_position) if hasattr(past_key_values, 'cache_position') else past_key_values.key_cache[0].shape[2]
   ```

### Cách 3: Dùng transformers version cũ hơn

```bash
source venv/bin/activate
pip install "transformers>=4.40.0,<4.41.0"
python app.py
```

---

## Code đã được cập nhật:

- ✅ Tự động patch DynamicCache khi import
- ✅ Thêm property `seen_tokens` tương thích
- ✅ Thêm method `get_max_length` nếu cần

---

## Kiểm tra version transformers:

```bash
python -c "import transformers; print(transformers.__version__)"
```

Version khuyến nghị để tránh lỗi: **4.40.x** hoặc **< 4.41**

---

## Nếu vẫn lỗi:

1. **Xóa cache và tải lại:**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
   ```

2. **Cài lại transformers:**
   ```bash
   pip uninstall transformers -y
   pip install transformers==4.40.0
   ```

3. **Chạy lại:**
   ```bash
   python app.py
   ```

