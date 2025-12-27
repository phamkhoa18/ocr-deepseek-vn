# 🔧 Khắc phục lỗi transformers

## Lỗi: "cannot import name 'LlamaFlashAttention2'"

### Nguyên nhân:
- Version `transformers` không tương thích với model
- Model cần version transformers cụ thể

### Giải pháp:

**Trên server Linux, chạy:**

```bash
# Đảm bảo đang trong virtual environment
source venv/bin/activate

# Cập nhật transformers lên version tương thích
pip install --upgrade "transformers>=4.46.0,<5.0.0"

# Hoặc cài lại tất cả từ requirements.txt (đã được cập nhật)
pip install -r requirements.txt
```

**Sau đó chạy lại:**
```bash
python app.py
```

---

## Nếu vẫn lỗi:

### Cách 1: Cài flash-attn (Khuyến nghị nếu có GPU)

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### Cách 2: Cập nhật transformers lên version mới nhất

```bash
pip install --upgrade transformers accelerate
```

### Cách 3: Kiểm tra version hiện tại

```bash
python -c "import transformers; print(transformers.__version__)"
```

Version khuyến nghị: **4.46.0 - 4.51.x**

---

## Lưu ý:

- Model DeepSeek-OCR yêu cầu `trust_remote_code=True` (bắt buộc)
- Không cần flash_attention_2 để chạy, chỉ cần để tăng tốc
- Code đã được cập nhật để tự động xử lý trường hợp không có flash attention

