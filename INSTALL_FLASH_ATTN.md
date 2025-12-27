# ⚡ Cài đặt flash-attn để khắc phục lỗi

## Vấn đề:

Model DeepSeek-OCR yêu cầu `LlamaFlashAttention2` từ transformers, nhưng class này chỉ có khi:
1. Cài `flash-attn` package, HOẶC
2. Dùng transformers version rất mới (>=4.46.0 với flash-attn support)

## Giải pháp nhanh:

### Trên server Linux của bạn:

```bash
# Đảm bảo đang trong virtual environment
source venv/bin/activate

# Cài flash-attn (Khuyến nghị - sẽ fix lỗi ngay)
pip install flash-attn==2.7.3 --no-build-isolation
```

**Lưu ý:** Cài flash-attn có thể mất 5-10 phút vì cần compile.

### Nếu cài flash-attn bị lỗi:

**Cách 1: Cập nhật transformers lên version mới nhất**
```bash
pip install --upgrade transformers>=4.46.0 accelerate
```

**Cách 2: Cài từ source (nếu cần)**
```bash
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

**Cách 3: Bỏ qua flash-attn, dùng transformers mới nhất**
```bash
pip install --upgrade transformers>=4.51.0 accelerate
```

---

## Kiểm tra sau khi cài:

```bash
python -c "import flash_attn; print('✅ flash-attn OK')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

---

## Sau đó chạy lại:

```bash
python app.py
```

---

## Tại sao cần flash-attn?

- Model code từ Hugging Face import `LlamaFlashAttention2`
- Class này chỉ có khi cài flash-attn hoặc dùng transformers rất mới
- Flash-attn giúp tăng tốc độ xử lý đáng kể trên GPU

---

## Nếu vẫn lỗi:

1. **Kiểm tra CUDA:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Cài lại từ đầu:**
   ```bash
   pip uninstall flash-attn -y
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

3. **Hoặc dùng transformers nightly:**
   ```bash
   pip install --upgrade transformers[torch] --pre
   ```

