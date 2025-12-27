# 🔧 Khắc phục lỗi thiếu dependencies

## Lỗi: "No module named 'addict' hoặc 'matplotlib'"

### Giải pháp nhanh:

Trên server Linux của bạn, chạy:

```bash
# Đảm bảo đang trong virtual environment
source venv/bin/activate

# Cài các dependencies còn thiếu
pip install addict matplotlib

# Hoặc cài lại tất cả từ requirements.txt (đã được cập nhật)
pip install -r requirements.txt
```

### Sau đó chạy lại:

```bash
python app.py
```

---

## Các dependencies cần thiết cho DeepSeek-OCR:

- ✅ `addict` - Để xử lý cấu hình
- ✅ `matplotlib` - Để visualization (nếu cần)
- ✅ `torch`, `transformers` - Core dependencies
- ✅ Các dependencies khác trong requirements.txt

---

## Nếu vẫn gặp lỗi:

1. **Kiểm tra virtual environment:**
   ```bash
   which python
   # Phải hiển thị: /root/apps/ocr-deepseek-vn/venv/bin/python
   ```

2. **Cập nhật pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Cài lại tất cả:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Kiểm tra GPU:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## Lưu ý:

- GPU của bạn: **GRID P40-24Q (24GB VRAM)** - Rất tốt! ✅
- Cấu hình hiện tại: `IMAGE_SIZE=1280` - Phù hợp với GPU 24GB
- Model sẽ tự động sử dụng GPU khi đã cài đủ dependencies

