# ⏱️ Model đang tải quá lâu?

## Thời gian tải model bình thường:

- **Tải từ Hugging Face**: 2-5 phút (tùy internet)
- **Tải lên GPU**: 5-10 phút (tùy GPU và model size)
- **Tổng cộng**: 7-15 phút lần đầu

## Model đang ở bước nào?

Nếu thấy:
- ✅ "Đang chuyển model lên GPU với dtype=bfloat16..." 
- ⏳ Đang chờ...

→ **Đây là bình thường!** Model đang tải ~6.7GB weights lên GPU.

## Kiểm tra xem có đang chạy không:

### Cách 1: Kiểm tra GPU usage
Mở terminal khác và chạy:
```bash
watch -n 1 nvidia-smi
```

Nếu thấy:
- GPU Memory tăng dần → Model đang tải
- GPU Memory không đổi → Có thể bị treo

### Cách 2: Kiểm tra process
```bash
ps aux | grep python
```

Nếu thấy process python đang chạy → Model vẫn đang tải

## Nếu đã chờ > 15 phút:

### Cách 1: Kiểm tra VRAM
```bash
nvidia-smi
```

Nếu VRAM đầy (>90%) → Model quá lớn, cần:
- Giảm IMAGE_SIZE trong config.py
- Hoặc dùng CPU (chậm hơn)

### Cách 2: Restart và dùng CPU tạm thời
```bash
# Dừng server (Ctrl+C)
# Sửa config.py: DEVICE = 'cpu'
python app.py
```

### Cách 3: Giảm kích thước model
```bash
# Trong config.py
IMAGE_SIZE = 512  # Thay vì 640
BASE_SIZE = 768  # Thay vì 1024
```

## Lưu ý:

- **Lần đầu tải**: Luôn mất nhiều thời gian
- **Lần sau**: Model đã cache, chỉ mất 1-2 phút
- **GPU 24GB**: Đủ để tải model, nhưng vẫn mất 5-10 phút

## Nếu bị treo thật sự:

1. **Kiểm tra log**: Xem có lỗi gì không
2. **Kiểm tra RAM**: `free -h` (cần ít nhất 16GB)
3. **Kiểm tra disk**: `df -h` (cần ít nhất 50GB trống)
4. **Restart server**: Ctrl+C rồi chạy lại

---

## Code đã được cập nhật:

- ✅ Thêm progress messages
- ✅ Hiển thị VRAM usage
- ✅ Kiểm tra OOM errors
- ✅ Thông báo rõ ràng hơn

