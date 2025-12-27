# 📋 Hướng dẫn cấu hình hệ thống

## 🎯 Các mức cấu hình được hỗ trợ

### ⭐ Cấu hình Tối thiểu (CPU Only)

**Phần cứng:**
- **CPU**: Intel i5 thế hệ 8+ hoặc AMD Ryzen 5 3000+ (4 cores trở lên)
- **RAM**: 8GB (tối thiểu) - 16GB (khuyến nghị)
- **Ổ cứng**: 50GB trống (để tải model ~20-30GB)
- **GPU**: Không cần

**Cấu hình trong `config.py`:**
```python
DEVICE = 'cpu'
DTYPE = 'float32'  # hoặc 'float16'
IMAGE_SIZE = 512   # Giảm kích thước để tiết kiệm RAM
BASE_SIZE = 768
```

**Hiệu suất:**
- ⏱️ Thời gian xử lý: 30-60 giây/ảnh
- 💾 RAM sử dụng: ~6-8GB
- ✅ **Có thể chạy được** nhưng chậm

---

### 🚀 Cấu hình Khuyến nghị (GPU Entry-level)

**Phần cứng:**
- **GPU**: NVIDIA GTX 1660, RTX 2060, RTX 3050 (6GB VRAM)
- **CPU**: Intel i5/i7 hoặc AMD Ryzen 5/7
- **RAM**: 16GB
- **Ổ cứng**: 50GB trống
- **CUDA**: 11.8+

**Cấu hình trong `config.py`:**
```python
DEVICE = 'cuda'
DTYPE = 'bfloat16'  # hoặc 'float16'
IMAGE_SIZE = 640
BASE_SIZE = 1024
```

**Hiệu suất:**
- ⏱️ Thời gian xử lý: 5-15 giây/ảnh
- 💾 VRAM sử dụng: ~4-6GB
- ✅ **Chạy tốt** cho hầu hết trường hợp

---

### ⚡ Cấu hình Tối ưu (GPU Mid-range)

**Phần cứng:**
- **GPU**: NVIDIA RTX 3060, RTX 3070, RTX 4060 (8-12GB VRAM)
- **CPU**: Intel i7/i9 hoặc AMD Ryzen 7/9
- **RAM**: 16-32GB
- **Ổ cứng**: 100GB trống
- **CUDA**: 11.8+

**Cấu hình trong `config.py`:**
```python
DEVICE = 'cuda'
DTYPE = 'bfloat16'
IMAGE_SIZE = 640
BASE_SIZE = 1024
```

**Hiệu suất:**
- ⏱️ Thời gian xử lý: 3-8 giây/ảnh
- 💾 VRAM sử dụng: ~6-8GB
- ✅ **Chạy rất tốt**, xử lý nhanh

---

### 🔥 Cấu hình Cao cấp (GPU High-end)

**Phần cứng:**
- **GPU**: NVIDIA RTX 3080, RTX 3090, RTX 4080, RTX 4090, A100 (10GB+ VRAM)
- **CPU**: Intel i9 hoặc AMD Ryzen 9
- **RAM**: 32GB+
- **Ổ cứng**: 100GB+ trống
- **CUDA**: 11.8+

**Cấu hình trong `config.py`:**
```python
DEVICE = 'cuda'
DTYPE = 'bfloat16'
IMAGE_SIZE = 1280  # Có thể tăng lên
BASE_SIZE = 1024
```

**Hiệu suất:**
- ⏱️ Thời gian xử lý: 1-3 giây/ảnh
- 💾 VRAM sử dụng: ~8-12GB
- ✅ **Chạy cực nhanh**, xử lý ảnh lớn

---

## 🔧 Cách kiểm tra cấu hình hệ thống

### Kiểm tra GPU (Windows)
```powershell
nvidia-smi
```

### Kiểm tra RAM (Windows)
```powershell
systeminfo | findstr "Total Physical Memory"
```

### Kiểm tra Python và PyTorch
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

---

## ⚙️ Tối ưu hóa theo cấu hình

### Nếu gặp lỗi "Out of Memory":

1. **Giảm IMAGE_SIZE:**
   ```python
   IMAGE_SIZE = 512  # Thay vì 640
   BASE_SIZE = 768   # Thay vì 1024
   ```

2. **Chuyển sang CPU:**
   ```python
   DEVICE = 'cpu'
   DTYPE = 'float32'
   ```

3. **Giảm batch size** (nếu xử lý nhiều ảnh):
   - Xử lý từng ảnh một

### Nếu chạy quá chậm:

1. **Đảm bảo có GPU:**
   ```python
   DEVICE = 'cuda'
   ```

2. **Sử dụng dtype nhẹ hơn:**
   ```python
   DTYPE = 'bfloat16'  # hoặc 'float16'
   ```

3. **Cài flash-attn** (nếu có GPU):
   ```bash
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

---

## 📊 Bảng so sánh cấu hình

| Cấu hình | GPU VRAM | RAM | Thời gian/ảnh | Khuyến nghị |
|----------|----------|-----|---------------|-------------|
| Tối thiểu | 0GB (CPU) | 8GB | 30-60s | ⚠️ Chậm, chỉ dùng khi không có GPU |
| Entry | 6GB | 16GB | 5-15s | ✅ Tốt cho hầu hết người dùng |
| Mid-range | 8-12GB | 16-32GB | 3-8s | ⭐ Khuyến nghị |
| High-end | 10GB+ | 32GB+ | 1-3s | 🔥 Tối ưu nhất |

---

## 🎯 Khuyến nghị cho bạn

**Nếu bạn có:**
- **Laptop/PC thông thường** (không có GPU NVIDIA) → Dùng cấu hình CPU, chấp nhận chậm
- **GPU NVIDIA 6GB** (GTX 1660, RTX 2060) → Cấu hình Entry-level, chạy tốt
- **GPU NVIDIA 8GB+** (RTX 3060, RTX 3070) → Cấu hình Mid-range, chạy rất tốt
- **GPU NVIDIA 10GB+** (RTX 3080, RTX 4090) → Cấu hình High-end, chạy cực nhanh

---

## 💡 Tips tối ưu

1. **Luôn kiểm tra VRAM trước:**
   ```bash
   nvidia-smi
   ```

2. **Bắt đầu với cấu hình thấp**, sau đó tăng dần nếu không lỗi

3. **Xử lý ảnh nhỏ hơn** nếu gặp lỗi memory

4. **Đóng các ứng dụng khác** khi chạy để giải phóng RAM/VRAM

5. **Sử dụng CPU nếu GPU không đủ VRAM** - chậm nhưng vẫn chạy được

---

## ❓ Câu hỏi thường gặp

**Q: Tôi chỉ có 4GB RAM, chạy được không?**
A: Rất khó. Khuyến nghị tối thiểu 8GB RAM.

**Q: GPU AMD có chạy được không?**
A: Hiện tại chỉ hỗ trợ NVIDIA CUDA. GPU AMD sẽ chạy trên CPU (chậm).

**Q: MacBook M1/M2 chạy được không?**
A: Có thể chạy trên CPU, nhưng cần cài PyTorch cho Mac (không dùng CUDA).

**Q: Model nặng bao nhiêu?**
A: Khoảng 20-30GB khi tải về từ Hugging Face.

**Q: Có thể chạy trên Google Colab không?**
A: Có! Colab có GPU miễn phí (T4) đủ để chạy.

