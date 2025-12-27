# 🐧 Hướng dẫn cài đặt trên Linux

## Cách 1: Sử dụng script tự động (Khuyến nghị)

### Bước 1: Cấp quyền thực thi
```bash
chmod +x install.sh run.sh
```

### Bước 2: Chạy script cài đặt
```bash
./install.sh
```

Script sẽ tự động:
- ✅ Tạo virtual environment
- ✅ Cài đặt PyTorch (tự động phát hiện CUDA)
- ✅ Cài đặt tất cả dependencies
- ✅ Cài flash-attn (nếu có thể)

### Bước 3: Chạy ứng dụng
```bash
./run.sh
```

Hoặc thủ công:
```bash
source venv/bin/activate
python app.py
```

---

## Cách 2: Cài đặt thủ công

### Bước 1: Tạo virtual environment
```bash
python3 -m venv venv
```

Nếu lỗi, cài python3-venv:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-venv

# CentOS/RHEL
sudo yum install python3-venv
```

### Bước 2: Kích hoạt virtual environment
```bash
source venv/bin/activate
```

### Bước 3: Cài đặt PyTorch

**Nếu có GPU NVIDIA:**
```bash
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**Nếu không có GPU (chỉ CPU):**
```bash
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 5: Cài flash-attn (Tùy chọn)
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

Nếu lỗi, có thể bỏ qua (model vẫn chạy được).

### Bước 6: Chạy ứng dụng
```bash
python app.py
```

---

## Cách 3: Sử dụng Conda (Nếu đã cài)

### Cài đặt Conda (nếu chưa có)

**Miniconda:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**Hoặc Anaconda:**
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
source ~/.bashrc
```

### Sau khi cài Conda:
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python app.py
```

---

## 🔧 Troubleshooting

### Lỗi: "python3-venv not found"
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-venv python3-pip

# CentOS/RHEL
sudo yum install python3-venv python3-pip
```

### Lỗi: "Permission denied"
```bash
chmod +x install.sh run.sh
```

### Lỗi: "CUDA not found" nhưng có GPU
- Kiểm tra: `nvidia-smi`
- Cài CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Hoặc dùng PyTorch CPU version

### Lỗi: "Out of Memory"
- Giảm `IMAGE_SIZE` trong `config.py`
- Hoặc đổi `DEVICE='cpu'` trong `config.py`

### Kiểm tra cấu hình hệ thống
```bash
python check_system.py
```

---

## 📋 Yêu cầu hệ thống

- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, Debian 10+)
- **Python**: 3.8+ (khuyến nghị 3.12+)
- **RAM**: 8GB+ (khuyến nghị 16GB+)
- **Ổ cứng**: 50GB+ trống
- **GPU**: Tùy chọn (NVIDIA với CUDA 11.8+)

---

## 🚀 Quick Start

```bash
# 1. Cấp quyền
chmod +x install.sh run.sh

# 2. Cài đặt
./install.sh

# 3. Chạy
./run.sh
```

Sau đó mở trình duyệt: **http://localhost:5000**

---

## 💡 Tips

1. **Chạy ở background:**
   ```bash
   nohup python app.py > app.log 2>&1 &
   ```

2. **Chạy với screen:**
   ```bash
   screen -S ocr
   source venv/bin/activate
   python app.py
   # Nhấn Ctrl+A, D để detach
   ```

3. **Chạy với systemd service:**
   - Tạo file `/etc/systemd/system/deepseek-ocr.service`
   - Xem hướng dẫn trong README.md

---

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
1. Python version: `python3 --version`
2. CUDA (nếu có GPU): `nvidia-smi`
3. Disk space: `df -h`
4. RAM: `free -h`

