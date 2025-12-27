# ⚡ Cài đặt nhanh trên Linux (Không cần Conda)

## Cách nhanh nhất:

```bash
# 1. Cấp quyền cho script
chmod +x install.sh run.sh

# 2. Chạy cài đặt tự động
./install.sh

# 3. Chạy ứng dụng
./run.sh
```

---

## Hoặc cài đặt thủ công:

```bash
# 1. Tạo virtual environment
python3 -m venv venv

# 2. Kích hoạt
source venv/bin/activate

# 3. Cài PyTorch (có GPU)
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Hoặc PyTorch CPU (không có GPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Cài dependencies
pip install -r requirements.txt

# 5. Chạy
python app.py
```

---

## Nếu lỗi "python3-venv not found":

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-venv python3-pip

# CentOS/RHEL
sudo yum install python3-venv python3-pip
```

---

## Kiểm tra GPU (nếu có):

```bash
nvidia-smi
```

Nếu có GPU → dùng PyTorch với CUDA
Nếu không có → dùng PyTorch CPU

