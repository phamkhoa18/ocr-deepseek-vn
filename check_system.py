"""
Script kiểm tra cấu hình hệ thống và đưa ra khuyến nghị
"""
import sys
import platform

def check_system():
    print("=" * 60)
    print("KIỂM TRA CẤU HÌNH HỆ THỐNG")
    print("=" * 60)
    
    # Python version
    print(f"\n🐍 Python: {sys.version.split()[0]}")
    
    # OS
    print(f"💻 Hệ điều hành: {platform.system()} {platform.release()}")
    
    # Check PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: Có sẵn")
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   - VRAM: {gpu_memory:.1f}GB")
            
            # Đánh giá GPU
            if gpu_memory >= 10:
                print("   ⭐ Đánh giá: GPU Cao cấp - Chạy rất tốt!")
            elif gpu_memory >= 8:
                print("   ⭐ Đánh giá: GPU Tốt - Chạy tốt!")
            elif gpu_memory >= 6:
                print("   ⭐ Đánh giá: GPU Entry-level - Chạy được!")
            else:
                print("   ⚠️ Đánh giá: GPU VRAM thấp - Có thể gặp lỗi OOM")
        else:
            print("❌ CUDA: Không có (sẽ chạy trên CPU - chậm hơn)")
    except ImportError:
        print("❌ PyTorch: Chưa cài đặt")
        print("   Chạy: pip install torch torchvision torchaudio")
    
    # Check RAM (Windows)
    try:
        if platform.system() == 'Windows':
            import psutil
            ram_total = psutil.virtual_memory().total / (1024**3)
            ram_available = psutil.virtual_memory().available / (1024**3)
            print(f"\n💾 RAM:")
            print(f"   - Tổng: {ram_total:.1f}GB")
            print(f"   - Còn trống: {ram_available:.1f}GB")
            
            if ram_total >= 32:
                print("   ⭐ Đánh giá: RAM Dồi dào - Tốt!")
            elif ram_total >= 16:
                print("   ⭐ Đánh giá: RAM Đủ - Tốt!")
            elif ram_total >= 8:
                print("   ⚠️ Đánh giá: RAM Tối thiểu - Có thể chậm")
            else:
                print("   ❌ Đánh giá: RAM Quá thấp - Không khuyến nghị")
    except ImportError:
        print("\n💾 RAM: Không thể kiểm tra (cài psutil để kiểm tra)")
        print("   Chạy: pip install psutil")
    except Exception as e:
        print(f"\n💾 RAM: Lỗi khi kiểm tra: {e}")
    
    # Check disk space
    try:
        import shutil
        disk_total, disk_used, disk_free = shutil.disk_usage('.')
        disk_free_gb = disk_free / (1024**3)
        print(f"\n💿 Ổ cứng:")
        print(f"   - Còn trống: {disk_free_gb:.1f}GB")
        
        if disk_free_gb >= 100:
            print("   ⭐ Đánh giá: Đủ dung lượng!")
        elif disk_free_gb >= 50:
            print("   ⚠️ Đánh giá: Đủ nhưng hơi ít (model ~30GB)")
        else:
            print("   ❌ Đánh giá: Thiếu dung lượng - Cần ít nhất 50GB")
    except Exception as e:
        print(f"\n💿 Ổ cứng: Lỗi khi kiểm tra: {e}")
    
    # Khuyến nghị
    print("\n" + "=" * 60)
    print("KHUYẾN NGHỊ CẤU HÌNH")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 10:
                print("\n✅ Cấu hình đề xuất trong config.py:")
                print("   DEVICE = 'cuda'")
                print("   DTYPE = 'bfloat16'")
                print("   IMAGE_SIZE = 1280")
                print("   BASE_SIZE = 1024")
            elif gpu_memory >= 8:
                print("\n✅ Cấu hình đề xuất trong config.py:")
                print("   DEVICE = 'cuda'")
                print("   DTYPE = 'bfloat16'")
                print("   IMAGE_SIZE = 640")
                print("   BASE_SIZE = 1024")
            elif gpu_memory >= 6:
                print("\n✅ Cấu hình đề xuất trong config.py:")
                print("   DEVICE = 'cuda'")
                print("   DTYPE = 'bfloat16'")
                print("   IMAGE_SIZE = 640")
                print("   BASE_SIZE = 1024")
            else:
                print("\n⚠️ Cấu hình đề xuất trong config.py:")
                print("   DEVICE = 'cuda'")
                print("   DTYPE = 'float16'")
                print("   IMAGE_SIZE = 512")
                print("   BASE_SIZE = 768")
        else:
            print("\n⚠️ Cấu hình đề xuất trong config.py (CPU):")
            print("   DEVICE = 'cpu'")
            print("   DTYPE = 'float32'")
            print("   IMAGE_SIZE = 512")
            print("   BASE_SIZE = 768")
            print("\n💡 Lưu ý: Chạy trên CPU sẽ rất chậm (30-60s/ảnh)")
    except:
        print("\n⚠️ Không thể đưa ra khuyến nghị (chưa cài PyTorch)")
    
    print("\n" + "=" * 60)
    print("Để chạy ứng dụng: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    check_system()

