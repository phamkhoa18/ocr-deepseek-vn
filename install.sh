#!/bin/bash

echo "=========================================="
echo "  DeepSeek-OCR Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment 'venv' already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf venv
    else
        echo "✅ Using existing virtual environment."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        echo ""
        echo "✅ Installation complete!"
        echo "Run: source venv/bin/activate && python app.py"
        exit 0
    fi
fi

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    echo "Try: sudo apt-get install python3-venv (Ubuntu/Debian)"
    echo "Or: sudo yum install python3-venv (CentOS/RHEL)"
    exit 1
fi

# Activate virtual environment
echo "✅ Virtual environment created."
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo ""
echo "🔥 Installing PyTorch..."
echo "Checking for CUDA..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  No NVIDIA GPU detected. Installing PyTorch CPU version..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo ""
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Try to install flash-attn (optional)
echo ""
echo "⚡ Attempting to install flash-attn (optional)..."
pip install flash-attn==2.7.3 --no-build-isolation 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ flash-attn installed successfully"
else
    echo "⚠️  flash-attn installation failed (this is OK, model will still work)"
fi

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment and run:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Or use the run script:"
echo "  ./run.sh"
echo ""

