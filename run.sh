#!/bin/bash

echo "=========================================="
echo "  DeepSeek-OCR Web Application"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Starting server..."
echo "📱 Open http://localhost:5000 in your browser"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Run the application
python app.py

