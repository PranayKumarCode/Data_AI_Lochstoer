#!/bin/bash

echo "========================================"
echo "  Assistant Startup"
echo "========================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Check if Ollama is running
echo "[1/4] Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ! Ollama not running, starting it..."
    # Start Ollama in background (if installed)
    if command -v ollama &> /dev/null; then
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
        echo "  ✓ Ollama started"
    else
        echo "  ✗ Ollama not found. Please install from https://ollama.com"
        exit 1
    fi
else
    echo "  ✓ Ollama already running"
fi

# Check if model exists
echo ""
echo "[2/4] Checking for qwen2.5:7b model..."
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "  ! Model not found. Downloading qwen2.5:7b..."
    ollama pull qwen2.5:7b
else
    echo "  ✓ Model ready"
fi

# Activate virtual environment
echo ""
echo "[3/4] Activating Python environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "  ✓ Environment activated"
else
    echo "  ✗ Virtual environment not found"
    exit 1
fi

# Run the client
echo ""
echo "[4/4] Starting Assistant..."
echo ""
echo "========================================"
python client_test_v2.py

echo ""
echo "========================================"
echo "  Session Complete"
echo "========================================"
read -p "Press Enter to exit..."
