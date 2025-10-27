#!/bin/bash

set -e

echo "=================================================="
echo "Starting MedRAX Backend Server"
echo "=================================================="
echo ""

# Prefer conda env if available
USE_CONDA=0
if command -v conda &> /dev/null; then
    USE_CONDA=1
fi

echo "Checking backend environment..."

cd backend

PIP_INSTALL=1
if [ $USE_CONDA -eq 1 ]; then
    echo "Using conda environment"
    # Read env name from environment.yml (fallback to medrax-backend)
    ENV_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
    if [ -z "$ENV_NAME" ]; then ENV_NAME="medrax-backend"; fi
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "   Creating conda env ($ENV_NAME) from environment.yml..."
        conda env create -f environment.yml
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    echo "   Python: $(python --version)"
    # Environment.yml installs requirements via pip already; skip redundant pip install at runtime
    PIP_INSTALL=0
else
    echo "Conda not found, using Python venv"
    # Check Python version
    echo "Checking Python version..."
    PYTHON_CMD=""
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo "   Using python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo ""
        echo "ERROR: Python not found!"
        echo ""
        echo "Please install Python 3.11:"
        echo "  macOS: brew install python@3.11"
        echo "  Ubuntu: sudo apt install python3.11"
        echo ""
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo "   Found Python $PYTHON_VERSION"

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment with Python $PYTHON_VERSION..."
        $PYTHON_CMD -m venv venv
        echo "   [OK] Virtual environment created"
    fi
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "   Virtual environment Python: $(python --version | awk '{print $2}')"
fi

# Set up model caching environment variables
echo ""
echo "Configuring model caching..."
export MODEL_CACHE_DIR="./model_cache"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

mkdir -p "$MODEL_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
echo "   [OK] Cache directories ready (existing data preserved)"

# Install/upgrade dependencies (pip works in both conda and venv envs)
echo ""
echo "Installing/upgrading dependencies..."
pip install --upgrade pip > /dev/null 2>&1 || true
echo "   [OK] Pip upgraded"

if [ $PIP_INSTALL -eq 1 ]; then
  echo "   Installing packages from requirements.txt..."
  pip install -r requirements.txt
else
  echo "   Skipping pip install (managed by conda environment.yml)"
fi

echo "   [OK] All dependencies installed"

# Create uploads directory
echo ""
echo "Checking uploads directory..."
mkdir -p uploads
echo "   [OK] Uploads directory ready"

# Initialize database if needed
if [ ! -f "medrax.db" ]; then
    echo ""
    echo "Initializing database..."
    python -m app.database.init_db
    echo "   [OK] Database initialized"
else
    echo ""
    echo "   [OK] Database exists (existing data preserved)"
fi

echo ""
echo "=================================================="
echo "Starting server..."
echo "=================================================="
echo ""
echo "Backend will be available at:"
echo "  API: http://localhost:8000"
echo "  Health: http://localhost:8000/health"
echo "  Interactive Docs: http://localhost:8000/docs"
echo "  ReDoc: http://localhost:8000/redoc"
echo ""
echo "Database: SQLite at ./medrax.db"
echo "Uploads: ./uploads/"
echo "Model Cache: $MODEL_CACHE_DIR"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Start the server
# Use --loop asyncio to avoid conflict with nest_asyncio (used by duckduckgo-search)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --loop asyncio
