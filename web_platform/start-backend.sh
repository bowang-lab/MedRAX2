#!/bin/bash

set -e

# GPU selection (optional). Respect existing CUDA_VISIBLE_DEVICES if set.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES not set; using all visible GPUs"
else
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

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
ENV_FILE="environment.yml"
CONDA_ENV_PATH="$(pwd)/conda_env"
if [ ! -d "$CONDA_ENV_PATH" ] && [ ! -f "$ENV_FILE" ]; then
    echo "   [WARNING] $ENV_FILE not found and conda_env missing; falling back to Python venv"
    USE_CONDA=0
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    # Export variables from .env file, properly handling spaces and special characters
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^[[:space:]]*# && -n "$key" ]]; then
            # Remove leading/trailing whitespace and export
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            export "$key=$value"
        fi
    done < .env
    echo "   [OK] Environment variables loaded"
else
    echo "   [WARNING] No .env file found in backend directory"
fi

PIP_INSTALL=1
if [ $USE_CONDA -eq 1 ]; then
    echo "Using conda environment"
    if [ -d "$CONDA_ENV_PATH" ]; then
        echo "   Using local conda env at $CONDA_ENV_PATH"
        # shellcheck disable=SC1091
        source "$(CONDA_NO_PLUGINS=true conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_PATH"
    else
        # Read env name from environment.yml (fallback to medrax-backend)
        ENV_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
        if [ -z "$ENV_NAME" ]; then ENV_NAME="medrax-backend"; fi
        if ! CONDA_NO_PLUGINS=true conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
            echo "   Creating conda env ($ENV_NAME) from environment.yml..."
            CONDA_NO_PLUGINS=true conda env create -f environment.yml
        fi
        # shellcheck disable=SC1091
        source "$(CONDA_NO_PLUGINS=true conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi
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

# Create cache directories if they don't exist
if [ -n "$MODEL_CACHE_DIR" ]; then
    mkdir -p "$MODEL_CACHE_DIR"
    echo "   [OK] Model cache directory ready: $MODEL_CACHE_DIR"
fi
if [ -n "$HUGGINGFACE_CACHE_DIR" ]; then
    mkdir -p "$HUGGINGFACE_CACHE_DIR"
    echo "   [OK] HuggingFace cache directory ready: $HUGGINGFACE_CACHE_DIR"
fi
if [ -n "$TORCH_CACHE_DIR" ]; then
    mkdir -p "$TORCH_CACHE_DIR"
    echo "   [OK] Torch cache directory ready: $TORCH_CACHE_DIR"
fi

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

# Create temp directory for tool outputs (segmentations, generated images, etc.)
echo ""
echo "Checking temp directory..."
mkdir -p temp
echo "   [OK] Temp directory ready for tool outputs"

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

# Check GPU support
echo ""
echo "Checking GPU support..."
GPU_CHECK=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "error")

if [ "$GPU_CHECK" = "cuda" ]; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
    echo "   [OK] GPU acceleration enabled"
    echo "       GPUs: $GPU_COUNT"
    echo "       PyTorch CUDA: $CUDA_VERSION"
elif [ "$GPU_CHECK" = "cpu" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "   [WARNING] NVIDIA GPU detected but PyTorch is CPU-only!"
        echo "   "
        echo "   To fix: Delete conda environment and recreate:"
        echo "     conda env remove -n alankrit-medrax2"
        echo "     conda env create -f backend/environment.yml"
        echo "   "
        echo "   Continuing with CPU (tools will be slower)..."
    else
        echo "   [OK] No GPU detected (CPU mode)"
    fi
else
    echo "   [WARNING] Could not check GPU status"
fi

# Validate tools (optional check, won't block startup)
echo ""
echo "Validating tools..."
if python ../../medrax/tools/validate_tools.py 2>/dev/null; then
    echo "   [OK] All tools validated successfully"
else
    echo "   WARNING: Tool validation found issues (see medrax/tools/validate_tools.py)"
    echo "   Continuing with startup..."
fi

echo ""
echo "=================================================="
echo "Starting server..."
echo "=================================================="
echo ""
BACKEND_PORT="${BACKEND_PORT:-8000}"

echo "Backend will be available at:"
echo "  API: http://localhost:$BACKEND_PORT"
echo "  Health: http://localhost:$BACKEND_PORT/health"
echo "  Interactive Docs: http://localhost:$BACKEND_PORT/docs"
echo "  ReDoc: http://localhost:$BACKEND_PORT/redoc"
echo ""
echo "Database: SQLite at ./medrax.db"
echo "Uploads: ./uploads/"
echo "Temp Files: ./temp/"
if [ -n "$MODEL_CACHE_DIR" ]; then
    echo "Model Cache: $MODEL_CACHE_DIR"
fi
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Start the server
# Enable eager loading of tools on startup for better user experience
export EAGER_LOAD_TOOLS=0

# Use --loop asyncio to avoid conflict with nest_asyncio (used by duckduckgo-search)
# Use 127.0.0.1 (localhost) for development security - change to 0.0.0.0 in production with proper firewall
# Removed --reload flag for stability (it was causing auto-restarts on file changes)
uvicorn app.main:app --host 127.0.0.1 --port $BACKEND_PORT --loop asyncio
