#!/bin/bash

set -e

echo "=================================================="
echo "Starting MedRAX Backend Server (production helper)"
echo "=================================================="
echo ""

# Expose preferred GPUs for multi-GPU tool placement (see TOOL_GPU_* in backend .env).
# Order 2,1 => cuda:0 maps to physical GPU 2, cuda:1 to physical GPU 1 (TOOL_GPU_PHYSICAL_ORDER default).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,1}
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Prefer conda env if available
USE_CONDA=0
if command -v conda &> /dev/null; then
    USE_CONDA=1
fi

echo "Checking backend environment..."

cd "$(dirname "$0")/backend"

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    while IFS='=' read -r key value; do
        if [[ ! "$key" =~ ^[[:space:]]*# && -n "$key" ]]; then
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            export "$key=$value"
        fi
    done < .env
    echo "   [OK] Environment variables loaded"
else
    echo "   [WARNING] No .env file found in backend directory"
fi

# Ensure critical secrets exist before boot
REQUIRED_VARS=("SECRET_KEY" "API_SECRET_KEY")
MISSING=0
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "   [ERROR] Missing required environment variable: $var"
        MISSING=1
    fi
done
if [ $MISSING -ne 0 ]; then
    echo ""
    echo "Please set the required secrets (e.g., in backend/.env or host env) and retry."
    exit 1
fi

# Enforce API secret usage by default
export REQUIRE_API_SECRET=${REQUIRE_API_SECRET:-true}

# CORS defaults (override in .env for your Vercel domain)
if [ -z "${CORS_ORIGINS:-}" ]; then
    export CORS_ORIGINS="https://med-rax-2.vercel.app"
    echo "   [INFO] CORS_ORIGINS not set; defaulting to ${CORS_ORIGINS}"
    echo "         Set CORS_ORIGINS to your exact frontend origin(s) for production."
else
    echo "   [OK] CORS_ORIGINS set to ${CORS_ORIGINS}"
fi

PIP_INSTALL=1
if [ $USE_CONDA -eq 1 ]; then
    echo "Using conda environment"
    ENV_NAME="medrax2"
    if ! CONDA_NO_PLUGINS=true conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "   Creating conda env ($ENV_NAME)..."
        CONDA_NO_PLUGINS=true conda create -n "$ENV_NAME" python=3.11 -y
    fi
    # shellcheck disable=SC1091
    source "$(CONDA_NO_PLUGINS=true conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    echo "   Python: $(python --version)"
else
    echo "Conda not found, using Python venv"
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
  echo "   Installing packages from pyproject.toml..."
  pip install -e ../..
fi

echo "   [OK] All dependencies installed"

# Create uploads and temp directories
echo ""
echo "Checking uploads directory..."
mkdir -p uploads
echo "   [OK] Uploads directory ready"

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
        echo "   To fix: Delete conda environment and recreate:"
        echo "     conda env remove -n medrax2"
        echo "     conda create -n medrax2 python=3.11 -y"
        echo "     pip install -e ../.."
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

# Allow host/port overrides with safer production defaults
BACKEND_HOST=${HOST:-0.0.0.0}
# Use a less-common port by default to reduce generic scans (override via PORT)
BACKEND_PORT=${PORT:-8787}

echo "Backend will be available at:"
echo "  API: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "  Health: http://${BACKEND_HOST}:${BACKEND_PORT}/health"
echo "  Interactive Docs: http://${BACKEND_HOST}:${BACKEND_PORT}/docs"
echo "  ReDoc: http://${BACKEND_HOST}:${BACKEND_PORT}/redoc"
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
export EAGER_LOAD_TOOLS=0

# Use 0.0.0.0 when running behind a reverse proxy / firewall for production
uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --loop asyncio
