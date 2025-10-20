#!/bin/bash

set -e

echo "=================================================="
echo "Starting MedRAX Backend Server"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."

# Try to find Python 3.11 first (recommended)
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
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "   Found Python $PYTHON_VERSION"

# Warn if not Python 3.11
if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "11" ]; then
    echo ""
    echo "   [WARNING] Python 3.11 is STRONGLY RECOMMENDED"
    echo "   Current version: $PYTHON_VERSION"
    
    if [ "$PYTHON_MINOR" -ge "13" ]; then
        echo ""
        echo "   [CRITICAL] Python 3.13+ has COMPATIBILITY ISSUES:"
        echo "   - SimpleITK not available"
        echo "   - Many packages missing wheels"
        echo "   - Medical imaging tools may not work"
        echo ""
        echo "   PLEASE install Python 3.11:"
        echo "   brew install python@3.11"
        echo ""
        read -p "   Continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled. Please install Python 3.11 first."
            exit 1
        fi
    elif [ "$PYTHON_MINOR" -eq "12" ]; then
        echo "   Python 3.12 works but 3.11 is more stable"
    elif [ "$PYTHON_MINOR" -lt "11" ]; then
        echo "   Python 3.11 offers better performance and compatibility"
    fi
    echo ""
fi

cd backend

# Check if virtual environment exists and if it's using the right Python version
if [ -d "venv" ]; then
    # Check if existing venv uses the correct Python version
    if [ -f "venv/bin/python" ]; then
        EXISTING_VENV_VERSION=$(venv/bin/python --version 2>&1 | awk '{print $2}')
        EXISTING_MINOR=$(echo $EXISTING_VENV_VERSION | cut -d. -f2)
        
        if [ "$EXISTING_MINOR" != "11" ]; then
            echo ""
            echo "   [WARNING] Existing venv uses Python $EXISTING_VENV_VERSION"
            echo "   Python 3.11 is required for best compatibility"
            echo "   The old venv will be backed up and a new one created"
            echo ""
            
            # Backup old venv with timestamp
            BACKUP_NAME="venv_backup_$(date +%Y%m%d_%H%M%S)"
            mv venv "$BACKUP_NAME"
            echo "   [OK] Old venv backed up as: $BACKUP_NAME"
            
            # Create new venv with correct Python
            echo "   Creating new virtual environment with Python $PYTHON_VERSION..."
            $PYTHON_CMD -m venv venv
            echo "   [OK] New virtual environment created"
        else
            echo "   [OK] Using existing virtual environment (Python $EXISTING_VENV_VERSION)"
        fi
    else
        echo "   [WARNING] Existing venv appears corrupted, recreating..."
        BACKUP_NAME="venv_backup_$(date +%Y%m%d_%H%M%S)"
        mv venv "$BACKUP_NAME"
        $PYTHON_CMD -m venv venv
        echo "   [OK] Virtual environment recreated"
    fi
else
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    $PYTHON_CMD -m venv venv
    echo "   [OK] Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Virtual environment Python: $VENV_PYTHON_VERSION"

# Set up model caching environment variables
echo ""
echo "Configuring model caching..."
export MODEL_CACHE_DIR="./model_cache"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

# Create cache directories (only create, never delete existing data)
mkdir -p "$MODEL_CACHE_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"
echo "   [OK] Cache directories ready (existing data preserved)"


# Install/upgrade dependencies
echo ""
echo "Installing/upgrading dependencies..."
echo "   (This may take 10-20 minutes on first run, ~5-10GB download)"
pip install --upgrade pip > /dev/null 2>&1
echo "   [OK] Pip upgraded"

echo "   Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "   [OK] All dependencies installed"

# Create uploads directory (only create, never delete)
echo ""
echo "Checking uploads directory..."
mkdir -p uploads
echo "   [OK] Uploads directory ready"

# Check if database exists (never delete existing database)
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
echo "Python Version: $VENV_PYTHON_VERSION"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
