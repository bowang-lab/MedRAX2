#!/bin/bash
# Comprehensive test runner script

set -e

echo "======================================================================"
echo "MedRAX Backend - Comprehensive Test Suite"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to backend directory
cd "$(dirname "$0")"

# Check if conda is available and environment exists
USE_CONDA=0
if command -v conda &> /dev/null; then
    ENV_NAME=$(grep -E '^name:' environment.yml 2>/dev/null | awk '{print $2}')
    if [ -n "$ENV_NAME" ] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        USE_CONDA=1
        echo "Using conda environment: $ENV_NAME"
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi
fi

# Fall back to venv if conda not available
if [ $USE_CONDA -eq 0 ]; then
    if [ ! -d "venv" ]; then
        echo -e "${RED}Error: Neither conda environment nor venv found!${NC}"
        echo "Create environment first:"
        echo "  Conda: conda env create -f environment.yml"
        echo "  Venv: python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    echo "Using Python venv"
    source venv/bin/activate
fi

echo ""
echo "Activated environment: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run tests
echo "======================================================================"
echo "Running Full Test Suite..."
echo "======================================================================"

python -m pytest tests/ -v --tb=short --durations=10

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Test Summary"
echo "======================================================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    
    # Run coverage if all tests pass
    echo ""
    echo "======================================================================"
    echo "Generating Coverage Report..."
    echo "======================================================================"
    python -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html -q
    
    echo ""
    echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
    
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo ""
    echo "To debug failed tests, run:"
    echo "  pytest tests/ -v --tb=long"
    echo "  pytest tests/test_specific.py::test_name -v --pdb"
fi

echo ""
echo "======================================================================"

exit $TEST_EXIT_CODE
