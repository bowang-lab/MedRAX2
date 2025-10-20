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

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Run ./setup.sh first to create the virtual environment."
    exit 1
fi

source venv/bin/activate

echo ""
echo "Activated virtual environment: $(which python)"
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
