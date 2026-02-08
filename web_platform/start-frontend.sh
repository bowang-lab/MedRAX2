#!/bin/bash

# MedRAX Web Platform - Frontend Development Server
echo "Starting MedRAX Frontend Server..."
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "Error: Must run from web_platform directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Check Node.js
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "Node.js not found!"
    echo ""
    echo "Please install Node.js 18+ from:"
    echo "  https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "   Found Node.js $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "npm not found!"
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "   Found npm $NPM_VERSION"

# Install dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo ""
    echo "Installing dependencies..."
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
    echo "   Dependencies installed"
    cd ..
else
    echo "   Dependencies already installed"
fi

# Check for .env.local
if [ ! -f "frontend/.env.local" ]; then
    echo ""
    echo "Creating .env.local..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
    echo "   Created .env.local with default API URL"
fi

# Display server info
echo ""
echo "=============================================="
echo "Starting Next.js with hot reload..."
echo ""
echo "Frontend will be available at:"
echo "  http://localhost:3000"
echo ""
echo "API Backend: http://localhost:8000"
echo ""
echo "First load might take ~10 seconds to compile"
echo "Subsequent changes will be instant!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Start frontend
cd frontend
exec npm run dev
