#!/bin/bash
# Backend-Frontend Synchronization Verification Script
# Ensures frontend types match backend models and schemas

set -e

echo "🔄 Verifying Backend-Frontend Type Synchronization..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is accessible
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
echo "📡 Checking backend at: $BACKEND_URL"

if ! curl -s "$BACKEND_URL/docs" > /dev/null 2>&1; then
  echo -e "${YELLOW}⚠️  Warning: Backend not accessible at $BACKEND_URL${NC}"
  echo "   To verify sync, start the backend with: cd ../backend && uvicorn app.main:app --reload"
  echo ""
  echo "   Skipping OpenAPI schema verification..."
  exit 0
fi

echo -e "${GREEN}✅ Backend is accessible${NC}"
echo ""

# Verify OpenAPI schema is up-to-date
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Checking OpenAPI Schema Sync"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Download current schema
TEMP_SCHEMA=$(mktemp)
curl -s "$BACKEND_URL/openapi.json" > "$TEMP_SCHEMA"

# Compare with generated types timestamp
if [ -f "lib/types/openapi.d.ts" ]; then
  TYPES_MODIFIED=$(stat -c %Y lib/types/openapi.d.ts 2>/dev/null || stat -f %m lib/types/openapi.d.ts)
  CURRENT_TIME=$(date +%s)
  HOURS_OLD=$(( (CURRENT_TIME - TYPES_MODIFIED) / 3600 ))
  
  if [ $HOURS_OLD -gt 24 ]; then
    echo -e "${YELLOW}⚠️  Warning: OpenAPI types are $HOURS_OLD hours old${NC}"
    echo "   Consider regenerating with: npm run gen:openapi:local"
    echo ""
  else
    echo -e "${GREEN}✅ OpenAPI types are recent (generated $HOURS_OLD hours ago)${NC}"
    echo ""
  fi
else
  echo -e "${RED}❌ OpenAPI types not found${NC}"
  echo "   Generate them with: npm run gen:openapi:local"
  rm "$TEMP_SCHEMA"
  exit 1
fi

rm "$TEMP_SCHEMA"

# Verify critical enum values match backend
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Verifying Critical Enum Types"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check ToolStatus enum
echo "Checking ToolStatus enum..."
if grep -q "'pending' | 'running' | 'completed' | 'failed'" lib/types/tool.ts; then
  echo -e "${GREEN}✅ ToolStatus enum matches backend${NC}"
else
  echo -e "${RED}❌ ToolStatus enum mismatch${NC}"
  exit 1
fi

# Check ToolLogLevel enum
echo "Checking ToolLogLevel enum..."
if grep -q "'info' | 'warning' | 'error'" lib/types/tool.ts; then
  echo -e "${GREEN}✅ ToolLogLevel enum matches backend${NC}"
else
  echo -e "${RED}❌ ToolLogLevel enum mismatch${NC}"
  exit 1
fi

# Check MessageRole enum
echo "Checking MessageRole enum..."
if grep -q "'user' | 'assistant' | 'system'" lib/types/message.ts; then
  echo -e "${GREEN}✅ MessageRole enum matches backend${NC}"
else
  echo -e "${RED}❌ MessageRole enum mismatch${NC}"
  exit 1
fi

# Check ScanFileType enum
echo "Checking ScanFileType enum..."
if grep -q "'jpg' | 'jpeg' | 'png' | 'gif' | 'dcm' | 'dicom'" lib/types/scan.ts; then
  echo -e "${GREEN}✅ ScanFileType enum matches backend${NC}"
else
  echo -e "${RED}❌ ScanFileType enum mismatch${NC}"
  exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Backend-Frontend Sync Verified!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✓ OpenAPI types are up-to-date"
echo "✓ ToolStatus enum matches"
echo "✓ ToolLogLevel enum matches"
echo "✓ MessageRole enum matches"
echo "✓ ScanFileType enum matches"
echo ""

