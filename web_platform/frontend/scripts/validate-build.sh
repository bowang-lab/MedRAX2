#!/bin/bash
# Complete Build Validation Script
# Runs all checks: types, linting, and production build

set -e

echo "🚀 Starting Complete Build Validation..."
echo ""

# Step 1: Type checking
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Step 1: TypeScript Type Checking"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm run type-check
echo "✅ Type checking passed!"
echo ""

# Step 2: Type safety audit
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Step 2: Type Safety Audit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/audit-types.sh
echo ""

# Step 3: ESLint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Step 3: ESLint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm run lint
echo "✅ Linting passed!"
echo ""

# Step 4: Production build
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏗️  Step 4: Production Build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npm run build
echo "✅ Production build completed!"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL VALIDATION CHECKS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✓ TypeScript type checking"
echo "✓ Type safety audit"
echo "✓ ESLint"
echo "✓ Production build"
echo ""
echo "🎉 Your code is production-ready!"

