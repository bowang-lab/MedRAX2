#!/bin/bash
# Type Safety Audit Script
# Checks for problematic type casts and ensures type safety standards

set -e

echo "🔍 Auditing Frontend Type Safety..."
echo ""

# Count 'as any' casts (excluding documented exceptions)
ANY_COUNT=$(find lib components -name "*.ts" -o -name "*.tsx" | xargs grep " as any" 2>/dev/null | wc -l || echo "0")
echo "❌ 'as any' casts found: $ANY_COUNT"

# Count 'as unknown' casts (excluding documented exceptions)
UNKNOWN_COUNT=$(find lib components -name "*.ts" -o -name "*.tsx" | xargs grep " as unknown" 2>/dev/null | wc -l || echo "0")
echo "⚠️  'as unknown' casts found: $UNKNOWN_COUNT"

# Count '@ts-ignore' comments
IGNORE_COUNT=$(find lib components -name "*.ts" -o -name "*.tsx" | xargs grep "@ts-ignore" 2>/dev/null | wc -l || echo "0")
echo "⚠️  '@ts-ignore' comments found: $IGNORE_COUNT"

# Count '@ts-expect-error' comments
EXPECT_ERROR_COUNT=$(find lib components -name "*.ts" -o -name "*.tsx" | xargs grep "@ts-expect-error" 2>/dev/null | wc -l || echo "0")
echo "⚠️  '@ts-expect-error' comments found: $EXPECT_ERROR_COUNT"

# Count documented/verified casts (these are OK)
DOCUMENTED_CASTS=$(find lib components -name "*.ts" -o -name "*.tsx" | xargs grep -E "as Api.*Response|as 'info'|as 'jpg'|// Backend" 2>/dev/null | wc -l || echo "0")
echo "✅ Documented/verified casts: $DOCUMENTED_CASTS"

echo ""
echo "📊 Summary:"
TOTAL_ISSUES=$((ANY_COUNT + IGNORE_COUNT + EXPECT_ERROR_COUNT))

if [ $ANY_COUNT -eq 0 ] && [ $IGNORE_COUNT -eq 0 ] && [ $EXPECT_ERROR_COUNT -eq 0 ]; then
  echo "✅ Type safety check PASSED!"
  echo "   - No 'as any' casts"
  echo "   - No '@ts-ignore' suppressions"
  echo "   - No '@ts-expect-error' suppressions"
  echo "   - $DOCUMENTED_CASTS documented casts for backend compatibility"
  exit 0
else
  echo "❌ Type safety issues found!"
  echo "   - Total problematic casts/ignores: $TOTAL_ISSUES"
  echo ""
  echo "🔎 Detailed breakdown:"
  
  if [ $ANY_COUNT -gt 0 ]; then
    echo ""
    echo "━━━ 'as any' casts ━━━"
    find lib components -name "*.ts" -o -name "*.tsx" | xargs grep -n " as any" 2>/dev/null | head -n 10
  fi
  
  if [ $IGNORE_COUNT -gt 0 ]; then
    echo ""
    echo "━━━ '@ts-ignore' comments ━━━"
    find lib components -name "*.ts" -o -name "*.tsx" | xargs grep -n "@ts-ignore" 2>/dev/null | head -n 10
  fi
  
  exit 1
fi

