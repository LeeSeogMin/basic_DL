#!/usr/bin/env bash
# Fast verification: lint and tests only.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "VERIFY_FAIL: $1"
  exit 1
}

# 실제로 수행한 검사 건수. 0이면 통과로 보지 않는다.
CHECKS=0

if [ -f package.json ]; then
  has_script() { node -e "process.exit(require('./package.json').scripts?.['$1'] ? 0 : 1)" 2>/dev/null; }
  if has_script lint; then npm run lint || fail "lint"; CHECKS=$((CHECKS + 1)); fi
  if has_script test; then npm test -- --run || fail "test"; CHECKS=$((CHECKS + 1)); fi
fi

if [ -f pyproject.toml ] || [ -f requirements.txt ] || [ -f practice/requirements.txt ]; then
  if command -v ruff >/dev/null 2>&1; then
    ruff check . || fail "lint"
    CHECKS=$((CHECKS + 1))
  fi
  if [ -d tests ] || ls test_*.py >/dev/null 2>&1; then
    pytest -q || fail "test"
    CHECKS=$((CHECKS + 1))
  fi
fi

# 빠른 검증에는 PDF 재생성이 필요 없는 두 게이트만 넣는다.
if [ -f scripts/course_gates.py ]; then
  python3 scripts/course_gates.py --gates g1,g2 || fail "course gates"
  CHECKS=$((CHECKS + 1))
fi

if [ "$CHECKS" -eq 0 ]; then
  echo "VERIFY_FAIL: 검사를 하나도 수행하지 못했다. 통과로 볼 근거가 없다."
  exit 1
fi

echo "=== 수행한 검사 ${CHECKS}건 ==="
echo "VERIFY_PASS"

