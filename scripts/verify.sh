#!/usr/bin/env bash
# Fast verification: lint and tests only.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "VERIFY_FAIL: $1"
  exit 1
}

if [ -f package.json ]; then
  has_script() { node -e "process.exit(require('./package.json').scripts?.['$1'] ? 0 : 1)" 2>/dev/null; }
  if has_script lint; then npm run lint || fail "lint"; fi
  if has_script test; then npm test -- --run || fail "test"; fi
fi

if [ -f pyproject.toml ] || [ -f requirements.txt ] || [ -f practice/requirements.txt ]; then
  command -v ruff >/dev/null 2>&1 && { ruff check . || fail "lint"; }
  if [ -d tests ] || ls test_*.py >/dev/null 2>&1; then
    pytest -q || fail "test"
  fi
fi

echo "VERIFY_PASS"

