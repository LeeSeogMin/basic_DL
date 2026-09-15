#!/usr/bin/env bash
# Auto-detecting verification harness.
# Prints HARNESS_PASS on success and HARNESS_FAIL on failure.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "HARNESS_FAIL: $1"
  exit 1
}

# 실제로 수행한 검사 건수. 0이면 통과로 보지 않는다.
CHECKS=0

run_step() {
  local name="$1"; shift
  echo "--- $name: $*"
  "$@" || fail "$name failed"
  CHECKS=$((CHECKS + 1))
}

echo "=== Harness started ($(date '+%Y-%m-%d %H:%M:%S')) ==="

DETECTED=0

if [ -f package.json ]; then
  DETECTED=1
  echo "[detect] Node.js project"

  if [ ! -d node_modules ] || [ package-lock.json -nt node_modules ] 2>/dev/null \
    || [ pnpm-lock.yaml -nt node_modules ] 2>/dev/null \
    || [ yarn.lock -nt node_modules ] 2>/dev/null; then
    if [ -f pnpm-lock.yaml ]; then
      run_step "install" pnpm install
    elif [ -f yarn.lock ]; then
      run_step "install" yarn install
    else
      run_step "install" npm install
    fi
  else
    echo "[skip] install (node_modules up to date)"
  fi

  has_script() { node -e "process.exit(require('./package.json').scripts?.['$1'] ? 0 : 1)" 2>/dev/null; }

  if has_script lint; then run_step "lint" npm run lint; else echo "[skip] lint (no script)"; fi
  if has_script typecheck; then run_step "typecheck" npm run typecheck; else echo "[skip] typecheck (no script)"; fi
  if has_script test; then run_step "test" npm test -- --run; else echo "[skip] test (no script)"; fi
  if has_script build; then run_step "build" npm run build; else echo "[skip] build (no script)"; fi
fi

if [ -f pyproject.toml ] || [ -f requirements.txt ] || [ -f practice/requirements.txt ]; then
  DETECTED=1
  echo "[detect] Python project"

  PY_RUN=""
  if command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
    PY_RUN="uv run"
  fi

  if command -v ruff >/dev/null 2>&1; then
    run_step "lint" ruff check .
  elif [ -n "$PY_RUN" ] && $PY_RUN ruff --version >/dev/null 2>&1; then
    run_step "lint" $PY_RUN ruff check .
  else
    echo "[skip] lint (ruff not found)"
  fi

  if [ -d tests ] || ls test_*.py >/dev/null 2>&1; then
    if [ -n "$PY_RUN" ]; then
      run_step "test" $PY_RUN pytest -q
    else
      run_step "test" pytest -q
    fi
  else
    echo "[skip] test (no tests found)"
  fi
fi

if [ -f scripts/course_gates.py ]; then
  DETECTED=1
  echo "[detect] 강의자료 저장소 - 학기 운영 게이트"
  run_step "course gates" python3 scripts/course_gates.py
fi

if [ -f Makefile ] && grep -qE '^verify:' Makefile; then
  DETECTED=1
  echo "[detect] Makefile verify target"
  run_step "make verify" make verify
fi

if [ "$DETECTED" -eq 0 ]; then
  echo "[info] 감지된 프로젝트 타입 없음"
  echo "[info] package.json / pyproject.toml / requirements.txt / Makefile 이 생기면 자동 검증이 활성화된다."
fi

# 검사를 하나도 하지 않은 실행은 통과가 아니다.
# lint도 test도 건너뛴 채 HARNESS_PASS를 찍으면 완료 판정이 무조건 통과한다.
if [ "$CHECKS" -eq 0 ]; then
  echo "HARNESS_FAIL: 검사를 하나도 수행하지 못했다. 통과로 볼 근거가 없다."
  exit 1
fi

echo "=== 수행한 검사 ${CHECKS}건 ==="
echo "HARNESS_PASS"

