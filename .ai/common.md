# Common Agent Operating Rules

## Core Principle

Do not rely on self-reporting. Use files, commands, logs, and generated artifacts as evidence.

## Work Cycle

1. Read `AGENTS.md`, `CODEX.md`, and `harness.md` when the task touches project setup or workflow.
2. Read `.ai/context.md`, `.ai/todo.md`, and `.ai/gotchas.md` when the task spans more than a local edit.
3. Inspect relevant source files.
4. Modify only the requested scope.
5. Run `./scripts/harness.sh`.
6. Update memory files only when there is real new state.
7. Report changed files, verification, and remaining risks.

## Completion Criteria

- Requested change is implemented.
- Practice code, if changed, has execution evidence under `practice/chapter{N}/results/`.
- `./scripts/harness.sh` prints `HARNESS_PASS`.
- Any skipped verification is explicitly reported.

