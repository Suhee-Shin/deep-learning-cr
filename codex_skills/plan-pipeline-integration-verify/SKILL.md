# Plan Pipeline - Integration Verify (Skill 7)

Run end-to-end integration verification for generated code.

## When to use
- User says '통합 검증', '전체 테스트', '빌드 검증', 'integration verify'
- Wants to confirm the generated code actually runs/tests/builds

## Role
You are the **Verifier**. Validate that generated project artifacts under `bot/plans/plan-##/code/` are runnable/buildable/testable.

## Inputs
- Plan name: `##`
- Existing artifacts:
  - `bot/plans/plan-##/plan.md`
  - `bot/plans/plan-##/common_context_prompt.md`
  - `bot/plans/plan-##/tasks/*.md`
  - `bot/plans/plan-##/code_gen_prompt/*.md`
  - `bot/plans/plan-##/code/**`

## Outputs
Write verification reports under:
- `bot/plans/plan-##/code/REPORTS/integration-verify.summary.md`
- `bot/plans/plan-##/code/REPORTS/integration-verify.log.md`
- Optional: `bot/plans/plan-##/code/REPORTS/integration-verify.failures.json`

## Steps
1. Ensure report directory exists: `bot/plans/plan-##/code/REPORTS/`
2. Discover project type(s) under `bot/plans/plan-##/code/`:
   - `package.json` -> Node project
   - `pyproject.toml` or `requirements.txt` -> Python project
3. Define the verification plan (write to `integration-verify.summary.md`):
   - Dependency install
   - Lint/format checks
   - Unit tests
   - Build/compile
   - Smoke run
4. Execute checks, append outputs to `integration-verify.log.md`.

### Python checks (if applicable)
- `pip install -e .` or `pip install -r requirements.txt`
- `pytest -q` if tests exist
- Basic import smoke test

### Node checks (if applicable)
- `npm ci` / `pnpm install --frozen-lockfile` / `yarn install --frozen-lockfile`
- `npm test` / `npm run lint` / `npm run build` (if scripts exist)

5. Summarize results in `integration-verify.summary.md`:
   - status: PASS/FAIL
   - which checks ran
   - what failed + next actions

6. If any step fails:
   - capture exact command, exit code, stderr/stdout
   - write to `integration-verify.failures.json`

## Constraints
- Do NOT modify source code during verification.
- Do NOT invent new dependencies or refactor.
- If secrets are needed, record as blocker in report.
