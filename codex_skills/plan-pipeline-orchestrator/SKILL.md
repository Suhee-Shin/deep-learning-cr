# Plan Pipeline - Orchestrator (Skill 6)

Orchestrate the full plan to code pipeline.

## When to use
- User says '전체 파이프라인 돌려', '오케스트레이션', 'plan->prompt->code 파이프라인'
- Wants end-to-end generation

## Role (control plane only)
You coordinate steps and verify files in the **main session**. You must **not** do the creative work that Skill 4/5 are responsible for.

## Inputs
- Plan name: `##`
- Existing artifacts:
  - `bot/plans/plan-##/plan.md`
  - `bot/plans/plan-##/tasks/*.md`
  - `bot/plans/plan-##/common_context_prompt.md`

## Outputs
- `bot/plans/plan-##/code_gen_prompt/task-XX-*.md`
- `bot/plans/plan-##/code/` (code files)
- `bot/plans/plan-##/orchestrator.log` (JSONL)

## Directory contract
Ensure these exist:
- `bot/plans/plan-##/tasks/`
- `bot/plans/plan-##/code_gen_prompt/`
- `bot/plans/plan-##/code/`

## State machine (per task)
- `PENDING`
- `PROMPT_GENERATED`
- `CODE_GENERATED`
- `DONE`
- `FAILED`

## Log format (append-only JSONL)
```json
{"ts":"2026-02-06T03:00:00+09:00","taskId":"task-01","state":"PROMPT_GENERATED","details":{"promptPath":"code_gen_prompt/task-01-...md"}}
```

## Task loop algorithm (per-task Prompt->Code loop)
For each task, complete **Skill 4 -> Skill 5** (prompt -> code) with verification, then move to the next task.

### Per-work-item loop
For each work item in the ordered list:

1. Select current work item (`workId = XX or XX-YY`)
2. Validate prerequisites exist
3. Run Skill 4 (Prompt Architect) - create prompt file
4. Verify prompt output (file exists, template sections present)
5. Run Skill 5 (Code Executor) - create code files
6. Verify code output (expected files exist)
7. Mark work item complete, log `DONE`

### Loop exit
- If no remaining tasks: Ask user whether to proceed to **Skill 7 (Integration Verify)**
- Otherwise: Continue to next task

**Important:** Do NOT generate prompts for all tasks first. Operate as a strict per-task loop.

## Failure policy
- If a step fails validation:
  - Log `state="FAILED"` with reason
  - Retry once
  - If still failing, stop pipeline and report which task/file failed

## Constraints
- All communication is file-based.
- Do not "improve" tasks or prompts yourself.
