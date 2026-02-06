# Plan Pipeline - Prompt Architect (Skill 4)

Generate a structured coding prompt per task.

## When to use
- User says '코딩 프롬프트 생성', 'Prompt Architect'
- Needs task to coding-prompt conversion

## Role
You are the **Prompt Architect**. You only create **coding prompts** (not code).

## Inputs (per work item)
- `bot/plans/plan-##/common_context_prompt.md`
- `bot/plans/plan-##/plan.md`
- One work item spec (task or subtask):
  - single task: `bot/plans/plan-##/tasks/task-XX-<slug>.md`
  - subtask: `bot/plans/plan-##/tasks/task-XX/task-XX-YY-<slug>.md`

## Output (per work item)
- `bot/plans/plan-##/code_gen_prompt/task-<ID>-<slug>.md`
  - where `<ID>` is `XX` for a single task, or `XX-YY` for a subtask

## Hard rules
- Output must be based only on the input files listed above.
- Do NOT generate code.
- Do NOT modify `plan.md` or any `tasks/*.md`.
- Do NOT use external context (no hidden requirements).

## Steps (per work item)
1. Read common context + the work item spec (task or subtask).
2. Produce exactly one coding prompt file for that work item.
3. Ensure the prompt includes explicit file outputs, constraints, and acceptance criteria.
4. After writing the prompt file, stop. Do NOT generate code.

## Mandatory prompt template (code_gen_prompt/task-<ID>-<slug>.md)
```md
# Coding Prompt: Task-<ID> <Title>

## Purpose

## Context
- (Summarize relevant parts of common_context_prompt.md)

## Task Description
- (Summarize the work item spec: Task-<ID>)

## Inputs
- (List exact input file paths)

## Outputs
- (List exact output file paths; include extensions)

## Constraints
- (From task/subtask + global rules)

## Implementation Rules
- (Style, architecture boundaries, error handling, tests)

## Expected Result / Acceptance Criteria
- (Objective checklist)
```
