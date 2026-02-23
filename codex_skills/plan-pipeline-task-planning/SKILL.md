# Plan Pipeline - Task Planning (Skill 2)

Expand a plan.md into independent executable task markdowns.

## When to use
- User says '계획 구체화', 'task로 쪼개', '플랜을 task로 분해'
- Wants Step to Task decomposition

## Goal
Convert each plan step into a **standalone task** specification that is detailed enough to succeed when executed sequentially.

## Inputs
- `bot/plans/plan-##/plan.md`

## Outputs
### Default (single-file task)
- `bot/plans/plan-##/tasks/task-01-<slug>.md`
- `bot/plans/plan-##/tasks/task-02-<slug>.md`
- ...

### When a step is large/complex (task folder + sequential subtasks)
- parent task folder: `bot/plans/plan-##/tasks/task-XX/`
- overview: `bot/plans/plan-##/tasks/task-XX/README.md`
- subtasks: `bot/plans/plan-##/tasks/task-XX/task-XX-01-<slug>.md`, ...

## Steps
1. Read `plan.md` and list the numbered steps.
2. For each step, decide whether it can be executed as a single task:
   - If yes: create one `tasks/task-XX-<slug>.md`.
   - If no (too large/ambiguous): create a folder `tasks/task-XX/` and split into sequential subtasks.
3. Each (sub)task must be runnable using **only** the files declared as inputs.
4. Ensure the execution order is explicit.
5. After generating all tasks, ask the user whether to proceed to **Skill 3 (Common Context Builder)**.
   - Example: "task 파일들 만들었어. 다음으로 공통 컨텍스트(common_context_prompt.md)를 만드는 Skill 3도 진행할까? (y/n)"

## Task Template (tasks/task-XX-<slug>.md)
```md
# Task XX: <Title>

## 목적
<what this task accomplishes>

## 입력
- <file(s) or user-provided text that must exist>

## 출력
- <file(s) to be produced>

## 제약
- <non-functional constraints, safety, performance, compatibility>
- 파일 기반 재현성 유지 (no hidden assumptions)

## 완료 조건
- <objective checklist>
```

## Constraints
- Do not edit `plan.md` (treat as source-of-truth).
- Do not generate code here.
- Prefer fewer, larger tasks over many tiny ones (aim 5-15 tasks unless the plan is huge).
