# Plan Pipeline - Common Context Builder (Skill 3)

Create the global common context prompt for a plan.

## When to use
- User says '공통 컨텍스트', 'global context'
- Needs shared rules/assumptions for downstream prompt+code generation

## Goal
Generate a single global context file that all downstream prompts and code generation must follow.

## Inputs
- `bot/plans/plan-##/plan.md`
- Task specs under `bot/plans/plan-##/tasks/`:
  - single task files: `tasks/task-XX-<slug>.md`
  - task folders: `tasks/task-XX/README.md` + `tasks/task-XX/task-XX-YY-<slug>.md`

## Outputs
- `bot/plans/plan-##/common_context_prompt.md`

## Steps
1. Read `plan.md` for project goal and rough steps.
2. Read all task specs under `tasks/`.
3. Infer and write:
   - shared assumptions
   - shared file layout
   - global rules (style, testing, error handling)
   - guidance for sequential execution of subtasks
4. Write `common_context_prompt.md`.
5. After writing, ask whether to proceed to **Skill 6 (Orchestrator)**.
   - Example: "common_context_prompt.md 만들었어. 이제 Skill 6(오케스트레이터)로 (task/subtask) 단위로 (Skill4 프롬프트 생성 -> Skill5 코드 생성) 루프를 돌려서 자동 생성할까? (y/n)"

## Output template (common_context_prompt.md)
```md
# Common Context Prompt

## Project Overview
<1-3 paragraphs summarizing what the plan is building>

## File/Directory Convention
- plan root: bot/plans/plan-##/
- tasks: tasks/task-XX-<slug>.md or tasks/task-XX/...
- prompts: code_gen_prompt/
- outputs: code/

## Global Rules
- 파일 기반 재현성: 출력은 입력 파일들만 근거로 생성
- 모듈화/책임 분리 우선
- 테스트/검증 기준 명시

## Tech Assumptions
- Language: <decide if specified; else leave 'TBD'>
- Runtime: <TBD>

## Definition of Done (global)
- <global acceptance criteria>
```

## Constraints
- Do not change any task content; only synthesize.
- If language/runtime is unknown, explicitly mark as TBD (do not guess).
