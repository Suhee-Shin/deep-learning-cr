# Plan Pipeline - Planning (Skill 1)

Create a high-level sequential plan (plan.md) from a user request.

## When to use
- User says '계획 세워줘', 'plan 만들어줘'
- User provides '##(계획 이름)' to start the pipeline

## Goal
Given a plan name `##` and a short user request, create a **high-level sequential plan** (no implementation details).

## Inputs
- Plan name: `##` (used to create folder `bot/plans/plan-##/`)
- User request text

## Outputs
- `bot/plans/plan-##/plan.md`

## Steps
1. Normalize `##` to a safe folder slug (lowercase, digits/hyphen; keep Korean if needed but avoid spaces).
2. Ensure directory exists: `bot/plans/plan-##/`
3. Write `plan.md` with a numbered, sequential outline.
   - Keep each step **one sentence**.
   - Do **not** split into tasks here.
4. After writing `plan.md`, ask the user whether to proceed to **Skill 2 (Task Planning)**.
   - Example: "plan.md 만들었어. 다음으로 Step들을 task 파일로 쪼개는(Skill 2) 것도 바로 진행할까? (y/n)"

## Output template (plan.md)
```md
# Plan: ##

## User request
- <paste the user request verbatim>

## Steps
1. 요구사항 분석
2. 핵심 기능 정의
3. 아키텍처 설계
4. 기능별 구현
5. 테스트 및 검증
```

## Constraints
- Do not write code.
- Do not create task files.
- Do not invent external dependencies unless the request explicitly requires them.
