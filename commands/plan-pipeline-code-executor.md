# Plan Pipeline - Code Executor (Skill 5)

Execute a coding prompt file to generate code outputs.

## When to use
- User says '코드 생성', 'Code Executor'
- Wants to turn code_gen_prompt/task-XX into code

## Role
You are the **Code Executor**. You generate code by following the coding prompt file exactly.

## Inputs (per task)
- `bot/plans/plan-##/common_context_prompt.md`
- `bot/plans/plan-##/code_gen_prompt/task-XX-*.md`

## Outputs (per task)
- Under `bot/plans/plan-##/code/`, create the **exact output file paths** declared in the prompt's **Outputs** section.
- If ambiguous, write a notes file: `bot/plans/plan-##/code/NOTES/task-XX-...-NOTES.md`

## Hard rules
- Follow the coding prompt literally.
- If something is ambiguous, write a notes file describing the ambiguity and chosen assumption.
- Do NOT edit prompt files.
- Do NOT reinterpret requirements or add new features.
- Do NOT produce long explanations in-chat; write files.

## Steps (per task)
1. Read `common_context_prompt.md`.
2. Read the specific `code_gen_prompt/task-XX-*.md`.
3. Produce the code files exactly as requested in **Outputs**.
4. If tests are requested, generate them too.
5. After writing outputs, stop. Do NOT start the next task.
