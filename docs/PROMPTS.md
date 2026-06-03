# Prompt Engineering

All prompts are **versioned, file-based templates** rendered with Jinja2 (`StrictUndefined`,
so a missing variable fails loudly rather than silently producing a malformed prompt). The
registry lives in [`prompts/registry.py`](../prompts/registry.py); load with
`load_prompt(name, version, **vars)`.

## Layout

```
prompts/
├── investigation/      v1.system.md · v1.user.md · v1.meta.json   # the agent's reasoning prompt
├── explanation/        v1.system.md · v1.user.md · v1.meta.json   # SHAP → plain-English narrator
└── judges/             groundedness · hallucination · disposition (.system.md/.user.md)
```

Each `*.meta.json` pins the intended `model`, `temperature`, and a description, so a prompt
change is a reviewable diff tied to its eval result (see [`EVALUATION.md`](EVALUATION.md)).

## Design rules (enforced in the system prompts)

- **Grounding:** every claim in a report's `summary` must cite a tool result; never fabricate
  evidence. Measured by the groundedness + hallucination judges.
- **Structured output:** the investigation prompt requires strict JSON
  (`disposition`/`confidence`/`summary`/`evidence`/`reason_codes`); the agent repairs/validates
  it and falls back to `REVIEW` on parse failure.
- **Untrusted input is never interpolated into a prompt.** Upstream free-text (analyst notes,
  case payloads) is screened by `agent/guardrails.py` for prompt-injection/PII and dropped
  before the LLM call — the structured `CASE_SIGNALS` line is the only case context passed.
- **Plain language** in explanations: no "SHAP/gain/log-odds" jargon for analyst-facing text.

## LLM-as-judge

The judge prompts (`prompts/judges/`) score a report 1–5 on a rubric and return
`{score, reasoning, pass}` (pass = score ≥ 3). The judge model/prompt is kept **distinct from
the generator** to limit self-preference bias.

## Changelog

| Prompt | Version | Notes |
|--------|---------|-------|
| investigation | v1 | Initial structured-disposition prompt with mandatory evidence citation. |
| explanation | v1 | SHAP/importance + rule-hit narrator for analysts. |
| judges/* | v1 | Groundedness, hallucination, and disposition rubrics (1–5, pass ≥ 3). |
