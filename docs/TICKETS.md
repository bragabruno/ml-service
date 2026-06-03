# ml-service — AI/ML Plane Backlog

**Repo:** `ml-service` (Python, pkg `ml_service`) · sibling to `backend`
**Package root:** `com.bragdev.frauddetection` (platform) · `ml_service` (Python)
**Owner:** BragDev LLC

> **This backlog *continues* the platform backlog in `../../docs/TICKETS.md`** (which ends at
> **FRAUD-161 / EPIC-23**). The AI/ML plane introduced here starts at **EPIC-24 / FRAUD-162**
> so it imports into the **same Linear workspace** with no id collisions. It does **not**
> duplicate the already-specified ML tickets (EPIC-09–13 FastAPI/training/lifecycle/
> integration, EPIC-17 active learning, EPIC-18 drift) — it *builds on* them and references
> them in `Dependencies`. New work = LLM/Agentic AI, Prompt Engineering, AI Evaluation,
> dbt, and Airflow — the capabilities the platform plan lacked.

---

## Summary

| Epic | Title | Tickets | Primary JD skills |
|---|---|---|---|
| **EPIC-24** | ML Service Repository & Developer Experience | FRAUD-162–167 | Python, GitHub, AI infra |
| **EPIC-25** | dbt Analytics, Feature & Training Marts | FRAUD-168–174 | **dbt**, **SQL**, Data Science |
| **EPIC-26** | Airflow Orchestration | FRAUD-175–179 | **Airflow**, ETL, MLOps |
| **EPIC-27** | LLM Foundation & Prompt Engineering | FRAUD-180–186 | **LLMs**, **Prompt Engineering**, AI safety |
| **EPIC-28** | Agentic Fraud Investigation Copilot | FRAUD-187–193 | **Agentic AI**, LLMs |
| **EPIC-29** | AI / LLM Evaluation Framework ⭐ | FRAUD-194–201 | **LLMs & AI Evaluation Frameworks**, AI safety |
| **EPIC-30** | AI Plane CI/CD, Documentation & Demo | FRAUD-202–207 | CI/CD, GitHub, communication |

**New milestones**
- **M8 — AI Plane MVP:** runnable `/predict` + dbt marts + investigation agent + eval suite (offline, mock LLM).
- **M9 — AI Plane Hardened:** CI eval/parity gates green, Airflow DAGs, docs + demo complete.

**Suggested new phase:** *Phase 9 — AI/ML Plane*. All epics below are Phase 9 unless noted.

---

# PHASE 9 — AI/ML PLANE

## EPIC-24 — ML Service Repository & Developer Experience
*Phase 9 · Lead: ML/DE · Depends on: EPIC-01*

The standalone `ml-service` repo: build tooling, local stack, synthetic data, pinned deps —
so everything below is reproducible and offline-runnable.

### FRAUD-162 — Repo scaffolding & Python toolchain
**Type:** Infrastructure · **Epic:** EPIC-24 · **Complexity:** S · **Owner:** ML
**Description:** Create the `ml-service` repo with a `src/` layout (pkg `ml_service`), `pyproject.toml`, and configured `ruff`, `mypy`, `pytest`, and pre-commit.
**Business Value:** A reproducible, lint/type-checked Python foundation mirroring the platform's quality bar.
**Acceptance Criteria:**
- `make test` runs ruff + mypy + pytest from a clean checkout.
- All dependencies pinned; none published <14 days ago; no `latest`.
**Technical Notes:** `uv` (or poetry); naming consistent with `bragabruno`/BragDev.
**Dependencies:** FRAUD-001

### FRAUD-163 — Local stack via docker-compose
**Type:** Infrastructure · **Epic:** EPIC-24 · **Complexity:** S · **Owner:** DO
**Description:** `docker-compose.yml` for Postgres + Redis + MLflow (+ optional Kafka) for local development.
**Business Value:** One-command dependencies so the service runs the same on any machine.
**Acceptance Criteria:**
- `docker compose up` brings dependencies healthy; service connects via env config.
**Technical Notes:** Pin image tags; align ports with the platform's compose conventions.
**Dependencies:** FRAUD-162

### FRAUD-164 — Synthetic data generator
**Type:** Story · **Epic:** EPIC-24 · **Complexity:** M · **Owner:** ML/DE
**Description:** Generate realistic synthetic transactions/devices/merchants/labels with injected fraud patterns (velocity bursts, new-device + foreign-country, chargeback histories), seeding both DuckDB and Postgres.
**Business Value:** Enables a fully offline, reproducible end-to-end demo and tests without real PII.
**Acceptance Criteria:**
- Deterministic given a seed; configurable volume + fraud base-rate; loads both targets.
- Output schema matches the 9 domain tables in `03-domain-model.md`.
**Technical Notes:** Use the platform's enums/keys (UUIDv7, Decision, LabelType).
**Dependencies:** FRAUD-162

### FRAUD-165 — Make/task runner for one-command flows
**Type:** Technical Task · **Epic:** EPIC-24 · **Complexity:** XS · **Owner:** ML
**Description:** `Makefile` targets: `demo`, `train`, `dbt`, `eval`, `test`, `serve`.
**Business Value:** Frictionless dev + a clean live-demo path for the interview.
**Acceptance Criteria:**
- Each target runs its flow end to end with sane defaults.
**Dependencies:** FRAUD-162

### FRAUD-166 — Config & secrets hygiene (12-factor)
**Type:** Technical Task · **Epic:** EPIC-24 · **Complexity:** S · **Owner:** ML
**Description:** `pydantic-settings` config; `.env.example`; `LLM_PROVIDER=mock|anthropic` toggle; DB target + MLflow URI via env.
**Business Value:** Secure, portable configuration; no secrets in the repo.
**Acceptance Criteria:**
- Config loads from env; `.env.example` documents every var; secrets are placeholders only.
**Technical Notes:** Fail-fast on missing required config.
**Dependencies:** FRAUD-162

### FRAUD-167 — CI skeleton (GitHub Actions)
**Type:** Infrastructure · **Epic:** EPIC-24 · **Complexity:** S · **Owner:** DO
**Description:** Baseline workflow running ruff + mypy + pytest on push/PR (dbt, parity, and eval gates added in EPIC-30/29).
**Business Value:** Visible green CI from commit 1; the spine for later quality gates.
**Acceptance Criteria:**
- Workflow runs on PR; failing lint/type/test blocks merge.
**Dependencies:** FRAUD-162

---

## EPIC-25 — dbt Analytics, Feature & Training Marts
*Phase 9 · Lead: DE · Depends on: EPIC-03, EPIC-10, EPIC-24*

The SQL transformation layer (dbt, dual-target DuckDB+Postgres): offline features, the
point-in-time training dataset, and analytics KPIs — all tested. This is the offline half
of the train/serve **parity contract**.

### FRAUD-168 — dbt project & dual-target profiles
**Type:** Story · **Epic:** EPIC-25 · **Complexity:** S · **Owner:** DE
**Description:** `dbt_project.yml` + profiles for `duckdb` (local/demo) and `postgres` (prod parity); shared models compile on both.
**Business Value:** Zero-infra local analytics + production fidelity from one codebase.
**Acceptance Criteria:**
- `dbt debug` passes for both targets; `dbt compile` succeeds on each.
**Technical Notes:** `dbt-duckdb` + `dbt-postgres` adapters; pin versions.
**Dependencies:** FRAUD-162

### FRAUD-169 — Staging models over the 9 domain tables
**Type:** Story · **Epic:** EPIC-25 · **Complexity:** M · **Owner:** DE
**Description:** `stg_*` models + declared sources for transactions, devices, merchants, fraud_labels, risk_scores, fraud_cases, model_versions.
**Business Value:** Clean, typed, documented base layer for all downstream marts.
**Acceptance Criteria:**
- Sources declared with freshness; staging tests (not_null/unique) pass.
**Technical Notes:** Match column names/types from `03-domain-model.md`.
**Dependencies:** FRAUD-168, FRAUD-013

### FRAUD-170 — Feature mart `fct_transaction_features`
**Type:** Story · **Epic:** EPIC-25 · **Complexity:** L · **Owner:** DE
**Description:** Engineer offline features — velocity (5m/24h), recency, device/merchant/geo, account age, failed attempts, chargeback history, merchant_risk_score — mirroring engine FRAUD-051 / ML FRAUD-066.
**Business Value:** Predictive, consistent features; the offline source of truth.
**Acceptance Criteria:**
- Every feature documented + tested; transformations deterministic.
**Technical Notes:** Window logic in reusable macros; must match the online contract exactly.
**Dependencies:** FRAUD-169

### FRAUD-171 — Training dataset model (point-in-time label join)
**Type:** Story · **Epic:** EPIC-25 · **Complexity:** M · **Owner:** DE
**Description:** Join `FraudLabel` ground truth point-in-time; latest-label-wins on supersession; add `sample_weight≈5.0` for missed fraud; no post-event leakage.
**Business Value:** Correct, reproducible supervised training data — the basis of model quality.
**Acceptance Criteria:**
- Point-in-time correct; supersession handled; weight column present; leakage test passes.
**Technical Notes:** Implements the dbt side of FRAUD-069.
**Dependencies:** FRAUD-170, FRAUD-069

### FRAUD-172 — Analytics / KPI marts (self-service)
**Type:** Story · **Epic:** EPIC-25 · **Complexity:** M · **Owner:** DE
**Description:** `fraud_kpis` mart: fraud detection rate, false-positive rate, fraud loss amount, review-queue size, analyst throughput.
**Business Value:** A self-service data product for Product/GTM — directly a JD responsibility.
**Acceptance Criteria:**
- KPIs match hand-computed values on the synthetic set; documented + tested.
**Dependencies:** FRAUD-169

### FRAUD-173 — dbt tests & data-quality gates
**Type:** Technical Task · **Epic:** EPIC-25 · **Complexity:** M · **Owner:** DE
**Description:** Schema tests + custom checks (value ranges, null rates, distribution sanity) that gate the pipeline.
**Business Value:** Stops bad data from silently degrading models (parallels FRAUD-070).
**Acceptance Criteria:**
- `dbt test` runs in CI; violations fail the build with a report.
**Technical Notes:** dbt generic/singular tests; optionally `pandera`/Great Expectations.
**Dependencies:** FRAUD-170

### FRAUD-174 — Offline↔online feature parity test
**Type:** Technical Task · **Epic:** EPIC-25 · **Complexity:** M · **Owner:** DE/ML
**Description:** Assert the dbt feature mart columns equal `features/contract.py` (names, order, types); fail loudly on any drift.
**Business Value:** Eliminates train/serve skew — the platform's #1 cause of silent model degradation.
**Acceptance Criteria:**
- Automated test fails on any schema/order/type mismatch between dbt and the online contract.
**Technical Notes:** Consumes the shared schema artifact from FRAUD-063/FRAUD-068.
**Dependencies:** FRAUD-170, FRAUD-068, FRAUD-063

---

## EPIC-26 — Airflow Orchestration
*Phase 9 · Lead: DE/MLOps · Depends on: EPIC-11, EPIC-12, EPIC-18, EPIC-25*

Author DAGs that orchestrate ETL, training/promotion, drift, and LLM-eval. DAGs are
authored for the architecture story; running a scheduler is optional for the demo.

### FRAUD-175 — Airflow project setup
**Type:** Infrastructure · **Epic:** EPIC-26 · **Complexity:** S · **Owner:** DO
**Description:** Local Airflow (standalone/Astro), `dags/` folder, connections via env.
**Business Value:** Foundation for scheduled, observable data + ML workflows.
**Acceptance Criteria:**
- All DAGs parse with no import errors; `airflow dags list` shows them.
**Technical Notes:** Pin Airflow; keep DAGs import-light (thin wrappers over the package).
**Dependencies:** FRAUD-162

### FRAUD-176 — Feature pipeline DAG
**Type:** Story · **Epic:** EPIC-26 · **Complexity:** S · **Owner:** DE
**Description:** Scheduled `dbt run` + `dbt test` refreshing the feature/training marts.
**Business Value:** Reliable, observable feature refresh — core ETL responsibility.
**Acceptance Criteria:**
- DAG runs dbt build/test; failures alert and stop downstream tasks.
**Dependencies:** FRAUD-175, FRAUD-170

### FRAUD-177 — Training & promotion DAG
**Type:** Story · **Epic:** EPIC-26 · **Complexity:** M · **Owner:** MLOps
**Description:** `dbt build → assemble training set → train → evaluate → gate → MLflow register → emit fraud.model.deployed`.
**Business Value:** Automates the model launch path end to end (supports "AI model launches").
**Acceptance Criteria:**
- A run produces a registered, gated model and a deploy event; failures block promotion.
**Technical Notes:** Reuses gate FRAUD-117 and registry EPIC-12.
**Dependencies:** FRAUD-175, FRAUD-071, FRAUD-078

### FRAUD-178 — Drift monitoring DAG
**Type:** Story · **Epic:** EPIC-26 · **Complexity:** M · **Owner:** MLOps
**Description:** Scheduled PSI/KS drift check; alert + emit `fraud.retraining.requested` on triggers (monthly / ~5000 labels / ~10% recall drop / detected drift, debounced).
**Business Value:** Keeps the model healthy as fraud patterns shift.
**Acceptance Criteria:**
- Drift computed against the training baseline; triggers fire per FRAUD-115 policy.
**Dependencies:** FRAUD-175, FRAUD-113

### FRAUD-179 — LLM evaluation DAG
**Type:** Story · **Epic:** EPIC-26 · **Complexity:** S · **Owner:** MLOps
**Description:** Scheduled run of the eval suite → report → gate, tracking agent/prompt quality over time.
**Business Value:** Continuous AI quality monitoring — "testing and performance improvements."
**Acceptance Criteria:**
- DAG runs the eval runner and publishes a report; regressions alert.
**Dependencies:** FRAUD-175, FRAUD-197

---

## EPIC-27 — LLM Foundation & Prompt Engineering
*Phase 9 · Lead: ML/Prompt · Depends on: EPIC-24*

Provider-agnostic LLM access (mock-default, Claude-optional), a versioned prompt registry,
and safety guardrails.

### FRAUD-180 — Provider-agnostic LLM client (mock + Anthropic)
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** M · **Owner:** ML
**Description:** Define an LLM `Protocol`; implement a deterministic `MockLLM` (default) and an `AnthropicClient` (real Claude); select via `LLM_PROVIDER`; add retry/timeout/token accounting.
**Business Value:** Offline-reproducible evals and one-flag swap to a production LLM — no demo-day network risk.
**Acceptance Criteria:**
- Identical interface for both backends; mock is deterministic; real client gated behind env + key.
**Technical Notes:** Pin the Anthropic SDK; never commit keys (placeholders + env only).
**Dependencies:** FRAUD-166

### FRAUD-181 — Prompt registry & versioning
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** M · **Owner:** Prompt
**Description:** File-based, versioned prompt templates (Jinja2) with metadata (model, temperature, version) and a load/render API + changelog.
**Business Value:** Prompt changes become reviewable, diffable, and pinned to eval results.
**Acceptance Criteria:**
- Templates load by name+version; rendering is tested; metadata captured per version.
**Dependencies:** FRAUD-162

### FRAUD-182 — Investigation prompt template v1
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** S · **Owner:** Prompt
**Description:** Author the system+user template for case investigation, instructing structured output (disposition, rationale, cited evidence).
**Business Value:** The prompt that drives the agent's analyst-facing output.
**Acceptance Criteria:**
- Produces schema-valid output on golden cases under the mock and real LLM.
**Dependencies:** FRAUD-181

### FRAUD-183 — Explanation narrator prompt (SHAP → plain English)
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** S · **Owner:** Prompt
**Description:** Template that turns SHAP/feature-importances + rule hits into an analyst-readable explanation.
**Business Value:** Human-readable "why" for each score; complements FRAUD-084/156.
**Acceptance Criteria:**
- Explanation references only real contributing features (no fabrication).
**Dependencies:** FRAUD-181, FRAUD-084

### FRAUD-184 — LLM guardrails: PII redaction
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** M · **Owner:** ML/Safety
**Description:** Redact PII from any payload before it reaches an LLM; field allowlist; reversible token map for rehydration.
**Business Value:** Data minimization and compliance — an AI-safety differentiator.
**Acceptance Criteria:**
- No raw PII leaves the service boundary; redaction unit-tested on synthetic PII.
**Dependencies:** FRAUD-180

### FRAUD-185 — Prompt-injection & output-validation defenses
**Type:** Story · **Epic:** EPIC-27 · **Complexity:** M · **Owner:** ML/Safety
**Description:** Defend against injected instructions in case data; validate/repair structured output; explicit refuse/uncertainty path.
**Business Value:** Robust, trustworthy agent behavior on adversarial inputs.
**Acceptance Criteria:**
- Known injection probes do not alter the disposition; invalid output is rejected, not guessed.
**Dependencies:** FRAUD-180, FRAUD-182

### FRAUD-186 — Structured output schemas (Pydantic)
**Type:** Technical Task · **Epic:** EPIC-27 · **Complexity:** S · **Owner:** ML
**Description:** Pydantic models for `InvestigationReport`, `Explanation`, `JudgeVerdict` with strict parsing + repair.
**Business Value:** Type-safe LLM I/O across the agent and eval framework.
**Acceptance Criteria:**
- Malformed model output is repaired or rejected; schemas reused by agent + eval.
**Dependencies:** FRAUD-180

---

## EPIC-28 — Agentic Fraud Investigation Copilot
*Phase 9 · Lead: ML · Depends on: EPIC-09, EPIC-14, EPIC-27*

A tool-using LLM agent that investigates flagged cases, gathers evidence, and recommends a
disposition for analysts — slotting into the existing HITL loop.

### FRAUD-187 — Investigation tool suite
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** M · **Owner:** ML
**Description:** Implement agent tools: `get_transaction_features`, `get_velocity`, `get_similar_cases`, `get_shap_explanation`, `get_rule_hits`, `get_customer_history` — typed and safely executed.
**Business Value:** Grounds the agent in real platform data instead of guesswork.
**Acceptance Criteria:**
- Each tool has a typed signature, validation, and a unit test against synthetic data.
**Dependencies:** FRAUD-170, FRAUD-059

### FRAUD-188 — Agent orchestration loop
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** L · **Owner:** ML
**Description:** Tool-use/reasoning loop (hand-rolled or LangGraph) with bounded steps and full tracing of tool calls; deterministic under `MockLLM`.
**Business Value:** The core Agentic AI capability.
**Acceptance Criteria:**
- Agent selects tools, iterates within a step budget, and terminates with a report; reproducible under mock.
**Technical Notes:** Prefer a thin hand-rolled loop before adding a heavy agent framework.
**Dependencies:** FRAUD-180, FRAUD-187

### FRAUD-189 — InvestigationReport synthesis
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** M · **Owner:** ML
**Description:** Synthesize a summary + cited evidence + recommended `Decision` (APPROVE/REVIEW/DECLINE) + confidence + reason codes.
**Business Value:** Turns reasoning into an actionable, auditable analyst artifact.
**Acceptance Criteria:**
- Output is schema-valid; every claim cites a tool result; decision uses platform enums.
**Dependencies:** FRAUD-188, FRAUD-186

### FRAUD-190 — POST /investigate endpoint
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** M · **Owner:** ML
**Description:** FastAPI route: case/transaction id → `InvestigationReport`; auth + observability.
**Business Value:** Exposes the copilot to the console and case workflow.
**Acceptance Criteria:**
- Returns a report within budget; 422 on bad input; metrics emitted.
**Dependencies:** FRAUD-189, FRAUD-058

### FRAUD-191 — Agent ↔ case-management integration (HITL)
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** M · **Owner:** ML/BE
**Description:** Attach the report to a `FraudCase`; consume `fraud.review.required` to auto-draft an investigation; analyst accept/override feeds the feedback loop.
**Business Value:** Routes agent output into the existing HITL loop (EPIC-14/16) instead of a silo.
**Acceptance Criteria:**
- New review cases get an auto-draft; analyst overrides are captured as labels/feedback.
**Dependencies:** FRAUD-190, FRAUD-090

### FRAUD-192 — Agent tracing & token/cost accounting
**Type:** Technical Task · **Epic:** EPIC-28 · **Complexity:** S · **Owner:** ML
**Description:** Structured per-investigation traces (tool calls, tokens, latency, cost) for eval + observability.
**Business Value:** Cost control and the raw material for evaluation.
**Acceptance Criteria:**
- Each run emits a trace record; traces feed the eval runner (FRAUD-197).
**Dependencies:** FRAUD-188

### FRAUD-193 — Active-learning hook: uncertain → agent triage
**Type:** Story · **Epic:** EPIC-28 · **Complexity:** S · **Owner:** ML
**Description:** Route mid-band uncertain predictions (FRAUD-109/110) to the agent for triage before a human.
**Business Value:** Cuts review cost and speeds the active-learning loop.
**Acceptance Criteria:**
- Uncertain cases get an agent triage note; clear cases skip it.
**Dependencies:** FRAUD-188, FRAUD-109

---

## EPIC-29 — AI / LLM Evaluation Framework ⭐
*Phase 9 · Lead: ML/Prompt · Depends on: EPIC-28*

Golden datasets, metrics, LLM-as-judge, reporting, and a CI regression gate for the agent
and prompts. **The headline JD capability — "Design AI/LLM evaluation frameworks."**

### FRAUD-194 — Golden evaluation dataset
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** M · **Owner:** ML
**Description:** Curate a versioned JSONL of labeled cases: inputs + expected disposition + reference facts + difficulty tags.
**Business Value:** Ground truth for measuring agent/prompt quality and catching regressions.
**Acceptance Criteria:**
- Dataset versioned + documented; covers easy/hard/adversarial slices.
**Dependencies:** FRAUD-164

### FRAUD-195 — Deterministic eval metrics
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** M · **Owner:** ML
**Description:** Implement disposition accuracy, citation correctness, tool-selection correctness, and latency/cost.
**Business Value:** Objective, cheap, reproducible signal that runs in CI without an LLM.
**Acceptance Criteria:**
- Metrics computed over the golden set; deterministic under `MockLLM`.
**Dependencies:** FRAUD-194, FRAUD-189

### FRAUD-196 — LLM-as-judge (groundedness / hallucination / faithfulness)
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** L · **Owner:** ML/Prompt
**Description:** Judge prompts + rubrics scoring groundedness, hallucination rate, faithfulness, and safety; the judge model/prompt is distinct from the generator to limit bias.
**Business Value:** Scalable qualitative evaluation — the rarest, most-demanded JD skill.
**Acceptance Criteria:**
- Judge scores correlate with human spot-checks on a sample; rubric documented.
**Technical Notes:** Calibrate judge; record judge model + prompt version with every score.
**Dependencies:** FRAUD-194, FRAUD-181

### FRAUD-197 — Eval runner
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** M · **Owner:** ML
**Description:** Run a given agent/prompt version over the golden set; collect deterministic metrics + judge verdicts; persist results.
**Business Value:** One command to evaluate any prompt/model change.
**Acceptance Criteria:**
- Produces a results artifact keyed by prompt + model version.
**Dependencies:** FRAUD-195, FRAUD-196

### FRAUD-198 — Eval report (prompt-version comparison)
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** S · **Owner:** ML
**Description:** Render a markdown/HTML report comparing prompt/model versions with a regression table and trends.
**Business Value:** Makes AI quality legible to non-ML stakeholders (Product/GTM/Research).
**Acceptance Criteria:**
- Report shows per-version metrics, deltas, and pass/fail vs gate.
**Dependencies:** FRAUD-197

### FRAUD-199 — Baselines & regression detection
**Type:** Technical Task · **Epic:** EPIC-29 · **Complexity:** S · **Owner:** ML
**Description:** Store metric baselines; detect regressions vs baseline per prompt/model version.
**Business Value:** Turns "looks fine" into an enforceable contract.
**Acceptance Criteria:**
- Baselines versioned; a regression is flagged with the offending metric + delta.
**Dependencies:** FRAUD-197

### FRAUD-200 — CI eval gate
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** M · **Owner:** MLOps
**Description:** Fail CI when groundedness < threshold, hallucination > threshold, or disposition accuracy regresses vs baseline.
**Business Value:** Prevents shipping degraded prompts/agents — operationalizes "AI model launches, testing."
**Acceptance Criteria:**
- Gate runs in GitHub Actions; a deliberately-bad prompt fails the build.
**Dependencies:** FRAUD-199, FRAUD-167

### FRAUD-201 — Safety & red-team eval suite
**Type:** Story · **Epic:** EPIC-29 · **Complexity:** M · **Owner:** ML/Safety
**Description:** Adversarial cases (prompt injection, PII-leak attempts, jailbreaks) asserting the guardrails hold.
**Business Value:** Demonstrates AI safety/alignment — a JD nice-to-have most candidates skip.
**Acceptance Criteria:**
- Red-team suite runs in CI; guardrail bypass fails the build.
**Dependencies:** FRAUD-196, FRAUD-184, FRAUD-185

---

## EPIC-30 — AI Plane CI/CD, Documentation & Demo
*Phase 9 · Lead: ML/DO · Depends on: EPIC-24…EPIC-29*

Make the whole plane defensible (gated CI), legible (docs/diagrams), and presentable
(one-command demo).

### FRAUD-202 — Full CI pipeline (lint/type/test/dbt/eval/parity)
**Type:** Story · **Epic:** EPIC-30 · **Complexity:** M · **Owner:** DO
**Description:** Extend CI to run ruff + mypy + pytest + `dbt build/test` + the parity test + the eval gate.
**Business Value:** Every quality contract enforced on every PR.
**Acceptance Criteria:**
- All gates run on PR; any failure blocks merge; status badges in README.
**Dependencies:** FRAUD-167, FRAUD-174, FRAUD-200

### FRAUD-203 — ARCHITECTURE.md + diagrams
**Type:** Story · **Epic:** EPIC-30 · **Complexity:** S · **Owner:** ML
**Description:** Document how `ml-service` fits the platform; mermaid for the extended component view, agent tool-use sequence, eval flow, dbt DAG, and Airflow DAGs.
**Business Value:** The "walk me through your system" artifact for the interview.
**Acceptance Criteria:**
- Diagrams render on GitHub; each new lane (dbt/Airflow/agent/eval) is shown.
**Dependencies:** —

### FRAUD-204 — EVALUATION.md (methodology)
**Type:** Story · **Epic:** EPIC-30 · **Complexity:** S · **Owner:** ML
**Description:** Metric definitions, judge rubrics, gating policy, and dataset governance.
**Business Value:** Proves depth on the rarest JD skill; reusable as a talking script.
**Acceptance Criteria:**
- Each metric defined with intent, formula/rubric, and threshold rationale.
**Dependencies:** FRAUD-196

### FRAUD-205 — PROMPTS.md + ADRs + MODEL_CARD.md
**Type:** Technical Task · **Epic:** EPIC-30 · **Complexity:** S · **Owner:** ML
**Description:** Prompt rationale/changelog; ADRs (LLM-provider abstraction, dbt dual-target, mock-first eval); a model card for the fraud model.
**Business Value:** Shows engineering judgment and responsible-AI documentation.
**Acceptance Criteria:**
- ≥3 ADRs; prompt changelog present; model card covers data, metrics, limitations.
**Dependencies:** —

### FRAUD-206 — One-command demo script
**Type:** Story · **Epic:** EPIC-30 · **Complexity:** S · **Owner:** ML
**Description:** `scripts/demo.sh`: generate → `dbt build` → train → serve → investigate a sample case → run eval → open the report.
**Business Value:** A reliable, repeatable live demo for the interview.
**Acceptance Criteria:**
- Runs end to end offline (mock LLM) on a clean checkout.
**Dependencies:** FRAUD-197, FRAUD-190

### FRAUD-207 — README + JD skill map
**Type:** Story · **Epic:** EPIC-30 · **Complexity:** S · **Owner:** ML
**Description:** Centerpiece README: architecture overview, quickstart, the JD-skill→file map, and diagrams/screenshots.
**Business Value:** What a recruiter/interviewer reads first; converts "does he have X?" into "here's X."
**Acceptance Criteria:**
- README links every JD skill to a concrete file/dir and a runnable command.
**Dependencies:** FRAUD-203

---

# Importing into Linear

Same mapping as `../../docs/TICKETS.md` §19 — these tickets are formatted identically:

- **Phase 9** → `phase:9` label (the platform uses `phase:1…8`).
- **Epic (EPIC-24…30)** → a project **milestone** *and* an `Epic`-labelled **parent issue** inside the existing **"Fraud Detection"** project (mirrors how EPIC-01…23 are modeled — not a separate project per epic).
- **Ticket (FRAUD-162…207)** → a **sub-issue** of its epic's parent issue, assigned to that epic's milestone.

**Field mapping** (matches the live BragDev workspace taxonomy)
- Title → the `FRAUD-### — Title` line (keep the id).
- Description / Business Value / Acceptance Criteria / Technical Notes → issue description (paste as-is; already Markdown).
- Complexity → **Estimate** points: `XS=1, S=2, M=3, L=5, XL=8`.
- Type → Story→`type:story`, Technical Task→`type:tech-task`, Infrastructure→`type:infra`, Spike→`type:spike`.
- Domain → `domain:ml-service` on every ticket, plus the epic's area: `area:dbt` (E25), `area:airflow` (E26), `area:llm` (E27), `area:agent` (E28), `area:eval` (E29); E24/E30 use `domain:infra`.
- Team → `team:ml` (ML/DE/Prompt/Safety), `team:devops` (DO/MLOps), `team:backend` (BE).
- `Dependencies` → add as **"blocked by"** relations (bulk/CSV import does **not** create relations).

**CSV note:** import issues first (with labels + estimates + project), then add the "blocked by"
relations from each ticket's `Dependencies` via the UI or Linear API. Cross-repo dependencies
on FRAUD-001…161 assume the platform backlog is imported into the same workspace.

> Want this as a flat CSV (one row per ticket, columns: Title, Description, Status, Priority,
> Estimate, Labels, Project, Cycle, Assignee) ready for Linear's importer? Say the word.

---

*AI/ML plane backlog — 7 epics (EPIC-24 → EPIC-30), 46 tickets (FRAUD-162 → FRAUD-207), Phase 9. Extends the platform backlog (FRAUD-001 → FRAUD-161).*
