# Plan — `ml-service`: the AI/ML plane of the Fraud Prevention Platform

## Context

`ml-service/` is an empty sibling directory to `backend/` in the FraudPreventionSystem
polyrepo (`backend` → `github.com/bragabruno/backend.git`). The platform is already
meticulously designed across 23 epics / 161 tickets, and the C4 docs **already spec
`ml-service` as a Python/FastAPI service**: `/health · /predict · /batch-predict ·
/model/version · /retrain`, a Pydantic **feature contract**, in-memory serving with
**atomic hot-swap**, **XGBoost** training, **MLflow** registry, **PSI/KS drift**, and
explainability — wired by REST (`fraud-engine` → `/predict`, circuit-breaker fallback)
and Kafka (consumes `fraud.confirmed`/`fraud.falsepositive`/`fraud.model.deployed`;
publishes `fraud.retraining.requested`). Stores: Postgres (9-table schema) + Redis.

**Why this matters now:** there is an interview tomorrow (2026-06-03, 17:00 BRT) for an
**AI/ML Data Engineer / Prompt Engineer** role. The JD's required skills are *Agentic AI,
ML, Data Science, SQL, Python, Prompt Engineering, dbt, Airflow, GitHub, LLMs & AI
Evaluation Frameworks*. **None of the LLM / Agentic / Prompt / AI-Eval / dbt / Airflow
skills appear anywhere in the existing 161-ticket plan** — even "Advanced Intelligence"
(EPIC-23) is graph/SHAP/Flink. So the empty `ml-service` is the ideal canvas: build the
ML inference service the architecture already calls for (proving ML/MLOps/Python/SQL)
**and** layer a coherent LLM + data-engineering plane that demonstrates every remaining
JD keyword — without contradicting the existing design.

**Outcome:** a complete, well-documented, runnable-where-it-counts `ml-service` repo on
GitHub that (a) plugs into the platform's existing contracts and (b) maps 1:1 to the JD.

**Decisions (confirmed with user):**
- **Scope = architecture breadth** — all lanes present as a coherent, documented system; optimize for "walk me through it" + an impressive GitHub. Guard a runnable core first.
- **LLM = provider-agnostic; deterministic MOCK is the default** (offline, free, reproducible evals — no dead-wifi risk in the interview); **real Anthropic Claude** swappable via one env var.
- **dbt = dual-target** — DuckDB (local/demo, zero-infra) + Postgres (prod-parity), shared models.
- **Repo = standalone sibling git repo** `ml-service`, remote under `github.com/bragabruno` (matches the polyrepo + "sibling repo to the backend").

---

## JD skill → component map (this is the "fit", and the interview script)

| JD required skill | Where it lives in `ml-service` | Ties to platform |
|---|---|---|
| **Agentic AI** | `src/ml_service/agent/` — tool-using **Fraud Investigation Copilot** | HITL / case-mgmt (EPIC-14/16) |
| **LLMs & AI Evaluation Frameworks** ⭐ | `src/ml_service/eval/` — golden sets, LLM-as-judge, groundedness/hallucination/disposition metrics, CI gate | "Support AI model launches, testing" |
| **Prompt Engineering** | `prompts/` versioned registry + `agent` + `explain/narrator.py` | Explainability (FRAUD-084/156) |
| **dbt** + **SQL** | `dbt/` — staging → feature mart → training mart → KPI marts | Feature parity (FRAUD-068), training data (FRAUD-069) |
| **Airflow** | `airflow/dags/` — ETL, training, drift, eval DAGs | EPIC-11/12/18 orchestration |
| **Machine Learning** + **Data Science** | `src/ml_service/training/`, `drift/`, `notebooks/` | EPIC-10/11/18 |
| **Python** | entire service | EPIC-09 |
| **GitHub / CI/CD** | `.github/workflows/ci.yml` with an **eval-gate** | EPIC-22 |
| Nice-to-have: **NLP / AI safety** | `agent/guardrails.py` (PII redaction, prompt-injection defense), eval safety metric | — |
| Nice-to-have: **experimentation** | eval baselines/regression + champion/challenger hooks | FRAUD-083/161 |

⭐ = highest-signal, rarest skill for this JD — over-invest here.

---

## Target repo structure (`ml-service/`)

```
ml-service/
├── README.md                      # centerpiece: architecture, JD-map, quickstart, diagrams
├── pyproject.toml                 # uv/poetry; PINNED versions (no `latest`)
├── .env.example                   # LLM_PROVIDER=mock|anthropic, DB targets, MLflow URI
├── docker-compose.yml             # postgres + redis + mlflow (+ optional kafka)
├── Makefile                       # make demo | train | dbt | eval | test | serve
├── .github/workflows/ci.yml       # ruff · mypy · pytest · dbt build+test · EVAL GATE · contract test
├── LICENSE                        # BragDev LLC
│
├── src/ml_service/
│   ├── app/                       # FastAPI (EPIC-09)
│   │   ├── main.py                # app factory + lifespan (model load/warmup → READY)
│   │   ├── config.py              # pydantic-settings, 12-factor
│   │   ├── observability.py       # structlog + prometheus (/metrics, score histogram) FRAUD-064
│   │   └── api/routes/            # health, predict (/predict,/batch-predict), model (/model/version,/retrain), explain, investigate
│   ├── schemas/                   # Pydantic: features (CONTRACT, versioned) · predict · investigation  FRAUD-063
│   ├── features/
│   │   ├── contract.py            # single source of truth: feature names/types/order → exports shared JSON  FRAUD-068
│   │   ├── transforms.py          # velocity/recency/encodings, deterministic  FRAUD-066 (mirror engine FRAUD-051)
│   │   ├── online_store.py        # Redis  FRAUD-067
│   │   ├── offline_store.py       # reads dbt training mart
│   │   └── parity.py              # train/serve parity check  FRAUD-068/086
│   ├── serving/                   # model_registry (MLflow) · serving_model (in-memory + ATOMIC hot-swap) FRAUD-061/087
│   ├── training/                  # dataset · train (XGBoost + LogReg baseline) · evaluate (PR-AUC,ROC-AUC,recall,FP,cost) · tune (optuna) · imbalance (scale_pos_weight) · registry · gate (FRAUD-117)  EPIC-11
│   ├── drift/                     # feature_drift (PSI/KS) · prediction_drift · report  EPIC-18
│   ├── explain/                   # importances (XGB gain) · shap_explain · narrator.py (LLM → plain-English)  FRAUD-084/156
│   ├── agent/                     # ⭐ AGENTIC AI
│   │   ├── llm/                   # base.py (Protocol) · mock_client.py (deterministic) · anthropic_client.py · factory.py
│   │   ├── tools.py               # get_features · get_velocity · get_similar_cases · get_shap · get_rule_hits · get_customer_history
│   │   ├── investigation_agent.py # tool-use loop → InvestigationReport(summary, cited evidence, disposition, confidence)
│   │   └── guardrails.py          # PII redaction · prompt-injection defense · output validation · uncertainty/refusal
│   └── events/                    # Kafka: consumer (confirmed/FP→retrain set; model.deployed→hot-swap) · producer (retraining.requested) · topics (7 names)
│
├── prompts/                       # ⭐ PROMPT ENGINEERING
│   ├── investigation/v1.md, v2.md ; explanation/v1.md ; triage/v1.md ; judges/*.md
│   ├── registry.py                # versioned load + Jinja2 render + metadata (model/temp/version)
│   └── README.md                  # design rationale + changelog
│
├── eval/                          # ⭐ AI EVALUATION FRAMEWORK (the star)
│   ├── datasets/golden_cases.jsonl# labeled cases + expected disposition + reference facts
│   ├── metrics/                   # groundedness · hallucination · disposition_accuracy · citation · faithfulness · safety
│   ├── judges/llm_judge.py        # LLM-as-judge w/ rubric prompts
│   ├── runner.py                  # run agent/prompt over golden set → metrics
│   ├── report.py                  # markdown/HTML report; per-prompt-version comparison
│   ├── gate.py                    # CI gate: fail on groundedness<X / hallucination>Y / disposition-accuracy regression
│   └── baselines/                 # stored metric baselines for regression detection
│
├── dbt/                           # ⭐ dbt + SQL (dual-target)
│   ├── dbt_project.yml ; profiles.yml   # targets: duckdb (local) + postgres (prod)
│   ├── models/staging/            # stg_transactions, _devices, _merchants, _fraud_labels, _risk_scores, _fraud_cases, _model_versions (sources = the 9 tables)
│   ├── models/marts/features/fct_transaction_features.sql   # velocity 5m/24h, recency, device/merchant/geo, acct age, failed attempts, chargebacks, merchant_risk
│   ├── models/marts/training/training_dataset.sql           # point-in-time label join · latest-wins · missed-fraud sample_weight≈5.0  FRAUD-069
│   ├── models/marts/analytics/fraud_kpis.sql                # detection rate, FP rate, fraud loss, queue size, analyst throughput (self-service data product)
│   ├── tests/ + schema.yml        # not_null/unique/relationships/accepted_values + parity test
│   ├── seeds/                     # mcc codes, country risk
│   └── macros/                    # feature-window macros
│
├── airflow/dags/                  # ⭐ AIRFLOW
│   ├── feature_pipeline_dag.py    # dbt run + dbt test (feature mart refresh)
│   ├── training_pipeline_dag.py   # dbt build → assemble set → train → evaluate → GATE → MLflow register → emit fraud.model.deployed
│   ├── drift_monitor_dag.py       # scheduled PSI/KS → alert + emit fraud.retraining.requested (triggers: monthly/5k labels/recall-drop/drift) FRAUD-115
│   └── llm_eval_dag.py            # scheduled eval suite → report → gate
│
├── data/generate_synthetic.py     # realistic synthetic txns/devices/merchants/labels w/ injected fraud patterns → seeds duckdb (+postgres)
├── scripts/demo.sh                # one-command E2E: generate→dbt build→train→serve→investigate→eval→open report
├── tests/                         # pytest: transforms, contract/PARITY, schemas, API, agent(mock), eval-metrics
├── notebooks/                     # EDA + model card + eval analysis (Data Science showcase)
└── docs/
    ├── ARCHITECTURE.md            # how ml-service fits the platform (extends C4 + 07-ml-and-data-pipelines lanes)
    ├── EVALUATION.md              # ⭐ eval methodology: metric defs, judge rubrics, gating policy (interview gold)
    ├── PROMPTS.md ; MODEL_CARD.md ; JD_SKILL_MAP.md
    ├── ADR/                       # 001-llm-provider-abstraction, 002-dbt-dual-target, 003-mock-first-eval
    └── diagrams/                  # mermaid: extended component · agent tool-use sequence · eval flow · dbt DAG · airflow DAGs
```

---

## What it reuses / must stay consistent with (the "fit")

- **Naming/owner:** repo `ml-service`, Python pkg `ml_service`, BragDev LLC, remote `github.com/bragabruno/ml-service`.
- **`/predict` contract** (ROADMAP + FRAUD-059/063): feature payload in → `fraudProbability`, `riskLevel`, model `version`, contributing factors out. The Java `fraud-engine` ML client (FRAUD-085) is the consumer.
- **Feature contract** = single Python source (`features/contract.py`) exported as a JSON artifact, consumed by the Pydantic schema (online) **and** asserted against the dbt training mart (offline) → parity test (FRAUD-068/086). This boundary is the strongest senior narrative.
- **7 Kafka topics** (exact names): `transactions.created`, `fraud.scored`, `fraud.review.required`, `fraud.confirmed`, `fraud.falsepositive`, `fraud.model.deployed`, `fraud.retraining.requested`.
- **9 domain tables** as dbt sources (from `03-domain-model.md`): users, transactions, devices, merchants, risk_scores, fraud_cases, fraud_labels, model_versions, audit_events.
- **Enums/conventions:** `Decision{APPROVE,REVIEW,DECLINE}`, `LabelType{FRAUD,LEGITIMATE}`, `ModelStatus{REGISTERED,APPROVED,DEPLOYED,ROLLED_BACK,ARCHIVED}`, reason codes, MLflow stages; **PR-AUC emphasis** (class imbalance), cost-weighted metric, missed-fraud `sample_weight≈5.0`, active-learning mid-band routing.
- **Deps:** PIN every version; none published <14 days ago; reuse the ROADMAP's declared stack (FastAPI, XGBoost, scikit-learn, pandas, numpy, MLflow, SHAP). Heaviest/optional: `apache-airflow`, `langgraph` — prefer authoring Airflow DAGs + a hand-rolled agent loop over heavy runtime deps if time is short.

---

## Build order (breadth-first, but a coherent repo at every checkpoint)

Push to GitHub at the end of Phase 0 so the repo looks real immediately; commit per phase.

0. **Skeleton + README + push** — git init, `pyproject.toml` (pinned), full dir tree with `__init__`/docstrings/READMEs + clear TODOs, `.gitignore`, LICENSE, `.env.example`, Makefile, `JD_SKILL_MAP.md`. *Repo reads as a complete system from commit 1.*
1. **Spine** — `features/contract.py` (+ JSON export), Pydantic predict schemas, FastAPI `/health` + `/predict` + `/model/version`, observability. *This is what literally plugs into the backend.*
2. **Data plane** — `data/generate_synthetic.py` → DuckDB (+Postgres optional); dbt staging + feature mart + training mart + KPI marts + dbt tests. *Runs offline end-to-end.*
3. **Training + MLflow** — `train.py` (XGBoost + LogReg baseline) on the training mart, `evaluate.py` (PR-AUC/ROC-AUC/recall/FP/cost), MLflow log+register, `gate.py`. *API serves a real artifact.*
4. **LLM plane** — provider-agnostic LLM client (mock default + Anthropic), `prompts/` registry + v1 templates, `investigation_agent` + tools + `/investigate`, `guardrails`.
5. **⭐ Eval framework** — golden dataset, metrics, LLM-as-judge, `runner` + `report` + `gate` + baselines.
6. **Orchestration + CI + docs** — 4 Airflow DAGs, GitHub Actions CI (ruff/mypy/pytest/dbt/eval-gate/parity), `docs/` (ARCHITECTURE, EVALUATION, PROMPTS, ADRs, mermaid diagrams), `demo.sh`, MODEL_CARD.

**If time runs short:** Phases 0–2 + README + diagrams already present a credible data-engineering story; 4–5 are the JD differentiators — protect them over polish on 3/6. Airflow DAGs + Kafka `events/` may remain authored-but-not-run (clearly labeled) — that's acceptable for breadth.

**Stretch (only if ahead):** champion/challenger shadow eval (FRAUD-083/161); SHAP narrator polish; a tiny Streamlit "self-service" KPI dashboard over `fraud_kpis`; optional PR to the parent repo extending `docs/diagrams/07-ml-and-data-pipelines.md` with the new LLM/agent/eval/dbt/Airflow lanes (ties the whole story together).

---

## Risks / watch-outs

- **Airflow runtime is heavy** — author DAGs for the architecture story; don't burn the deadline standing up a scheduler. `airflow standalone`/Astro only if Phases 0–5 are done.
- **Don't let breadth become hollow** — every directory needs a real docstring/README and a coherent stub, not empty files. Reviewers open folders.
- **Mock LLM must be deterministic** so evals and CI are reproducible; real Claude is opt-in via `LLM_PROVIDER=anthropic` (+ key). Never commit keys (placeholders + env only).
- **Parity test is the credibility anchor** — if `features/contract.py` and the dbt `training_dataset` columns ever diverge, the test must fail loudly (fail-fast).

---

## Verification (how we'll prove it works)

- `make test` → pytest + **ruff** + **mypy** green (transforms, schemas, **parity/contract**, API, agent-with-mock, eval-metrics).
- `make dbt` → `dbt build && dbt test` green on **DuckDB**; same models compile against the Postgres target.
- `make train` → produces an MLflow run + registered model; `evaluate` report shows PR-AUC/recall/FP.
- API smoke: `curl /health`, `POST /predict` (ROADMAP sample payload), `GET /model/version`, `POST /investigate` (returns a structured `InvestigationReport`).
- `make eval` → eval report (markdown/HTML) with groundedness/hallucination/disposition-accuracy; **`eval/gate.py` exits non-zero on regression** (proven by tweaking a threshold).
- `make demo` (`scripts/demo.sh`) → full chain: generate → dbt build → train → serve → investigate a sample case → run eval → open report.
- **CI**: pushing to GitHub runs the same gates in `.github/workflows/ci.yml`, including the eval gate — the visible proof of "AI evaluation framework + CI/CD" for the interviewer.
