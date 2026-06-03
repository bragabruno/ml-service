# Architecture — `ml-service`

## System Context

`ml-service` is the AI/ML plane of the Fraud Prevention Platform. It sits alongside the Java `backend` (Spring Boot) and provides:

1. **ML inference** — real-time fraud scoring via `/predict` (uncertain MEDIUM-band scores get an inline agent triage)
2. **Agentic investigation** — LLM-powered case analysis via `/investigate`
3. **HITL case integration** — `fraud.review.required` → auto-drafted report attached to the case; analyst overrides become labels via `/feedback`
4. **Offline analytics** — dbt feature/training/KPI marts
5. **Model training** — XGBoost + LogReg with MLflow registry
6. **Continuous learning** — drift/label-volume/schedule triggers emit `fraud.retraining.requested` (debounced + audited) → orchestrated train/evaluate/register → promotion guardrail (eligibility only; deploy stays human)
7. **AI evaluation** — golden datasets, LLM-as-judge, CI gates

## Component View

```mermaid
graph TB
    subgraph "Fraud Prevention Platform"
        BE[backend<br/>Java Spring Boot]
        KAFKA[Kafka<br/>7 topics]
        PG[(PostgreSQL<br/>9 tables)]
        REDIS[(Redis<br/>feature cache)]
    end

    subgraph "ml-service (Python/FastAPI)"
        API[FastAPI<br/>/predict /investigate /feedback<br/>/events/review-required /retrain /model/*]
        SERVING[Serving Model<br/>atomic hot-swap]
        TRANSFORMS[Feature Transforms<br/>34 features]
        AGENT[Investigation Agent<br/>tool-use loop]
        LLM[LLM Client<br/>Mock + Anthropic]
        CASE[Case Client<br/>Mock + HTTP]
        EVAL[Eval Framework<br/>golden sets + judges]
        TRAINING[Training Pipeline<br/>XGBoost + MLflow]
        ORCH[Retraining Orchestrator<br/>+ promotion guardrail]
        EVENTS[Event Producer/Consumer<br/>Mock + Kafka]
        DBT[dbt<br/>DuckDB + Postgres]
        AIRFLOW[Airflow DAGs<br/>4 pipelines]
    end

    BE -->|REST /predict| API
    BE -->|Kafka events| KAFKA
    KAFKA -->|fraud.review.required| EVENTS
    KAFKA -->|fraud.retraining.requested| EVENTS
    KAFKA -->|fraud.confirmed| TRAINING
    EVENTS -->|auto-draft| AGENT
    EVENTS -->|orchestrate| ORCH
    AGENT --> CASE
    ORCH --> TRAINING
    ORCH -->|fraud.retraining.requested| EVENTS
    API --> TRANSFORMS --> SERVING
    API --> AGENT --> LLM
    TRAINING --> SERVING
    DBT --> PG
    TRANSFORMS --> REDIS
    EVAL --> AGENT
    AIRFLOW --> DBT
    AIRFLOW --> TRAINING
    AIRFLOW --> EVAL
    AIRFLOW -->|drift trigger| EVENTS
```

## Agent Tool-Use Sequence

```mermaid
sequenceDiagram
    participant C as Analyst/Console
    participant A as /investigate
    participant AG as Agent Loop
    participant T as Tools
    participant L as LLM (Mock/Claude)

    C->>A: POST /investigate {txn_id}
    A->>AG: investigate(txn_id)
    loop max_steps=6
        AG->>L: system + user prompt + tool results
        L-->>AG: tool_call or final report
        alt tool_call
            AG->>T: execute tool
            T-->>AG: result
        else final
            AG-->>A: InvestigationReport
        end
    end
    A-->>C: {disposition, evidence, confidence}
```

## dbt DAG

```mermaid
graph LR
    SRC[sources<br/>9 tables] --> STG[staging<br/>stg_*]
    STG --> FCT[fct_transaction_features<br/>34 features]
    STG --> KPI[fraud_kpis]
    FCT --> TRN[training_dataset<br/>point-in-time labels]
    SEEDS[seeds<br/>country_risk, mcc_codes] --> FCT
```

## Airflow DAGs

| DAG | Schedule | Purpose |
|---|---|---|
| `feature_pipeline` | daily | dbt seed + run + test + parity |
| `training_pipeline` | weekly | dbt build → train → evaluate → gate → register |
| `drift_monitor` | weekly | PSI/KS drift → emit `fraud.retraining.requested` (debounced + audited) |
| `llm_eval` | weekly | eval runner → gate → report |

## Continuous-Learning Loop

Every stage is automated except the one that changes production — `deploy` stays a human action,
so upstream automation can recommend but never silently swap the serving model.

```mermaid
graph LR
    DRIFT[drift_monitor<br/>PSI/KS] -->|drift detected| REQ
    LABELS[label volume] -->|threshold| REQ
    SCHED[schedule] --> REQ
    REQ[request_retraining<br/>debounce + audit] -->|fraud.retraining.requested| CONS[retraining consumer]
    CONS --> ORCH[run_retraining<br/>train → eval → gate → register candidate]
    ORCH --> PROMO[evaluate_promotion<br/>vs incumbent PR-AUC/cost]
    PROMO -->|eligible + human approval| DEPLOY[POST /model/deploy]
```

## HITL Auto-Draft

```mermaid
graph LR
    RR[fraud.review.required] -->|HTTP or Kafka| H[handle_review_required]
    H --> INV[investigate → report]
    INV --> ATTACH[Case Client.attach_report]
    FB[/feedback override/] -->|DECLINE→FRAUD, APPROVE→LEGITIMATE| LBL[record_label]
```

## Data Flow

1. **Online**: Transaction → backend → `/predict` → transforms → model → score → Kafka `fraud.scored` (MEDIUM-band scores also get an inline agent triage note)
2. **Offline**: Raw tables → dbt staging → feature mart → training dataset → XGBoost → MLflow → hot-swap
3. **Investigation**: Case flagged → `/investigate` → agent gathers evidence → LLM synthesizes report → HITL queue
4. **HITL auto-draft**: `fraud.review.required` → `handle_review_required` → agent report attached to the case; analyst override via `/feedback` → derived `FRAUD`/`LEGITIMATE` label
5. **Continuous learning (closed loop)**: drift/label-volume/schedule → debounced+audited `fraud.retraining.requested` → orchestrated train/evaluate/register candidate → promotion guardrail flags eligibility → human approves `POST /model/deploy`
