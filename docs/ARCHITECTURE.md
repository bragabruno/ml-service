# Architecture — `ml-service`

## System Context

`ml-service` is the AI/ML plane of the Fraud Prevention Platform. It sits alongside the Java `backend` (Spring Boot) and provides:

1. **ML inference** — real-time fraud scoring via `/predict`
2. **Agentic investigation** — LLM-powered case analysis via `/investigate`
3. **Offline analytics** — dbt feature/training/KPI marts
4. **Model training** — XGBoost + LogReg with MLflow registry
5. **AI evaluation** — golden datasets, LLM-as-judge, CI gates

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
        API[FastAPI<br/>/predict /investigate /health]
        SERVING[Serving Model<br/>atomic hot-swap]
        TRANSFORMS[Feature Transforms<br/>34 features]
        AGENT[Investigation Agent<br/>tool-use loop]
        LLM[LLM Client<br/>Mock + Anthropic]
        EVAL[Eval Framework<br/>golden sets + judges]
        TRAINING[Training Pipeline<br/>XGBoost + MLflow]
        DBT[dbt<br/>DuckDB + Postgres]
        AIRFLOW[Airflow DAGs<br/>4 pipelines]
    end

    BE -->|REST /predict| API
    BE -->|Kafka events| KAFKA
    KAFKA -->|fraud.confirmed| TRAINING
    API --> TRANSFORMS --> SERVING
    API --> AGENT --> LLM
    TRAINING --> SERVING
    DBT --> PG
    TRANSFORMS --> REDIS
    EVAL --> AGENT
    AIRFLOW --> DBT
    AIRFLOW --> TRAINING
    AIRFLOW --> EVAL
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
    loop max_steps=8
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
| `drift_monitor` | weekly | PSI/KS drift → alert → retraining trigger |
| `llm_eval` | weekly | eval runner → gate → report |

## Data Flow

1. **Online**: Transaction → backend → `/predict` → transforms → model → score → Kafka `fraud.scored`
2. **Offline**: Raw tables → dbt staging → feature mart → training dataset → XGBoost → MLflow → hot-swap
3. **Investigation**: Case flagged → `/investigate` → agent gathers evidence → LLM synthesizes report → HITL queue
