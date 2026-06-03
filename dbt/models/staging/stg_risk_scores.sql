with source as (
    select * from {{ source('fraud_platform', 'risk_scores') }}
),

renamed as (
    select
        id as risk_score_id,
        transaction_id,
        model_version_id,
        cast(ml_score as double) as ml_score,
        cast(rules_score as double) as rules_score,
        cast(aggregate_score as double) as aggregate_score,
        decision,
        degraded_mode,
        reason_codes,
        cast(created_at as timestamp) as created_at
    from source
)

select * from renamed
