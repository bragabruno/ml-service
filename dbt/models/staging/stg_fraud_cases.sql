with source as (
    select * from {{ source('fraud_platform', 'fraud_cases') }}
),

renamed as (
    select
        id as case_id,
        transaction_id,
        risk_score_id,
        assignee_id,
        status as case_status,
        severity,
        cast(opened_at as timestamp) as opened_at,
        cast(sla_due_at as timestamp) as sla_due_at,
        cast(resolved_at as timestamp) as resolved_at
    from source
)

select * from renamed
