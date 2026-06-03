with source as (
    select * from {{ source('fraud_platform', 'fraud_labels') }}
),

renamed as (
    select
        id as label_id,
        transaction_id,
        case_id,
        analyst_id,
        label,
        cast(confidence as double) as label_confidence,
        reason as label_reason,
        cast(labeled_at as timestamp) as labeled_at
    from source
)

select * from renamed
