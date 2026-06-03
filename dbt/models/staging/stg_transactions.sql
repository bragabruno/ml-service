with source as (
    select * from {{ source('fraud_platform', 'transactions') }}
),

renamed as (
    select
        id as transaction_id,
        user_id,
        merchant_id,
        device_id,
        cast(amount as double) as amount,
        currency,
        ip_address,
        country,
        status as transaction_status,
        idempotency_key,
        cast(created_at as timestamp) as created_at
    from source
)

select * from renamed
