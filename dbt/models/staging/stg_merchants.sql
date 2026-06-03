with source as (
    select * from {{ source('fraud_platform', 'merchants') }}
),

renamed as (
    select
        id as merchant_id,
        name as merchant_name,
        mcc,
        risk_tier as merchant_risk_tier,
        country as merchant_country
    from source
)

select * from renamed
