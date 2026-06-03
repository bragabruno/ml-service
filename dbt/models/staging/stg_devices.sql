with source as (
    select * from {{ source('fraud_platform', 'devices') }}
),

renamed as (
    select
        id as device_id,
        fingerprint as device_fingerprint,
        type as device_type,
        trusted as device_is_trusted,
        cast(first_seen as timestamp) as device_first_seen,
        cast(last_seen as timestamp) as device_last_seen
    from source
)

select * from renamed
