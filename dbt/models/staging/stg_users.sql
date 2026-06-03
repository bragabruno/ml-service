with source as (
    select * from {{ source('fraud_platform', 'users') }}
),

renamed as (
    select
        id as user_id,
        username,
        email,
        role,
        status as user_status,
        home_country,
        cast(created_at as timestamp) as created_at,
        cast(updated_at as timestamp) as updated_at,
        version
    from source
)

select * from renamed
