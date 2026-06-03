with source as (
    select * from {{ source('fraud_platform', 'audit_events') }}
),

renamed as (
    select
        id as audit_event_id,
        actor,
        action,
        target_type,
        target_id,
        before as before_state,
        after as after_state,
        correlation_id,
        cast(created_at as timestamp) as created_at
    from source
)

select * from renamed
