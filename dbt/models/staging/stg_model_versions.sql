with source as (
    select * from {{ source('fraud_platform', 'model_versions') }}
),

renamed as (
    select
        id as model_version_id,
        version,
        mlflow_run_id,
        metrics,
        status as model_status,
        cast(deployed_at as timestamp) as deployed_at,
        cast(created_at as timestamp) as created_at
    from source
)

select * from renamed
