with features as (
    select * from {{ ref('fct_transaction_features') }}
),

latest_labels as (
    select
        transaction_id,
        label,
        label_confidence,
        labeled_at,
        row_number() over (
            partition by transaction_id
            order by labeled_at desc
        ) as rn
    from {{ ref('stg_fraud_labels') }}
),

labels as (
    select
        transaction_id,
        label,
        label_confidence,
        labeled_at
    from latest_labels
    where rn = 1
)

select
    f.transaction_id,

    f.amount,
    f.amount_log,
    f.amount_usd,
    f.txn_hour,
    f.txn_day_of_week,
    f.is_weekend,
    f.is_night,
    f.velocity_5m,
    f.velocity_1h,
    f.velocity_24h,
    f.amount_5m,
    f.amount_1h,
    f.amount_24h,
    f.user_account_age_days,
    f.user_txn_count_30d,
    f.user_failed_attempts_24h,
    f.device_age_days,
    f.device_is_trusted,
    f.device_is_new,
    f.device_txn_count_24h,
    f.merchant_risk_score,
    f.merchant_txn_volume_24h,
    f.merchant_fraud_rate_30d,
    f.country_risk,
    f.is_foreign_country,
    f.ip_country_mismatch,
    f.mcc_risk,
    f.days_since_last_txn,
    f.avg_amount_30d,
    f.amount_to_avg_ratio,
    f.distinct_merchants_24h,
    f.distinct_countries_24h,
    f.chargeback_count_90d,
    f.has_chargeback_history,

    f.created_at,

    case when l.label = 'FRAUD' then 1 else 0 end as is_fraud,
    l.label as label_type,
    l.label_confidence,
    l.labeled_at,

    case
        when l.label = 'FRAUD' then {{ var('sample_weight_missed_fraud', 5.0) }}
        else 1.0
    end as sample_weight

from features f
inner join labels l on f.transaction_id = l.transaction_id
where l.labeled_at >= f.created_at
