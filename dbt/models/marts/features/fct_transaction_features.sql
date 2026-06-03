with txn as (
    select * from {{ ref('stg_transactions') }}
),

users as (
    select * from {{ ref('stg_users') }}
),

devices as (
    select * from {{ ref('stg_devices') }}
),

merchants as (
    select * from {{ ref('stg_merchants') }}
),

country_risk as (
    select * from {{ ref('country_risk') }}
),

mcc_risk as (
    select * from {{ ref('mcc_codes') }}
),

velocity as (
    select
        t.transaction_id,
        t.user_id,
        t.created_at,
        t.amount,

        {{ velocity_count('t.user_id', 't.created_at', 5) }} as velocity_5m,
        {{ velocity_count('t.user_id', 't.created_at', 60) }} as velocity_1h,
        {{ velocity_count('t.user_id', 't.created_at', 1440) }} as velocity_24h,

        {{ velocity_sum('t.user_id', 't.created_at', 't.amount', 5) }} as amount_5m,
        {{ velocity_sum('t.user_id', 't.created_at', 't.amount', 60) }} as amount_1h,
        {{ velocity_sum('t.user_id', 't.created_at', 't.amount', 1440) }} as amount_24h,

        count(distinct t.merchant_id) over (
            partition by t.user_id
            order by t.created_at
            range between interval '1440' minute preceding and current row
        ) as distinct_merchants_24h,

        count(distinct t.country) over (
            partition by t.user_id
            order by t.created_at
            range between interval '1440' minute preceding and current row
        ) as distinct_countries_24h,

        lag(t.created_at) over (partition by t.user_id order by t.created_at) as prev_txn_ts,

        count(*) over (
            partition by t.user_id
            order by t.created_at
            range between interval '43200' minute preceding and current row
        ) as user_txn_count_30d,

        avg(t.amount) over (
            partition by t.user_id
            order by t.created_at
            range between interval '43200' minute preceding and current row
        ) as avg_amount_30d

    from txn t
),

device_velocity as (
    select
        t.transaction_id,
        count(*) over (
            partition by t.device_id
            order by t.created_at
            range between interval '1440' minute preceding and current row
        ) - 1 as device_txn_count_24h
    from txn t
),

merchant_stats as (
    select
        t.merchant_id,
        t.created_at as ref_ts,
        count(*) over (
            partition by t.merchant_id
            order by t.created_at
            range between interval '1440' minute preceding and current row
        ) - 1 as merchant_txn_volume_24h
    from txn t
),

merchant_fraud_rates as (
    select
        t.merchant_id,
        t.created_at as ref_ts,
        avg(case when fl.label = 'FRAUD' then 1.0 else 0.0 end) over (
            partition by t.merchant_id
            order by t.created_at
            range between interval '43200' minute preceding and current row
        ) as merchant_fraud_rate_30d
    from txn t
    left join {{ ref('stg_fraud_labels') }} fl on t.transaction_id = fl.transaction_id
),

user_chargebacks as (
    select
        t.user_id,
        t.created_at as ref_ts,
        count(case when fl.label = 'FRAUD' then 1 end) over (
            partition by t.user_id
            order by t.created_at
            range between interval '129600' minute preceding and current row
        ) as chargeback_count_90d
    from txn t
    left join {{ ref('stg_fraud_labels') }} fl on t.transaction_id = fl.transaction_id
)

select
    t.transaction_id,
    t.user_id,
    t.merchant_id,
    t.device_id,
    t.amount,
    ln(1.0 + t.amount) as amount_log,
    t.amount as amount_usd,

    extract(hour from t.created_at) as txn_hour,
    extract(dow from t.created_at) as txn_day_of_week,
    case when extract(dow from t.created_at) in (0, 6) then true else false end as is_weekend,
    case when extract(hour from t.created_at) between 0 and 4 then true else false end as is_night,

    greatest(0, v.velocity_5m) as velocity_5m,
    greatest(0, v.velocity_1h) as velocity_1h,
    greatest(0, v.velocity_24h) as velocity_24h,
    greatest(0.0, v.amount_5m) as amount_5m,
    greatest(0.0, v.amount_1h) as amount_1h,
    greatest(0.0, v.amount_24h) as amount_24h,

    extract(day from t.created_at - u.created_at) as user_account_age_days,
    v.user_txn_count_30d,
    0 as user_failed_attempts_24h,

    extract(day from t.created_at - d.device_first_seen) as device_age_days,
    d.device_is_trusted,
    case when extract(day from t.created_at - d.device_first_seen) <= 7 then true else false end as device_is_new,
    dv.device_txn_count_24h,

    {{ risk_tier_to_score('m.merchant_risk_tier') }} as merchant_risk_score,
    ms.merchant_txn_volume_24h,
    coalesce(mfr.merchant_fraud_rate_30d, 0.0) as merchant_fraud_rate_30d,

    coalesce(cr.risk_score, 0.5) as country_risk,
    case when t.country != u.home_country then true else false end as is_foreign_country,
    false as ip_country_mismatch,

    coalesce(mr.risk_score, 0.3) as mcc_risk,

    extract(day from t.created_at - v.prev_txn_ts) as days_since_last_txn,
    coalesce(v.avg_amount_30d, t.amount) as avg_amount_30d,
    t.amount / nullif(v.avg_amount_30d, 0) as amount_to_avg_ratio,

    v.distinct_merchants_24h,
    v.distinct_countries_24h,

    coalesce(uc.chargeback_count_90d, 0) as chargeback_count_90d,
    case when coalesce(uc.chargeback_count_90d, 0) > 0 then true else false end as has_chargeback_history,

    t.created_at

from txn t
left join users u on t.user_id = u.user_id
left join devices d on t.device_id = d.device_id
left join merchants m on t.merchant_id = m.merchant_id
left join velocity v on t.transaction_id = v.transaction_id
left join device_velocity dv on t.transaction_id = dv.transaction_id
left join merchant_stats ms on t.merchant_id = ms.merchant_id and t.created_at = ms.ref_ts
left join merchant_fraud_rates mfr on t.merchant_id = mfr.merchant_id and t.created_at = mfr.ref_ts
left join user_chargebacks uc on t.user_id = uc.user_id and t.created_at = uc.ref_ts
left join country_risk cr on t.country = cr.country_code
left join mcc_risk mr on m.mcc = mr.mcc_code
