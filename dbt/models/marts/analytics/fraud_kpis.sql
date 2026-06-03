with labels as (
    select * from {{ ref('stg_fraud_labels') }}
),

cases as (
    select * from {{ ref('stg_fraud_cases') }}
),

risk_scores as (
    select * from {{ ref('stg_risk_scores') }}
),

transactions as (
    select * from {{ ref('stg_transactions') }}
),

label_summary as (
    select
        count(*) as total_labels,
        count(case when label = 'FRAUD' then 1 end) as fraud_labels,
        count(case when label = 'LEGITIMATE' then 1 end) as legitimate_labels
    from labels
),

case_summary as (
    select
        count(*) as total_cases,
        count(case when case_status = 'RESOLVED_FRAUD' then 1 end) as resolved_fraud,
        count(case when case_status = 'RESOLVED_LEGIT' then 1 end) as resolved_legit,
        count(case when case_status in ('OPEN', 'ASSIGNED', 'IN_REVIEW') then 1 end) as open_queue,
        count(case when case_status = 'ESCALATED' then 1 end) as escalated
    from cases
),

score_summary as (
    select
        count(*) as total_scored,
        count(case when decision = 'DECLINE' then 1 end) as declined,
        count(case when decision = 'REVIEW' then 1 end) as reviewed,
        count(case when decision = 'APPROVE' then 1 end) as approved
    from risk_scores
),

amount_summary as (
    select
        coalesce(sum(case when fl.label = 'FRAUD' then t.amount else 0 end), 0) as fraud_loss_amount
    from transactions t
    left join labels fl on t.transaction_id = fl.transaction_id
)

select
    ls.total_labels,
    ls.fraud_labels,
    ls.legitimate_labels,

    cs.total_cases,
    cs.resolved_fraud,
    cs.resolved_legit,
    cs.open_queue,
    cs.escalated,

    ss.total_scored,
    ss.declined,
    ss.reviewed,
    ss.approved,

    case
        when ls.fraud_labels > 0
        then cast(cs.resolved_fraud as double) / ls.fraud_labels
        else 0.0
    end as fraud_detection_rate,

    case
        when ss.total_scored > 0
        then cast(ss.reviewed as double) / ss.total_scored
        else 0.0
    end as review_rate,

    case
        when ls.legitimate_labels > 0 and ss.reviewed > 0
        then cast(ss.declined as double) / nullif(ls.legitimate_labels, 0)
        else 0.0
    end as false_positive_rate,

    am.fraud_loss_amount,

    current_timestamp as computed_at

from label_summary ls
cross join case_summary cs
cross join score_summary ss
cross join amount_summary am
