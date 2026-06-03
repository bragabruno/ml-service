select
    transaction_id,
    count(*) as dup_count
from {{ ref('fct_transaction_features') }}
group by transaction_id
having count(*) > 1
