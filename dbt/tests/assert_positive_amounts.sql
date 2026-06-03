select *
from {{ ref('fct_transaction_features') }}
where amount < 0
