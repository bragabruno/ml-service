select *
from {{ ref('training_dataset') }}
where labeled_at < created_at
