{% macro velocity_count(user_id, txn_ts, window_minutes) %}
    count(*) over (
        partition by {{ user_id }}
        order by {{ txn_ts }}
        range between interval '{{ window_minutes }}' minute preceding and current row
    ) - 1
{% endmacro %}

{% macro velocity_sum(user_id, txn_ts, amount_col, window_minutes) %}
    sum({{ amount_col }}) over (
        partition by {{ user_id }}
        order by {{ txn_ts }}
        range between interval '{{ window_minutes }}' minute preceding and current row
    ) - {{ amount_col }}
{% endmacro %}

{% macro days_between(ts_col, ref_ts) %}
    extract(day from {{ ref_ts }} - {{ ts_col }})
{% endmacro %}

{% macro risk_tier_to_score(risk_tier_col) %}
    case {{ risk_tier_col }}
        when 'LOW' then 0.1
        when 'MEDIUM' then 0.5
        when 'HIGH' then 0.9
        else 0.5
    end
{% endmacro %}
