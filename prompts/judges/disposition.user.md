Evaluate whether this fraud investigation reached the correct disposition.

## Investigation Report

```json
{{ report | tojson }}
```

## Evidence Gathered

{% for ev in evidence %}
**{{ ev.tool }}:** {{ ev.finding }}
{% endfor %}

## Expected Disposition (if available)

{{ expected_disposition | default("Not provided — evaluate based on evidence alone") }}

Rate the disposition accuracy.
