Evaluate the following fraud investigation report for groundedness.

## Investigation Report

```json
{{ report | tojson }}
```

## Cited Evidence (Tool Results)

{% for ev in evidence %}
**{{ ev.tool }}:** {{ ev.finding }}
{% endfor %}

Rate the groundedness of this report.
