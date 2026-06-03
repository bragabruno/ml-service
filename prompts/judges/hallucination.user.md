Evaluate the following fraud investigation report for hallucination.

## Investigation Report

```json
{{ report | tojson }}
```

## Available Evidence (Tool Results)

{% for ev in evidence %}
**{{ ev.tool }}:** {{ ev.finding }}
{% endfor %}

Rate whether this report contains hallucinated (fabricated) information.
