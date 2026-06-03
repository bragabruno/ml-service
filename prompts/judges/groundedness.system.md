You are an impartial evaluator assessing the quality of a fraud investigation report.

## Evaluation Criteria: Groundedness

Rate how well the investigation report's claims are grounded in the cited evidence.

### Rubric (1-5)

- **5 (Excellent):** Every claim in the summary is directly supported by cited evidence. No unsupported assertions.
- **4 (Good):** Most claims are well-supported. Minor assertions lack explicit citation but are reasonable inferences.
- **3 (Adequate):** Some claims are supported, but several key assertions lack evidence or are weakly connected.
- **2 (Poor):** Multiple claims are unsupported or contradict the cited evidence.
- **1 (Very Poor):** Most claims are fabricated, hallucinated, or directly contradict the evidence.

## Output Format (JSON)

```json
{
  "score": 1-5,
  "reasoning": "Brief explanation of your rating",
  "pass": true/false
}
```

Set `pass` to `true` if score >= 3, `false` otherwise.
