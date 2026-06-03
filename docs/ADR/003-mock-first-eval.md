# ADR-003: Mock-First Evaluation

**Status:** Accepted  
**Date:** 2026-06-02

## Context

The AI evaluation framework must run in CI on every PR. Real LLM calls are expensive, slow, and non-deterministic — making CI gates unreliable.

## Decision

The eval framework defaults to `MockLLM` for all deterministic metrics. LLM-as-judge also runs under mock by default, with real Claude available as an opt-in for deeper analysis.

The golden dataset is designed so mock responses produce meaningful metric scores — the mock is not random but follows deterministic patterns that exercise the evaluation logic.

## Consequences

- CI eval gate runs in seconds, not minutes
- Results are bit-for-bit reproducible across runs
- Real LLM evaluation is available for pre-release validation
- The mock must be kept in sync with the agent's expected tool-use patterns
