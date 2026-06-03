# ADR-001: LLM Provider Abstraction

**Status:** Accepted  
**Date:** 2026-06-02

## Context

The investigation agent needs LLM access for case analysis. Production requires a real model (Anthropic Claude), but development and CI need deterministic, offline-reproducible results.

## Decision

Implement a provider-agnostic `LLMClient` protocol with two backends:
- **MockLLM** (default): Deterministic, seed-based responses. No network required.
- **AnthropicClient**: Real Claude API. Gated behind `LLM_PROVIDER=anthropic` + API key.

Selection via environment variable `LLM_PROVIDER=mock|anthropic`.

## Consequences

- Evals are reproducible in CI without API keys or network
- One-flag swap to production LLM for demos
- No vendor lock-in — new providers implement the same Protocol
- Mock responses are structured to exercise the same code paths as real responses
