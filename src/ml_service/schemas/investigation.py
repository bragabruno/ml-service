from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class LabelType(str, Enum):
    FRAUD = "FRAUD"
    LEGITIMATE = "LEGITIMATE"


class ModelStatus(str, Enum):
    REGISTERED = "REGISTERED"
    APPROVED = "APPROVED"
    DEPLOYED = "DEPLOYED"
    ROLLED_BACK = "ROLLED_BACK"
    ARCHIVED = "ARCHIVED"


class Evidence(BaseModel):
    tool: str
    finding: str


class InvestigationReport(BaseModel):
    transaction_id: str
    disposition: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[Evidence]
    reason_codes: list[str]
    trace_id: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    latency_ms: float | None = None
    safety_flags: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    transaction_id: str
    model_version: str
    fraud_probability: float
    risk_level: str
    explanation: str
    top_features: list[dict[str, float]]


class JudgeVerdict(BaseModel):
    metric: str
    score: int = Field(ge=1, le=5)
    reasoning: str
    passed: bool


class AgentTriage(BaseModel):
    disposition: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[Evidence]
    model: str | None = None


class PredictResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    model_version: str
    contributing_factors: list[str]
    agent_triage: AgentTriage | None = None


class AnalystFeedback(BaseModel):
    transaction_id: str
    case_id: str | None = None
    original_disposition: Decision
    analyst_decision: Decision
    feedback_type: str = "override"
    notes: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    recorded: bool


class PredictRequest(BaseModel):
    transaction_id: str
    amount: float
    new_device: bool = False
    failed_attempts: int = 0
    country: str = "US"
    currency: str = "USD"
    merchant_id: str | None = None
    device_id: str | None = None
    user_id: str | None = None


class InvestigateRequest(BaseModel):
    transaction_id: str
    case_id: str | None = None
    amount: float = 0.0
    currency: str = "USD"
    country: str = "US"
    new_device: bool = False
    failed_attempts: int = 0
    user_id: str | None = None
    merchant_id: str | None = None
    device_id: str | None = None
    untrusted_notes: str | None = None
