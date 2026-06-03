from __future__ import annotations

from enum import Enum
from typing import Optional

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
    trace_id: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


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


class PredictRequest(BaseModel):
    transaction_id: str
    amount: float
    new_device: bool = False
    failed_attempts: int = 0
    country: str = "US"
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    user_id: Optional[str] = None


class PredictResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    model_version: str
    contributing_factors: list[str]


class InvestigateRequest(BaseModel):
    transaction_id: str
    case_id: Optional[str] = None
