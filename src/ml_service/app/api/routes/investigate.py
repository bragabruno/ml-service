from __future__ import annotations

from fastapi import APIRouter

from ml_service.agent.investigation_agent import investigate
from ml_service.agent.llm.factory import get_llm_client
from ml_service.schemas.investigation import InvestigateRequest, InvestigationReport

router = APIRouter()


@router.post("/investigate", response_model=InvestigationReport)
async def investigate_transaction(request: InvestigateRequest) -> InvestigationReport:
    llm = get_llm_client()
    report, trace = investigate(request, llm)
    return report
