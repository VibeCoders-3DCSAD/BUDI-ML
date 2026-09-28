from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import TrustedHistoryContext, get_trusted_history_context
from app.schemas.anomaly import (
    AnomalyRequest,
    AnomalyResponse,
    DetectionType,
)
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import anomaly_service

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])
logger = logging.getLogger(__name__)


def _run(user_id: str, request: AnomalyRequest) -> AnomalyResponse:
    start = time.perf_counter()
    decisions = []
    overspending = []
    if request.detection_type in (DetectionType.ANOMALOUS, DetectionType.BOTH):
        decisions = anomaly_service.detect(request.transactions)
    if request.detection_type in (DetectionType.OVERSPENDING, DetectionType.BOTH):
        if not request.budget_allocations:
            raise HTTPException(
                status_code=422,
                detail="budget_allocations required when detection_type is OVERSPENDING or BOTH",
            )
        overspending = anomaly_service.overspending_detect(
            request.transactions, [budget.model_dump() for budget in request.budget_allocations]
        )
    return AnomalyResponse(
        response_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        user_id=user_id,
        decisions=decisions,
        overspending_transactions=overspending,
        ruleset_version=anomaly_service.RULESET_VERSION,
        status=ModuleStatus.SUCCESS,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=anomaly_service.RULESET_VERSION,
            strategy_used="personal_subcategory_iqr",
        ),
    )


class BatchRequest(BaseModel):
    requests: list[AnomalyRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[AnomalyResponse]


@router.post("/detect", response_model=AnomalyResponse)
async def detect(
    request: AnomalyRequest,
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> AnomalyResponse:
    logger.info(
        "anomaly request received user_id=%s transactions=%d detection_type=%s",
        history.user_id,
        len(request.transactions),
        request.detection_type,
    )
    return _run(history.user_id, request)


@router.post("/detect/batch", response_model=BatchResponse)
async def detect_batch(
    batch: BatchRequest,
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> BatchResponse:
    logger.info(
        "anomaly batch request received user_id=%s requests=%d transactions=%d",
        history.user_id,
        len(batch.requests),
        sum(len(request.transactions) for request in batch.requests),
    )
    return BatchResponse(results=[_run(history.user_id, req) for req in batch.requests])
