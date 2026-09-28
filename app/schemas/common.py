from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ModuleStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    FALLBACK = "FALLBACK"


class ApiMetadata(BaseModel):
    processing_time_ms: float
    model_version: str
    strategy_used: str | None = None


class Transaction(BaseModel):
    transaction_id: str | None = None
    date: str
    amount: float = Field(gt=0)
    category: str
    transaction_type: str = Field(pattern="^(income|expense)$")
    description: str | None = None
