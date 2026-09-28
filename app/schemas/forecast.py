from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction


class ForecastRequest(BaseModel):
    user_id: str
    historical_transactions: list[Transaction] = Field(min_length=1)
    categories: list[str] = Field(min_length=1)


class ForecastPoint(BaseModel):
    category: str
    month: str
    quarter: str
    amount: float
    user_baseline_amount: float
    hfce_multiplier: float
    hfce_forecast_amount_million_php: float
    explanation: str


class ForecastResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    forecasts: list[ForecastPoint]
    model_version: str
    status: ModuleStatus
    metadata: ApiMetadata
