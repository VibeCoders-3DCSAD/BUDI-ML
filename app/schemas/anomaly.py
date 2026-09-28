from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction


class DetectionType(StrEnum):
    ANOMALOUS = "ANOMALOUS"
    OVERSPENDING = "OVERSPENDING"
    BOTH = "BOTH"


class BudgetAllocationInput(BaseModel):
    category_id: str
    budget_amount: float = Field(ge=0)


class AnomalyTransaction(Transaction):
    transaction_id: str = Field(min_length=1)
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        """Normalize ISO dates to UTC timestamps for deterministic ordering."""
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("date must be an ISO-8601 date or timestamp") from exc
        if "T" in value and parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone offset")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat()

    @field_validator("category")
    @classmethod
    def validate_canonical_category(cls, value: str) -> str:
        canonical = value.strip().lower()
        if canonical != value:
            raise ValueError("category must be a canonical lowercase identifier")
        return canonical


class AnomalyRequest(BaseModel):
    transactions: list[AnomalyTransaction] = Field(min_length=1)
    detection_type: DetectionType = DetectionType.ANOMALOUS
    budget_allocations: list[BudgetAllocationInput] | None = None

    @model_validator(mode="after")
    def require_unique_transaction_ids(self) -> AnomalyRequest:
        identifiers = [transaction.transaction_id for transaction in self.transactions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("transactions must have unique transaction_id values")
        return self


class DecisionState(StrEnum):
    ANOMALOUS = "ANOMALOUS"
    NORMAL = "NORMAL"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    INSUFFICIENT_VARIABILITY = "INSUFFICIENT_VARIABILITY"


class AnomalyDecision(BaseModel):
    transaction_id: str
    category: str
    usable_history_count: int = Field(ge=0)
    state: DecisionState
    lower_bound: float | None = Field(default=None, ge=0)
    upper_bound: float | None = Field(default=None, ge=0)
    historical_median: float | None = Field(default=None, gt=0)
    usual_spending: float | None = Field(default=None, gt=0)
    severity: float | None = Field(default=None, ge=0)
    reason: str


class OverspendingTransaction(BaseModel):
    transaction_id: str
    budget_excess: float
    category: str
    period: str | None = None


class AnomalyResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    decisions: list[AnomalyDecision]
    overspending_transactions: list[OverspendingTransaction]
    ruleset_version: str
    status: ModuleStatus
    metadata: ApiMetadata
