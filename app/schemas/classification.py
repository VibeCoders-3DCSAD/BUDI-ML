from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ApiMetadata, ModuleStatus


class ClassificationDataStatus(StrEnum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


class CreditCardCycle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed_on: date
    revolving_balance: float = Field(ge=0)


class CreditCardAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)
    cycles: list[CreditCardCycle] = Field(min_length=1)


class ClassificationV2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    liquid_emergency_savings: float | None = Field(default=None, ge=0)
    average_monthly_recurring_essential_expenses: float | None = Field(default=None, ge=0)
    monthly_disposable_income: float | None = Field(default=None, ge=0)
    required_monthly_debt_payments: float | None = Field(default=None, ge=0)
    monthly_basic_living_costs: float | None = Field(default=None, ge=0)
    credit_card_accounts: list[CreditCardAccount] | None = None


class RuleAssessment(BaseModel):
    status: ClassificationDataStatus
    label: str | None = None
    value: float | int | None = None
    threshold: float | int
    inputs: dict[str, float | int]


class CreditCardAssessment(RuleAssessment):
    account_id: str


class ClassificationV2Result(BaseModel):
    emergency_savings: RuleAssessment
    debt_burden: RuleAssessment
    financial_margin: RuleAssessment
    credit_card_behavior: list[CreditCardAssessment]


class ClassificationV2Response(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    classification: ClassificationV2Result
    status: ModuleStatus
    metadata: ApiMetadata
