from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ApiMetadata, ModuleStatus

CLASSIFIER_MODES = ("STANDARD", "QUESTIONNAIRE")


class ClassificationMode(StrEnum):
    STANDARD = "STANDARD"
    QUESTIONNAIRE = "QUESTIONNAIRE"


class IncomePattern(StrEnum):
    NO_CURRENT_INCOME = "no_current_income"
    PREDICTABLE = "predictable_income"
    VARIABLE = "variable_income"


class ObligationLoad(StrEnum):
    NO_INCOME_WITH_OBLIGATIONS = "no_income_with_obligations"
    NO_INCOME_WITHOUT_OBLIGATIONS = "no_income_without_obligations"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmergencyRunway(StrEnum):
    LESS_THAN_ONE_MONTH = "less_than_1_month"
    ONE_TO_THREE_MONTHS = "1_to_3_months"
    THREE_TO_SIX_MONTHS = "3_to_6_months"
    SIX_PLUS_MONTHS = "6_plus_months"


class QuestionnaireAnswers(BaseModel):
    """Validated answers to the user-facing financial profile questions."""

    income_pattern: IncomePattern
    obligation_load: ObligationLoad
    emergency_runway: EmergencyRunway

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_answers(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        income_variability = value.get("income_variability")
        emergency_runway = value.get("emergency_runway")
        income_pattern = (
            {
                "stable": "predictable_income",
                "variable": "variable_income",
                "irregular": "variable_income",
            }.get(income_variability)
            if isinstance(income_variability, str)
            else value.get("income_pattern")
        )
        obligation_load = value.get("obligation_level", value.get("obligation_load"))
        normalized_runway = (
            {
                "low": "less_than_1_month",
                "medium": "1_to_3_months",
                "3": "3_to_6_months",
                "high": "6_plus_months",
            }.get(emergency_runway, emergency_runway)
            if isinstance(emergency_runway, str)
            else emergency_runway
        )
        return {
            "income_pattern": income_pattern,
            "obligation_load": obligation_load,
            "emergency_runway": normalized_runway,
        }


class PFPClassifyRequest(BaseModel):
    user_id: str
    classification_mode: ClassificationMode
    payload: dict
    questionnaire_answers: QuestionnaireAnswers | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_questionnaire(self) -> PFPClassifyRequest:
        if self.classification_mode == ClassificationMode.QUESTIONNAIRE:
            answers = self.payload.get("questionnaire_answers")
            self.questionnaire_answers = QuestionnaireAnswers.model_validate(answers)
        return self


class PFPClassification(BaseModel):
    prediction: str
    financial_stability_score: float = Field(ge=0, le=1)
    financial_weight_score: float = Field(ge=0, le=1)
    financial_tolerance_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    status: ModuleStatus
    tier_used: int
    model_name: str


class PFPClassifyResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    classification: PFPClassification
    metadata: ApiMetadata
