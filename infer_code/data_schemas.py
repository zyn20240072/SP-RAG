from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# 1. Basic components
# ==============================================================================


class Asset(BaseModel):
    """Retrieved text or image asset."""

    model_config = ConfigDict(extra="ignore")

    asset_id: str = Field(default="unknown")
    short_id: Optional[str] = Field(
        default=None,
        description="Step-prefixed short ID such as S1-A1.",
    )
    type: Literal["text", "image"] = "text"
    content: str = ""
    source_queries: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# 2. Validation output
# ==============================================================================


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    asset_id: str = Field(default="unknown")
    type: Literal["text", "image"] = Field(default="text")
    reasoning: str = Field(default="", description="Optional validator rationale.")
    relevance_score: int = Field(
        default=1,
        ge=1,
        le=5,
        description="High-recall relevance score on a 1-5 scale.",
    )
    search_suggestion: Literal[
        "search_image",
        "search_text",
        "search_both",
        "none",
    ] = Field(
        default="none",
        description="Modality suggestion when the current asset is partial or unclear.",
    )

    @field_validator("search_suggestion", mode="before")
    @classmethod
    def normalize_suggestion(cls, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.lower().strip()
            if "both" in normalized:
                return "search_both"
            if "image" in normalized:
                return "search_image"
            if "text" in normalized:
                return "search_text"
        return "none"


# ==============================================================================
# 3. Cognitive state and actions
# ==============================================================================


class ConflictResolution(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conflict_desc: str = Field(default="", description="Description of the conflict.")
    winning_source: Literal["IMAGE", "TEXT", "NEWER_SOURCE", "NONE"] = Field(
        default="NONE"
    )


class MissingInformation(BaseModel):
    """One explicit unresolved information need."""

    model_config = ConfigDict(extra="ignore")

    description: str = Field(
        default="",
        description="Exact missing answer-bearing entity, value, attribute, or relation.",
    )
    suggested_modality: Literal["text", "image", "both"] = Field(default="text")

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("suggested_modality", mode="before")
    @classmethod
    def normalize_modality(cls, value: Any) -> str:
        normalized = str(value or "text").strip().lower()
        if "both" in normalized or "multi" in normalized:
            return "both"
        if "image" in normalized or "visual" in normalized:
            return "image"
        return "text"


class Belief(BaseModel):
    """Provenance-linked compact epistemic state."""

    model_config = ConfigDict(extra="ignore")

    fact: str = Field(default="")
    source_id: str = Field(
        default="unknown",
        description="One or more source anchors such as S1-A1.",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(default="LOW")

    @field_validator("fact", mode="before")
    @classmethod
    def normalize_fact(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("source_id", mode="before")
    @classmethod
    def normalize_source_id(cls, value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value or "unknown").strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.upper().strip()
            if normalized in {"HIGH", "MEDIUM", "LOW"}:
                return normalized
        return "LOW"


class SearchAction(BaseModel):
    """Modality-specific retrieval action with adaptive depth."""

    model_config = ConfigDict(extra="ignore")

    modality: Literal["text", "image", "both"] = "text"
    query: str = Field(default="")
    top_n: int = Field(default=3, description="Adaptive retrieval depth.")

    @field_validator("modality", mode="before")
    @classmethod
    def normalize_modality(cls, value: Any) -> str:
        normalized = str(value or "text").strip().lower()
        if "both" in normalized or "multi" in normalized:
            return "both"
        if "image" in normalized or "visual" in normalized:
            return "image"
        return "text"

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str:
        return " ".join(str(value or "").split())

    @field_validator("top_n", mode="before")
    @classmethod
    def normalize_top_n(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 3
        return max(1, min(parsed, 10))


# ==============================================================================
# 4. Two-stage policy outputs
# ==============================================================================


def _normalize_decision_value(value: Any) -> str:
    """Normalize common model variants without mapping INSUFFICIENT to SUFFICIENT."""
    if not isinstance(value, str):
        return "FURTHER_SEARCH_NEEDED"

    normalized = value.upper().strip().replace("-", "_").replace(" ", "_")

    if normalized in {
        "REFERENCE_SUFFICIENT",
        "FURTHER_SEARCH_NEEDED",
        "NO_SEARCH_NEEDED",
    }:
        return normalized

    if "NO_SEARCH" in normalized or normalized in {"DIRECT", "ANSWER_DIRECTLY"}:
        return "NO_SEARCH_NEEDED"

    if (
        "INSUFFICIENT" in normalized
        or "FURTHER" in normalized
        or "CONTINUE_SEARCH" in normalized
        or "MORE_EVIDENCE" in normalized
    ):
        return "FURTHER_SEARCH_NEEDED"

    if (
        normalized == "SUFFICIENT"
        or "REFERENCE_SUFFICIENT" in normalized
        or "EVIDENCE_SUFFICIENT" in normalized
    ):
        return "REFERENCE_SUFFICIENT"

    return "FURTHER_SEARCH_NEEDED"


class Step1CognitiveOutput(BaseModel):
    """Belief-gap-decision output of the cognitive policy."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    decision: Literal[
        "REFERENCE_SUFFICIENT",
        "FURTHER_SEARCH_NEEDED",
        "NO_SEARCH_NEEDED",
    ] = "FURTHER_SEARCH_NEEDED"
    missing_info_gaps: List[MissingInformation] = Field(default_factory=list)
    belief_state: List[Belief] = Field(default_factory=list)
    dataset_vibe_analysis: str = Field(default="", alias="dataset_vibe")

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: Any) -> str:
        return _normalize_decision_value(value)

    @field_validator("belief_state", mode="before")
    @classmethod
    def normalize_beliefs(cls, value: Any) -> List[Belief]:
        if not value:
            return []
        if isinstance(value, (dict, Belief)):
            value = [value]
        normalized: List[Belief] = []
        for item in value if isinstance(value, list) else []:
            if isinstance(item, Belief):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Belief(**item))
        return normalized

    @field_validator("missing_info_gaps", mode="before")
    @classmethod
    def normalize_gaps(cls, value: Any) -> List[MissingInformation]:
        if not value:
            return []

        if isinstance(value, (str, dict, MissingInformation)):
            value = [value]

        normalized: List[MissingInformation] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(
                    MissingInformation(description=item, suggested_modality="both")
                )
            elif isinstance(item, dict):
                normalized.append(MissingInformation(**item))
            elif isinstance(item, MissingInformation):
                normalized.append(item)
        return normalized


class Step2ActionOutput(BaseModel):
    """Gap-driven retrieval actions generated by the planner."""

    model_config = ConfigDict(extra="ignore")
    next_search_actions: List[SearchAction] = Field(default_factory=list)

    @field_validator("next_search_actions", mode="before")
    @classmethod
    def normalize_actions(cls, value: Any) -> List[SearchAction]:
        if not value:
            return []
        if isinstance(value, (dict, SearchAction)):
            value = [value]
        normalized: List[SearchAction] = []
        for item in value if isinstance(value, list) else []:
            if isinstance(item, SearchAction):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(SearchAction(**item))
        return normalized


# ==============================================================================
# 5. Unified trace container
# ==============================================================================


class MetaControllerOutput(BaseModel):
    """Combined output stored in asymmetric memory and execution logs."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    meta_controller_think: str = Field(default="")
    decision: Literal[
        "REFERENCE_SUFFICIENT",
        "FURTHER_SEARCH_NEEDED",
        "NO_SEARCH_NEEDED",
    ] = "FURTHER_SEARCH_NEEDED"
    missing_info_gaps: List[MissingInformation] = Field(default_factory=list)
    belief_state: List[Belief] = Field(default_factory=list)
    dataset_vibe_analysis: str = Field(default="", alias="dataset_vibe")
    next_search_actions: List[SearchAction] = Field(default_factory=list)
    useful_asset_ids: List[str] = Field(default_factory=list)
    extracted_answer: Optional[str] = None

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: Any) -> str:
        return _normalize_decision_value(value)

    @field_validator("belief_state", mode="before")
    @classmethod
    def normalize_beliefs(cls, value: Any) -> List[Belief]:
        return Step1CognitiveOutput.normalize_beliefs(value)

    @field_validator("missing_info_gaps", mode="before")
    @classmethod
    def normalize_gaps(cls, value: Any) -> List[MissingInformation]:
        return Step1CognitiveOutput.normalize_gaps(value)


# ==============================================================================
# 6. Draft and final answer
# ==============================================================================


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: Literal["OPTIMIZED", "UNABLE_TO_OPTIMIZE"] = "OPTIMIZED"
    actions: List[SearchAction] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    current_answer: str = ""
    status: Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"] = "PARTIAL"
    open_gaps: List[str] = Field(default_factory=list)


class FinalAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    final_answer: Any
