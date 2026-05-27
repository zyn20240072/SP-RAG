import json
from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Asset(BaseModel):
    model_config = ConfigDict(extra='ignore')
    asset_id: str = Field(default="unknown")
    
    short_id: Optional[str] = Field(default=None, description="Step-prefixed short ID like S1-A1")
    
    type: Literal['text', 'image'] = 'text'
    content: str = ""
    source_queries: List[str] = []
    metadata: Dict[str, Any] = {}


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    asset_id: str = Field(default="unknown")
    type: Literal['text', 'image'] = Field(default="text")
    
    reasoning: str = Field(default="", description="Validator's thought process")
    relevance_score: int = Field(default=1, ge=1, le=5, description="1-5 heuristic score for relevance") 
    search_suggestion: Literal['search_image', 'search_text', 'search_both', 'none'] = Field(
        default='none', 
        description="Suggestion for next step if content is messy or partial."
    )
    
    @field_validator('search_suggestion', mode='before')
    @classmethod
    def normalize_suggestion(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.lower().strip()
            if 'image' in v: return 'search_image'
            if 'text' in v: return 'search_text'
            if 'both' in v: return 'search_both'
        return 'none'


class ConflictResolution(BaseModel):
    model_config = ConfigDict(extra='ignore')
    conflict_desc: str = Field(default="", description="Description of the conflict")
    winning_source: Literal['IMAGE', 'TEXT', 'NEWER_SOURCE', 'NONE'] = Field(default="IMAGE")

class MissingInformation(BaseModel):
    model_config = ConfigDict(extra='ignore')
    description: str = Field(default="", description="")
    suggested_modality: Literal['text', 'image', 'both'] = Field(default='text')

class Belief(BaseModel):
    model_config = ConfigDict(extra='ignore')
    fact: str = Field(default="")
    source_id: str = Field(default="unknown", description="Useful IDs (如 S1-A1)")
    confidence: Literal['HIGH', 'MEDIUM', 'LOW'] = Field(default='LOW')
    
    # Optional: ensure confidence validator explicitly returns one of the literals if parsing fails
    @field_validator('confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.upper().strip()
            if v in ['HIGH', 'MEDIUM', 'LOW']: return v
        return 'LOW'

class SearchAction(BaseModel):
    model_config = ConfigDict(extra='ignore')
    modality: Literal['text', 'image', 'both'] = 'text'
    query: str = Field(default="")
    top_n: int = Field(default=3, description="Dynamic Retrieval Sizing")

class Step1CognitiveOutput(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)
    
    decision: Literal['REFERENCE_SUFFICIENT', 'FURTHER_SEARCH_NEEDED', 'NO_SEARCH_NEEDED'] = 'FURTHER_SEARCH_NEEDED'
    
    missing_info_gaps: List[MissingInformation] = Field(default_factory=list)
    belief_state: List[Belief] = Field(default_factory=list)
    dataset_vibe_analysis: str = Field(default="", alias="dataset_vibe")

    @field_validator('decision', mode='before')
    @classmethod
    def normalize_decision(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.upper().strip()
            if 'SUFFICIENT' in v: return 'REFERENCE_SUFFICIENT'
            if 'NO' in v and 'SEARCH' in v: return 'NO_SEARCH_NEEDED'
        return 'FURTHER_SEARCH_NEEDED'
        
    @field_validator('missing_info_gaps', mode='before')
    @classmethod
    def normalize_gaps(cls, v: Any) -> List[MissingInformation]:
        if not v: return []
        res = []
        for item in v:
            if isinstance(item, str):
                res.append(MissingInformation(description=item, suggested_modality='both'))
            elif isinstance(item, dict):
                res.append(MissingInformation(**item))
            elif isinstance(item, MissingInformation):
                res.append(item)
        return res

class Step2ActionOutput(BaseModel):
    model_config = ConfigDict(extra='ignore')
    next_search_actions: List[SearchAction] = Field(default_factory=list)


class MetaControllerOutput(BaseModel):
    # ✨ 同理修复
    model_config = ConfigDict(extra='ignore', populate_by_name=True)
    
    meta_controller_think: str = Field(default="")
    decision: Literal['REFERENCE_SUFFICIENT', 'FURTHER_SEARCH_NEEDED', 'NO_SEARCH_NEEDED'] = 'FURTHER_SEARCH_NEEDED'
    
    missing_info_gaps: List[MissingInformation] = Field(default_factory=list)
    belief_state: List[Belief] = Field(default_factory=list)
    dataset_vibe_analysis: str = Field(default="", alias="dataset_vibe")
    
    next_search_actions: List[SearchAction] = Field(default_factory=list)
    useful_asset_ids: List[str] = Field(default_factory=list) 
    extracted_answer: Optional[str] = None
    
    @field_validator('missing_info_gaps', mode='before')
    @classmethod
    def normalize_gaps(cls, v: Any) -> List[MissingInformation]:
        if not v: return []
        res = []
        for item in v:
            if isinstance(item, str):
                res.append(MissingInformation(description=item, suggested_modality='both'))
            elif isinstance(item, dict):
                res.append(MissingInformation(**item))
            elif isinstance(item, MissingInformation):
                res.append(item)
        return res

class SearchPlan(BaseModel):
    model_config = ConfigDict(extra='ignore')
    decision: Literal['OPTIMIZED', 'UNABLE_TO_OPTIMIZE'] = 'OPTIMIZED'
    actions: List[SearchAction] = Field(default_factory=list)

class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra='ignore')
    current_answer: str = ""
    status: Literal['COMPLETE', 'PARTIAL', 'INSUFFICIENT'] = 'PARTIAL'
    open_gaps: List[str] = Field(default_factory=list)

class FinalAnswerOutput(BaseModel):
    final_answer: Any