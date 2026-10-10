import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

try:
    import json_repair
except ImportError:  # Optional dependency; standard JSON remains supported.
    json_repair = None

from pydantic import BaseModel

from api_handler import call_qwen_api, process_image_for_api
from config import TEMPERATURE
from data_schemas import (
    Asset,
    Belief,
    DraftAnswer,
    FinalAnswerOutput,
    MissingInformation,
    Step1CognitiveOutput,
    Step2ActionOutput,
    ValidationResult,
)


T = TypeVar("T", bound=BaseModel)


class LLMRoles:
    _debug_counter = 0

    def __init__(self, model_config: Dict[str, Any], image_root_path: str):
        self.model_config = model_config
        self.image_root_path = image_root_path
        # Per-agent state. It must not be shared across concurrent samples.
        self._seen_asset_ids: set[str] = set()

    def clear_memory(self) -> None:
        self._seen_asset_ids.clear()

    def _debug_print_raw(self, role_name: str, raw_text: str) -> None:
        LLMRoles._debug_counter += 1
        if LLMRoles._debug_counter % 10 == 0:
            print("\n" + "🌟" * 25)
            print(f"👀 [DEBUG - every 10 calls] RAW LLM OUTPUT ({role_name})")
            print("🌟" * 25)
            print(raw_text)
            print("🌟" * 25 + "\n")

    def _get_role_config(self, role: str) -> Dict[str, Any]:
        """Resolve one role configuration without passing the nested config by mistake."""
        if role in self.model_config:
            return self.model_config[role]

        # Backward compatibility for a single, non-nested model configuration.
        if "model_name" in self.model_config:
            return self.model_config

        raise KeyError(
            f"Role '{role}' is not configured. Available roles: "
            f"{sorted(self.model_config.keys())}"
        )

    def _parse_json_response(
        self,
        response: str,
        model_class: Type[T],
    ) -> Tuple[T, str]:
        think_content = ""
        try:
            think_match = re.search(
                r"<think>(.*?)</think>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if think_match:
                think_content = think_match.group(1).strip()
                target_str = response.replace(think_match.group(0), "").strip()
            else:
                target_str = response

            target_str = re.sub(r"</?answer>", "", target_str, flags=re.IGNORECASE)
            target_str = re.sub(
                r"```(?:json)?\s*|```",
                "",
                target_str,
                flags=re.IGNORECASE,
            ).strip()

            # Closed-source APIs occasionally add a short preface around an otherwise
            # valid object. Prefer the outermost JSON object in that case.
            if not target_str.startswith(("{", "[")):
                first_object = target_str.find("{")
                last_object = target_str.rfind("}")
                first_array = target_str.find("[")
                last_array = target_str.rfind("]")
                candidates = []
                if first_object >= 0 and last_object > first_object:
                    candidates.append(target_str[first_object:last_object + 1])
                if first_array >= 0 and last_array > first_array:
                    candidates.append(target_str[first_array:last_array + 1])
                if candidates:
                    target_str = max(candidates, key=len)

            try:
                if json_repair is not None:
                    data = json_repair.loads(target_str)
                else:
                    data = json.loads(target_str)
            except Exception as exc:
                raise ValueError("JSON decoding failed") from exc

            if isinstance(data, list):
                data = next(
                    (item for item in data if isinstance(item, dict)),
                    data[0] if data else {},
                )

            if not isinstance(data, dict) or not data:
                raise ValueError("Parsed result is not a populated JSON object")

            if hasattr(model_class, "model_fields"):
                allowed_fields = set(model_class.model_fields.keys())
                aliases = {
                    field.alias
                    for field in model_class.model_fields.values()
                    if field.alias
                }
                data = {
                    key: value
                    for key, value in data.items()
                    if key in allowed_fields or key in aliases
                }

            return model_class.model_validate(data), think_content
        except Exception as exc:
            raise ValueError(f"JSON parsing failed: {exc}") from exc

    @staticmethod
    def _normalize_answer_value(raw_answer: Any) -> str:
        """Normalize scalar, list, and mapping answers consistently across branches."""
        if isinstance(raw_answer, dict):
            elements: List[str] = []
            for key, value in raw_answer.items():
                if isinstance(value, list):
                    elements.extend(str(item).strip() for item in value)
                else:
                    elements.append(f"{key}: {value}".strip())
            return ", ".join(dict.fromkeys(item for item in elements if item))
        if isinstance(raw_answer, list):
            return ", ".join(
                dict.fromkeys(
                    str(item).strip() for item in raw_answer if str(item).strip()
                )
            )
        return str(raw_answer or "").strip()

    # ==============================================================================
    # Layer 1: High-recall validator
    # ==============================================================================

    def validate_asset(
        self,
        asset: Asset,
        question: str,
    ) -> Tuple[ValidationResult, str]:
        # The 1-5 validator rubric is intentionally unchanged from the original code.
        head_text = """Role: High-Recall Auditor.
Task: Evaluate the individual asset for relevance to the user's question and identify quality-driven search needs.

### System Context
- Asset Content:
"""
        mixed_content: List[Dict[str, Any]] = [
            {"type": "text", "text": head_text}
        ]
        meta_info = (
            f"Asset ID: {asset.asset_id}\n"
            f"Type: {asset.type.upper()}\n"
        )

        if asset.type == "image":
            mixed_content.append(
                {
                    "type": "text",
                    "text": meta_info + "(Visual Content Below):\n<image>",
                }
            )
            image_obj = process_image_for_api(asset.content, self.image_root_path)
            if image_obj:
                mixed_content.append(image_obj)
            else:
                mixed_content.append(
                    {"type": "text", "text": "[Error: Image file missing]"}
                )
        else:
            mixed_content.append(
                {
                    "type": "text",
                    "text": meta_info + f"Content:\n{asset.content}",
                }
            )

        tail_text = f"""
- User Question: "{question}"

### JUDGMENT RULES
1. Relevance Score (1-5 Scale):
   - [5] Exact Match / Golden Evidence: The asset directly and completely answers the question, or provides undeniable visual/textual evidence. (Equivalent to a direct answer).
   - [4] Highly Relevant (Partial Clues): The asset cannot answer the question alone, but provides crucial sub-entity information, background, or direct clues required to piece together the answer.
   - [3] Ambiguous / Edge Case: The asset mentions core entities, but the context is unclear, or the content is too corrupted/noisy to confidently determine its utility without further context.
   - [2] Marginally Relevant: Shares broad topical keywords with the question, but provides absolutely zero actionable information to solve the specific query.
   - [1] Completely Irrelevant: Irrelevant noise. Shares no meaningful semantic overlap with the core intent of the question.

2. Search Suggestion (Based on Asset Quality):
   - "search_image": Use if the current TEXT is messy OCR, disjointed, or requires visual confirmation to verify.
   - "search_text": Use if the current IMAGE is blurry, incomplete, or requires descriptive textual details to understand.
   - "search_both": Use if the asset introduces a new critical concept that requires broad multimodal exploration.
   - "none": Use if the asset is perfectly clear (Score 5) or completely irrelevant (Score 1-2) requiring no specific modal fallback.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "relevance_score": <int between 1 and 5>,
  "search_suggestion": "search_image" | "search_text" | "search_both" | "none"
}}"""
        mixed_content.append({"type": "text", "text": tail_text})

        max_format_retries = 2
        last_error: Optional[Exception] = None
        for attempt in range(max_format_retries):
            try:
                role_cfg = self._get_role_config("validator")
                response = call_qwen_api(
                    role_cfg,
                    interleaved_content=mixed_content,
                    temperature=TEMPERATURE,
                )
                result, think_content = self._parse_json_response(
                    response,
                    ValidationResult,
                )
                result.asset_id = asset.asset_id
                result.type = asset.type
                return result, think_content
            except Exception as exc:
                last_error = exc
                print(
                    f"⚠️ [Validator Retry {attempt + 1}/{max_format_retries}] "
                    f"error: {exc}"
                )

        raise RuntimeError(
            f"Validator persistently failed JSON format: {last_error}"
        )

    # ==============================================================================
    # Layer 2 / Step 1: belief-gap-decision perception
    # ==============================================================================

    def evaluate_cognitive_state(
        self,
        question: str,
        assets_in_view: List[Asset],
        prev_beliefs: List[Belief],
        past_queries_report: str,
        short_id_map: Dict[str, str],
        dataset_vibe: str = "",
        validation_map: Optional[Dict[str, ValidationResult]] = None,
        evidence_signal_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Step1CognitiveOutput, str, List[Asset]]:
        beliefs_str = (
            json.dumps(
                [belief.model_dump() for belief in prev_beliefs],
                ensure_ascii=False,
            )
            if prev_beliefs
            else "[]"
        )
        vibe_str = dataset_vibe or "No quirks identified yet."

        new_assets: List[Asset] = []
        for asset in assets_in_view:
            if asset.asset_id in self._seen_asset_ids:
                continue
            new_assets.append(asset)

        head_text = """Role: Cognitive Orchestrator (Meta Controller).
Task: Analyze the evidence pool, update the belief state, identify missing information gaps, and evaluate dataset characteristics.

### System Context
- NEW Evidence Pool:
"""
        mixed_content: List[Dict[str, Any]] = [
            {"type": "text", "text": head_text}
        ]

        if not new_assets:
            mixed_content.append(
                {
                    "type": "text",
                    "text": (
                        "No unseen assets are available in this round. "
                        "Reassess the Accumulated Belief State and search history."
                    ),
                }
            )

        for asset in new_assets:
            short_id = short_id_map.get(asset.asset_id, "?")
            validation = validation_map.get(asset.asset_id) if validation_map else None
            relevance_score: Any = (
                validation.relevance_score if validation else "UNKNOWN"
            )
            suggestion = validation.search_suggestion if validation else "none"
            evidence_signal = (evidence_signal_map or {}).get(asset.asset_id, {})
            query_hits = evidence_signal.get("unique_query_count", 1)
            repeat_bonus = evidence_signal.get("repeat_bonus", 0.0)
            admission_score = evidence_signal.get(
                "admission_score",
                relevance_score,
            )
            header = (
                f"\n=== [Asset {short_id}] [Type: {asset.type.upper()}] "
                f"[Validator Search Suggestion: {suggestion}] "
                f"[Validator Rating: {relevance_score}] "
                f"[Distinct Query Hits: {query_hits}] "
                f"[Repeat Bonus: {repeat_bonus}] "
                f"[Admission Score: {admission_score}] ===\n"
            )

            if asset.type == "image":
                mixed_content.append(
                    {"type": "text", "text": header + "(Visual Content Below)"}
                )
                image_obj = process_image_for_api(
                    asset.content,
                    self.image_root_path,
                )
                if image_obj:
                    mixed_content.append(image_obj)
                else:
                    mixed_content.append(
                        {"type": "text", "text": "[Image unavailable]"}
                    )
            else:
                mixed_content.append(
                    {
                        "type": "text",
                        "text": header + f"Content:\n{asset.content}",
                    }
                )

        tail_text = f"""
- Question: "{question}"
- Accumulated Belief State: {beliefs_str}
- Search History:
{past_queries_report}
- Dataset Strategy/Vibe Analysis:
{vibe_str}

### ROLEPLAY INSTRUCTIONS
*CROSS-MODAL ARBITRATION*: Preserve only claims that remain supported by the clearest available assets. Do not blindly retain conflicting beliefs. If a conflict cannot be resolved, omit the disputed claim from `belief_state` and describe the unresolved conflict in `missing_info_gaps`.
*SOFT VALIDATION*: The validator rating is a noisy local relevance estimate, not an irreversible filter. Distinct-query hits and the bounded repeat bonus indicate retrieval consistency. Judge the original content directly; a score-2 asset with sufficient repeat support may still contain useful answer or bridge evidence.
*BELIEF INTERNALIZATION*: Carry forward only question-relevant beliefs that remain evidence-supported. Distill new useful evidence into standalone beliefs, preserve provenance in `source_id` (e.g., "S1-A1"), and assign confidence according to the cited asset. A belief may cite multiple assets when joint support is required.
*ANALYSIS OF DATA CHARACTERISTICS*: Update `dataset_vibe_analysis` only with retrieval-relevant observations, such as modality mismatch, OCR corruption, sparse evidence, or repeated source patterns.
*MISSING INFORMATION ANALYSIS*: Identify the exact answer-bearing entity, value, attribute, or relation still absent. Leave `missing_info_gaps` empty only when the current cited beliefs and raw assets are sufficient for the requested answer.
*DECISION CONTRACT*:
- Use `REFERENCE_SUFFICIENT` only when the current cited beliefs and raw assets jointly support the complete answer.
- Use `FURTHER_SEARCH_NEEDED` when external evidence is still required. Provide at least one explicit unresolved gap.
- Use `NO_SEARCH_NEEDED` only when the question can be answered without external evidence.
- Do not use `NO_SEARCH_NEEDED` merely because the current retrieval round failed. If current assets are unhelpful but external evidence is required, use `FURTHER_SEARCH_NEEDED` and specify a corrected or broader gap.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "belief_state": [
    {{
      "fact": "Clear standalone fact relevant to the question.",
      "source_id": "S#-A#",
      "confidence": "HIGH" | "MEDIUM" | "LOW"
    }}
  ],
  "dataset_vibe_analysis": "Retrieval-relevant dataset observations.",
  "missing_info_gaps": [
    {{
      "description": "Exact missing answer-bearing information.",
      "suggested_modality": "text" | "image" | "both"
    }}
  ],
  "decision": "REFERENCE_SUFFICIENT" | "FURTHER_SEARCH_NEEDED" | "NO_SEARCH_NEEDED"
}}"""
        mixed_content.append({"type": "text", "text": tail_text})

        max_format_retries = 2
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("cognitive")
            response = call_qwen_api(
                role_cfg,
                interleaved_content=mixed_content,
                temperature=TEMPERATURE,
            )
            self._debug_print_raw("Step 1 (Cognition)", response)
            try:
                output, think_content = self._parse_json_response(
                    response,
                    Step1CognitiveOutput,
                )
                self._seen_asset_ids.update(asset.asset_id for asset in new_assets)
                return output, think_content, new_assets
            except Exception as exc:
                print(response)
                print(
                    f"\n⚠️ [Cognition Retry {attempt + 1}/{max_format_retries}] "
                    f"output invalid: {exc}"
                )
                if attempt == max_format_retries - 1:
                    raise RuntimeError(
                        f"Step 1 persistently failed: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable cognition failure")

    # ==============================================================================
    # Layer 2 / Step 2: gap-driven retrieval planning
    # ==============================================================================

    def plan_next_actions(
        self,
        question: str,
        dataset_vibe: str,
        step1_gaps: List[MissingInformation],
        beliefs: List[Belief],
        past_queries_report: str,
    ) -> Tuple[Step2ActionOutput, str]:
        gaps_str = (
            json.dumps(
                [gap.model_dump() for gap in step1_gaps],
                ensure_ascii=False,
            )
            if step1_gaps
            else "[]"
        )
        beliefs_str = (
            json.dumps(
                [belief.model_dump() for belief in beliefs],
                ensure_ascii=False,
            )
            if beliefs
            else "[]"
        )

        prompt_text = f"""Role: Expert Search Engineer (Query Optimizer).
Task: Translate unresolved information gaps into precise modality-specific vector-search actions.

### System Context
- Question: "{question}"
- Dataset Strategy/Vibe Analysis: "{dataset_vibe}"
- Grounded Belief Anchors: {beliefs_str}
- Identified Information Gaps: {gaps_str}
- Historical Search Logs:
{past_queries_report}

### ROLEPLAY INSTRUCTIONS
*GAP FIRST*: Every action must target one explicit unresolved gap. Do not search for information already grounded in the beliefs.
*GROUNDED QUERYING*: Combine a verified entity, source title, document clue, or relation from the belief anchors with the exact missing attribute or relation. If no reliable belief anchor exists, use exact entities from the question and gap; never invent an answer or source name.
*MODALITY SELECTION*: Follow the gap's suggested modality and the dataset observations. Split a `both` need into separate text and image actions when appropriate.
*DYNAMIC SIZING*: Use top_n 2-3 for narrow factual gaps and top_n 4-6 for broad or uncertain gaps.
*NON-REDUNDANCY*: Do not repeat a previous query or a semantically equivalent search intent. Reformulate around the unresolved relation rather than adding generic words.
*ACTION LIMIT*: Return at most two high-value actions for the current round.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "next_search_actions": [
    {{
      "modality": "text" | "image",
      "query": "Verified source/entity anchor plus the exact missing relation or attribute",
      "top_n": 2
    }}
  ]
}}"""

        max_format_retries = 2
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("planner")
            response = call_qwen_api(
                role_cfg,
                prompt=prompt_text,
                temperature=TEMPERATURE,
            )
            self._debug_print_raw("Step 2 (Action Plan)", response)
            try:
                output, think_content = self._parse_json_response(
                    response,
                    Step2ActionOutput,
                )
                # Keep only executable, non-empty actions.
                output.next_search_actions = [
                    action
                    for action in output.next_search_actions[:2]
                    if action.query.strip()
                ]
                return output, think_content
            except Exception as exc:
                print(
                    f"\n⚠️ [Planner Retry {attempt + 1}/{max_format_retries}] "
                    f"output invalid: {exc}"
                )
                if attempt == max_format_retries - 1:
                    raise RuntimeError(
                        f"Step 2 persistently failed: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable planner failure")

    # ==============================================================================
    # Layer 3: final evidence-grounded response
    # ==============================================================================

    def generate_final_response(
        self,
        question: str,
        draft: Optional[DraftAnswer],
        final_assets: List[Asset],
        short_id_map: Dict[str, str],
        prev_beliefs: List[Belief],
    ) -> Tuple[str, str]:
        beliefs_str = (
            json.dumps(
                [belief.model_dump() for belief in prev_beliefs],
                ensure_ascii=False,
            )
            if prev_beliefs
            else "[]"
        )

        head_text = """Role: Expert Synthesizer (Final Responder).
Task: Produce an evidence-grounded answer from the final belief-evidence bundle.

### System Context
- Original Evidence Assets:
"""
        mixed_content: List[Dict[str, Any]] = [
            {"type": "text", "text": head_text}
        ]

        if not final_assets:
            mixed_content.append(
                {"type": "text", "text": "No cited evidence available."}
            )
        else:
            for asset in final_assets:
                short_id = short_id_map.get(asset.asset_id, "?")
                header = (
                    f"\n=== [Asset {short_id}] "
                    f"[Type: {asset.type.upper()}] ===\n"
                )
                if asset.type == "image":
                    mixed_content.append(
                        {
                            "type": "text",
                            "text": header + "(Visual Observation Below)",
                        }
                    )
                    image_obj = process_image_for_api(
                        asset.content,
                        self.image_root_path,
                    )
                    if image_obj:
                        mixed_content.append(image_obj)
                    else:
                        mixed_content.append(
                            {"type": "text", "text": "[Image unavailable]"}
                        )
                else:
                    mixed_content.append(
                        {
                            "type": "text",
                            "text": header + f"{asset.content}\n",
                        }
                    )

        tail_text = f"""
- User Question: "{question}"
- Grounded Belief Index:
{beliefs_str}

### ROLEPLAY INSTRUCTIONS
1. Treat each belief as a provenance-linked pointer to its cited original asset, not as independent evidence.
2. Verify every fact used in the answer against the corresponding original asset label (for example, [Asset S1-A1] or [Asset SF-A1]). The original asset is authoritative if the belief omits a condition, scope, qualifier, or visual detail.
3. Use the belief index to locate and combine relevant evidence. Ignore any belief that cannot be verified against its cited asset.
4. Return the minimal complete answer required by the question. Include every requested item, unit, qualifier, or relation; do not add explanations unless requested.
5. Cross-check that the answer addresses the exact subject and relation in the question.
6. If the included assets do not support the requested answer, return "Information insufficient."

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "final_answer": "The minimal complete answer requested by the question."
}}"""
        mixed_content.append({"type": "text", "text": tail_text})

        max_format_retries = 2
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("responder")
            response = call_qwen_api(
                role_cfg,
                interleaved_content=mixed_content,
                temperature=TEMPERATURE,
            )
            self._debug_print_raw("Final Responder", response)

            try:
                final_output, think_content = self._parse_json_response(
                    response,
                    FinalAnswerOutput,
                )
                answer_text = self._normalize_answer_value(
                    final_output.final_answer
                )

                if not answer_text:
                    raise ValueError("Empty final answer")
                return answer_text, think_content
            except Exception as exc:
                print(f"Wrong output: {response}")
                print(
                    f"\n⚠️ [Final Responder Retry {attempt + 1}/"
                    f"{max_format_retries}] output invalid: {exc}"
                )
                if attempt == max_format_retries - 1:
                    raise RuntimeError(
                        "Final Responder persistently failed JSON formatting: "
                        f"{exc}"
                    ) from exc

        raise RuntimeError("Unreachable responder failure")

    def generate_fallback_response(self, question: str) -> Tuple[str, str]:
        """Direct-answer branch used only for NO_SEARCH_NEEDED."""
        prompt_text = f"""User Question: "{question}"

The cognitive policy has determined that external evidence is unnecessary.
Answer using internal knowledge and return the minimal complete answer.
Do not add a preface.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "final_answer": "The minimal complete answer requested by the question."
}}"""
        max_format_retries = 2
        last_error: Optional[Exception] = None
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("fallback")
            response = call_qwen_api(
                role_cfg,
                prompt=prompt_text,
                temperature=TEMPERATURE,
            )
            self._debug_print_raw("Fallback Responder", response)
            try:
                output, think_content = self._parse_json_response(
                    response,
                    FinalAnswerOutput,
                )
                answer_text = self._normalize_answer_value(output.final_answer)
                if not answer_text:
                    raise ValueError("Empty fallback answer")
                return answer_text, think_content
            except Exception as exc:
                last_error = exc
                print(
                    f"⚠️ [Fallback Retry {attempt + 1}/{max_format_retries}] "
                    f"output invalid: {exc}"
                )

        raise RuntimeError(
            f"Fallback responder persistently failed: {last_error}"
        ) from last_error
