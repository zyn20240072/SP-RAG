import json
import re
from typing import List, Dict, Any, Type, TypeVar, Tuple, Optional
from pydantic import BaseModel
import json_repair 

from api_handler import call_qwen_api, process_image_for_api
from data_schemas import (
    ValidationResult, SearchAction, Asset, DraftAnswer, 
    MissingInformation, Belief, 
    Step1CognitiveOutput, Step2ActionOutput,
    FinalAnswerOutput
)
from config import TEMPERATURE

T = TypeVar('T', bound=BaseModel)

class LLMRoles:

    def _get_role_config(self, role: str) -> Dict[str, Any]:
        if role in self.model_config:
            return self.model_config[role]
        return self.model_config

    def __init__(self, model_config: Dict[str, Any], image_root_path: str):
        self.model_config = model_config
        self.image_root_path = image_root_path
        self.clear_memory()

    def clear_memory(self):
        self._seen_asset_ids.clear()

    def _parse_json_response(self, response: str, model_class: Type[T]) -> Tuple[T, str]:
        think_content = ""
        target_str = ""
        try:
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
            if think_match:
                think_content = think_match.group(1).strip()
                target_str = response.replace(think_match.group(0), '').strip()
            else:
                target_str = response

            target_str = re.sub(r'</?answer>', '', target_str, flags=re.IGNORECASE)
            target_str = re.sub(r'```json\n?|```', '', target_str).strip()
            
            try:
                data = json_repair.loads(target_str)
            except:
                raise ValueError("json_repair parsing failed.")
                
            if isinstance(data, list):
                data = next((item for item in data if isinstance(item, dict)), data[0] if data else {})

            if not isinstance(data, dict) or not data:
                raise ValueError("Parsed result is not a valid or populated JSON dictionary.")

            if isinstance(data, dict) and hasattr(model_class, "model_fields"):
                allowed_fields = model_class.model_fields.keys()
                data = {k: v for k, v in data.items() if k in allowed_fields or k in [f.alias for f in model_class.model_fields.values() if f.alias]}

            return model_class.model_validate(data), think_content
        except Exception as e:
            raise ValueError(f"JSON Parsing failed: {e}")

    # ==============================================================================
    #  Layer 1: Validator
    # ==============================================================================
    def validate_asset(self, asset: Asset, question: str) -> Tuple[ValidationResult, str]:
        head_text = f"""Role: High-Recall Auditor. 
Task: Evaluate the individual asset for relevance to the user's question and identify quality-driven search needs.

### System Context
- Asset Content:
"""
        mixed_content = [{"type": "text", "text": head_text}]
        meta_info = f"Asset ID: {asset.asset_id}\nType: {asset.type.upper()}\n"
        
        if asset.type == 'image':
            mixed_content.append({"type": "text", "text": meta_info + "(Visual Content Below):\n<image>"})
            img_obj = process_image_for_api(asset.content, self.image_root_path)
            if img_obj: mixed_content.append(img_obj)
            else: mixed_content.append({"type": "text", "text": "[Error: Image file missing]"})
        else:
            mixed_content.append({"type": "text", "text": meta_info + f"Content:\n{asset.content}"})

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
        last_error = None
        for attempt in range(max_format_retries):
            try:
                role_cfg = self._get_role_config("validator")
                response = call_qwen_api(role_cfg, interleaved_content=mixed_content, temperature=TEMPERATURE)
                result, think_content = self._parse_json_response(response, ValidationResult)
                result.asset_id = asset.asset_id 
                result.type = asset.type
                return result, think_content
            except Exception as e:
                last_error = e
                print(f"⚠️ [Validator Retry {attempt + 1}/{max_format_retries}] error: {e}")
                
        raise RuntimeError(f"Validator persistently failed JSON format: {last_error}")

    # ==============================================================================
    #  Layer 2 - Step 1: cognition
    # ==============================================================================
    def evaluate_cognitive_state(self, 
                                 question: str, 
                                 assets_in_view: List[Asset],
                                 prev_beliefs: List[Belief],
                                 past_queries_report: str,
                                 short_id_map: dict,
                                 dataset_vibe: str = "", 
                                 validation_map: Dict[str, ValidationResult] = None) -> Tuple[Step1CognitiveOutput, str, List[Asset]]: 
        
        beliefs_str = json.dumps([b.model_dump() for b in prev_beliefs], ensure_ascii=False) if prev_beliefs else "[]"
        vibe_str = dataset_vibe if dataset_vibe else "No quirks identified yet."
        
        new_assets = []
        for asset in assets_in_view:
            if asset.asset_id not in self._seen_asset_ids:
                new_assets.append(asset)
                self._seen_asset_ids.add(asset.asset_id)
        
        head_text = f"""Role: Cognitive Orchestrator (Meta Controller).
Task: Analyze the evidence pool, update the belief state, identify missing information gaps, and evaluate dataset characteristics.

### System Context
- NEW Evidence Pool:
"""
        mixed_content = [{"type": "text", "text": head_text}]

        if not new_assets:
             mixed_content.append({"type": "text", "text": "No new relevant assets found in this search round. Rely strictly on your Accumulated Belief State."})
        
        for asset in new_assets:
            short_id = short_id_map.get(asset.asset_id, "?")
            val_result = validation_map.get(asset.asset_id) if validation_map else None
            relevance_score = val_result.relevance_score if val_result else "UNKNOWN"
            valid_sgg =  val_result.search_suggestion if val_result else "none"

            header = f"\n=== [Asset {short_id}] [Type: {asset.type.upper()}] [Validator Search Suggestion: {valid_sgg}] [Validator Rating: {relevance_score}] ===\n"

            if asset.type == 'image':
                mixed_content.append({"type": "text", "text": header + "(Visual Content Below)"})
                img_obj = process_image_for_api(asset.content, self.image_root_path)
                if img_obj: mixed_content.append(img_obj)
            else:
                mixed_content.append({"type": "text", "text": header + f"Content:\n{asset.content}"})

        tail_text = f"""
- Question: "{question}"
- Accumulated Belief State: {beliefs_str}
- Dataset Strategy/Vibe Analysis: \n{vibe_str}

### ROLEPLAY INSTRUCTIONS
*CROSS-MODAL ARBITRATION*: Anchor reasoning entirely on the clearer assets if conflicts exist. If a conflict arises with the Accumulated Belief State, retain both in `belief_state`.
*BELIEF INTERNALIZATION*: First, you should choose the Accumulated Belief States which are genuinely relevant to the problem and include them in `belief_state`. Then, You MUST distill useful evidence into the `belief_state`, use the `source_id` field (e.g., "S1-A1") to track where the fact came from and give the `confidence` field to describe your confidence.
*ANALYSIS of DATA CHARACTERISTICS*: Based on the Dataset Strategy/Vibe Analysis and the data types and characteristics of the New Evidence Pool, update the `dataset_vibe_analysis` field and analyze the needed modalities and information.
*Missing Information Analysis*: Based on the Accumulated Belief State and the NEW Evidence Pool, assess what additional information is potentially required to address the current problem, and provide explicit recommendations regarding the search modalities to be employed. Don't write anything in `missing_info_gaps` if the info is sufficient.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "belief_state":[ {{"fact": "Clear, standalone sentence containing the extracted fact which is relevant to the question.", "source_id": "S#-A#", "confidence": "HIGH" | "MEDIUM" | "LOW"}} ],
  "dataset_vibe_analysis": "Note on dataset quirks or patterns.",  
  "missing_info_gaps": [
    {{
      "description": "What exact info is missing.",
      "suggested_modality": "text" | "image" | "both"
    }}
  ],
  "decision": "REFERENCE_SUFFICIENT" | "FURTHER_SEARCH_NEEDED" | "NO_SEARCH_NEEDED"
}}"""
        mixed_content.append({"type": "text", "text": tail_text})

        max_format_retries = 2 
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("cognitive")
            response = call_qwen_api(role_cfg, interleaved_content=mixed_content, temperature=TEMPERATURE)
            try:
                output, think_content = self._parse_json_response(response, Step1CognitiveOutput)
                if not output or not output.decision: raise ValueError("Invalid JSON output.")
                return output, think_content, new_assets
            except Exception as e:
                print(response)
                print(f"\n⚠️ [Cognition Retry {attempt + 1}/{max_format_retries}] output invalid: {e}")
                if attempt == max_format_retries - 1:
                    raise RuntimeError(f"Step 1 persistently failed: {e}")

    # ==============================================================================
    #  Layer 2 - Step 2: planner
    # ==============================================================================
    def plan_next_actions(self, question: str, dataset_vibe: str, step1_gaps: List[MissingInformation], past_queries_report: str) -> Tuple[Step2ActionOutput, str]:
        gaps_str = json.dumps([g.model_dump() for g in step1_gaps], ensure_ascii=False) if step1_gaps else "[]"
        
        prompt_text = f"""Role: Expert Search Engineer (Query Optimizer).
Task: Translate identified information targets into highly effective, keyword-dense search queries for a vector database.

### System Context
- Question: "{question}"
- Dataset Strategy/Vibe Analysis: "{dataset_vibe}"
- Identified Information Targets: {gaps_str}
- Historical Search Logs: \n{past_queries_report}

### ROLEPLAY INSTRUCTIONS
*DYNAMIC SIZING*: If the scope of the gap is broad., cast a wider net (top_n: 4, 5, 6). If targeting specifics, use narrow net (top_n: 2, 3).
*MODALITY SELECTION*: Select the appropriate modalities for retrieval based on the recommendations in the Dataset Strategy/Vibe Analysis and Identified Information Targets.
*AVOID REPETITION*: It must not duplicate previously searched queries; instead, it requires a degree of exploration centered around the specific problem and identified gaps.
*HyDE TRANSLATION*: Formulate rich semantic strings combining keywords and add the hypothetical declarative answers if you have the enough confidence.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "next_search_actions": [ 
    {{
      "modality": "text" | "image", 
      "query": "Rich semantic string combining keywords or the hypothetical declarative answer...", 
      "top_n": n
    }} 
  ]
}}"""
        max_format_retries = 2 
        for attempt in range(max_format_retries):
            role_cfg = self._get_role_config("planner")
            response = call_qwen_api(role_cfg, prompt=prompt_text, temperature=TEMPERATURE)            
            try:
                output, think_content = self._parse_json_response(response, Step2ActionOutput)
                return output, think_content
            except Exception as e:
                print(f"\n⚠️ [Planner Retry {attempt + 1}/{max_format_retries}] output invalid: {e}")
                if attempt == max_format_retries - 1: 
                    raise RuntimeError(f"Step 2 persistently failed: {e}")

    # ==============================================================================
    #  Layer 3: ✨ Final Responder
    # ==============================================================================
    def generate_final_response(self, question: str, draft: DraftAnswer, final_assets: List[Asset], short_id_map: dict, prev_beliefs: List[Belief]) -> Tuple[str, str]:
            draft_content = draft.model_dump_json(indent=2) if draft else "No confirmed draft."
            beliefs_str = json.dumps([b.model_dump() for b in prev_beliefs], ensure_ascii=False) if prev_beliefs else "[]"
            
            head_text = f"""Role: Expert Synthesizer (Final Responder).
Task: Verify facts against the ground truth evidence, resolve the cognitive draft, and formulate a direct, natural language answer.

### System Context
- Ground Truth Evidence (Anchors):
"""
            mixed_content = [{"type": "text", "text": head_text}]

            if not final_assets:
                mixed_content.append({"type": "text", "text": "No cited evidence available."})
            else:
                for asset in final_assets:
                    short_id = short_id_map.get(asset.asset_id, "?")
                    header = f"\n=== [Asset {short_id}] [Type: {asset.type.upper()}] ===\n"

                    if asset.type == 'image':
                        mixed_content.append({"type": "text", "text": header + "(Visual Observation Below)"})
                        
                        img_obj = process_image_for_api(asset.content, self.image_root_path)
                        if img_obj: 
                            mixed_content.append(img_obj)
                    else:
                        mixed_content.append({"type": "text", "text": header + f"{asset.content}\n"})

            tail_text = f"""
- User Question: "{question}"
- Accumulated Belief State (Core Facts):
    {beliefs_str}

### ROLEPLAY INSTRUCTIONS 
1. Synthesize: Use the logical backbone from the Belief State, but STRICTLY verify and extract exact details against the Evidence using their [Asset S#-A#] references one by one.
2. Directness: State the answer immediately. Do NOT use robotic phrases like "Based on the evidence...".
3. Verification: Before providing the answer, please cross-check it against the original question once more to ensure that the subject of the answer is correct.
4. Fidelity: If the available information is insufficient to provide an answer, please state this explicitly.

### OUTPUT SPECIFICATION (JSON ONLY)
{{
  "final_answer": "The exact answer entities/values."
}}"""
            mixed_content.append({"type": "text", "text": tail_text})
            
            max_format_retries = 2 
            for attempt in range(max_format_retries):
                role_cfg = self._get_role_config("responder")
                response = call_qwen_api(role_cfg, interleaved_content=mixed_content, temperature=TEMPERATURE)

                try:
                    final_output, think_content = self._parse_json_response(response, FinalAnswerOutput)
                    
                    raw_ans = final_output.final_answer
                    if isinstance(raw_ans, dict):
                        elements = []
                        for k, v in raw_ans.items():
                            if isinstance(v, list):
                                elements.extend([str(i).strip() for i in v])
                            else:
                                elements.append(f"{k}: {v}".strip())
                        ans_text = ", ".join(list(dict.fromkeys(elements)))
                    elif isinstance(raw_ans, list):
                        ans_text = ", ".join(list(dict.fromkeys([str(i).strip() for i in raw_ans])))
                    else:
                        ans_text = str(raw_ans).strip()
                    return ans_text, think_content
                    
                except Exception as e:
                    print(f"Wrong output: {response}")
                    print(f"\n⚠️ [Final Responder Retry {attempt + 1}/{max_format_retries}] output invalid: {e}")
                    if attempt == max_format_retries - 1: 
                        raise RuntimeError(f"Final Responder persistently failed JSON formatting: {e}")

    def generate_fallback_response(self, question: str) -> str:
        prompt_text = f"User Question: \"{question}\"\nTask: Answer using internal knowledge. Start with 'Based on general knowledge...'."
        response = call_qwen_api(self.model_config, prompt=prompt_text, temperature=TEMPERATURE)
        return response.strip()