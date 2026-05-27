import json
from typing import List, Dict, Optional, Set, Any, Tuple
from data_schemas import (
    Asset, ValidationResult, DraftAnswer, 
    MetaControllerOutput, SearchAction, Belief
)

class MemorySystem:
    def __init__(self, question: str):
        self.original_question = question
        self.assets: Dict[str, Asset] = {}
        
        # key: asset_id, value: ValidationResult
        self.validations: Dict[str, ValidationResult] = {}
        self.draft_history: List[DraftAnswer] = [] 
        
        self.query_history_log: List[Dict[str, Any]] = []
        self.dataset_vibe_log: List[Dict[str, Any]] = [] 
        
        self.asset_bonus_scores: Dict[str, float] = {}
        
        self.step_history: List[Dict[str, Any]] = []
        
        self.final_answer: Optional[str] = None 
        self.final_responder_think: Optional[str] = None
        self.short_id_to_raw: Dict[str, str] = {}

    # ==============================================================================
    #  1. foundation
    # ==============================================================================

    def add_assets(self, new_assets: List[Asset]) -> int:
        count = 0
        for asset in new_assets:
            if asset.asset_id not in self.assets:
                self.assets[asset.asset_id] = asset
                count += 1
            else:
                existing_asset = self.assets[asset.asset_id]
                new_queries = [q for q in asset.source_queries if q not in existing_asset.source_queries]
                
                if new_queries:
                    existing_asset.source_queries.extend(new_queries)
                    self.boost_assets([asset.asset_id], bonus=0.5)

        return count

    def get_unvalidated_assets(self) -> List[Asset]:
        return [asset for asset in self.assets.values() if asset.asset_id not in self.validations]

    def update_validation(self, results: List[ValidationResult]):
        for res in results:
            self.validations[res.asset_id] = res

    def get_past_queries_set(self) -> Set[Tuple[str, str]]:
        return {(log.get('modality', 'text'), log['query']) for log in self.query_history_log}
        
    def get_past_queries_list(self) -> List[str]:
        return [log['query'] for log in self.query_history_log]

    def add_query_log(self, step: int, modality: str, query: str, new_count: int, valid_count: int = 0):
        self.query_history_log.append({
            "step": step,
            "modality": modality,
            "query": query,
            "new_count": new_count,
            "valid_count": valid_count  
        })

    def update_dataset_vibe(self, step: int, vibe: str):
        if vibe:
            self.dataset_vibe_log.append({"step": step, "vibe": vibe})

    def get_search_history_report(self) -> str:
        if not self.query_history_log:
            return "No searches have been performed yet."
        
        lines = []
        for log in self.query_history_log:
            new_count = log['new_count']
            valid_count = log.get('valid_count', 0)
            
            if new_count == 0:
                status = "❌ Ineffective (0 hits)"
            elif valid_count == 0:
                status = f"⚠️ Irrelevant (Found {new_count} assets, but Validator rejected all)"
            else:
                status = f"✅ Effective (Found {valid_count} useful assets)"
                
            lines.append(
                f"[Step {log['step']}] [{log['modality'].upper()}] Query: \"{log['query']}\" -> {status}"
            )
        
        return "\n".join(lines)

    # ==============================================================================
    #  2. Short ID Routing
    # ==============================================================================

    def bind_short_ids_for_step(self, step: int, assets: List[Asset]) -> Dict[str, str]:
        """return: raw_id -> short_id """
        raw_to_short = {}
        for i, asset in enumerate(assets):
            short_id = f"S{step}-A{i+1}"
            raw_to_short[asset.asset_id] = short_id
            
            # 记录反向映射，供大模型输出后解析使用
            self.short_id_to_raw[short_id] = asset.asset_id
        return raw_to_short

    def resolve_short_ids(self, short_ids: List[str]) -> List[str]:
        raw_ids = []
        for sid in short_ids:
            if sid in self.short_id_to_raw:
                raw_ids.append(self.short_id_to_raw[sid])
        return raw_ids

    # ==============================================================================
    #  3. evidence selection
    # ==============================================================================

    def boost_assets(self, asset_ids: List[str], bonus: float = 0.5):
        for aid in asset_ids:
            self.asset_bonus_scores[aid] = self.asset_bonus_scores.get(aid, 0.0) + bonus

    def get_context_window_assets(self, max_tokens: int = 4000, exclude_ids: Set[str] = None) -> List[Asset]:
        exclude_ids = exclude_ids or set() 
        
        candidates = []
        for aid, val in self.validations.items():
            if aid in exclude_ids:  
                continue 

            score = float(getattr(val, 'relevance_score', 0))
            
            score += self.asset_bonus_scores.get(aid, 0.0)
            
            if score > 0:
                asset = self.assets.get(aid)
                if asset: candidates.append((score, asset))

        candidates.sort(key=lambda x: x[0], reverse=True)
        
        selected_assets = []
        current_tokens = 0
        
        for score, asset in candidates:
            est_tokens = len(asset.content) / 2 if asset.type == 'text' else 1500 
            if current_tokens + est_tokens <= max_tokens:
                selected_assets.append(asset)
                current_tokens += est_tokens
                
        if not selected_assets:
            all_assets = list(self.assets.values())
            return all_assets[-3:]
            
        return selected_assets

    def elect_best_evidence_set(self, max_token_limit: int = 6000) -> Tuple[List[Asset], str]:
        if not self.validations:
            return [], "NO_VALIDATED_DATA"

        candidates = []
        for v in self.validations.values():
            strong_signal = getattr(v, 'relevance_score', 0) >= 3
            has_bonus = self.asset_bonus_scores.get(v.asset_id, 0.0) > 0
            
            if strong_signal or has_bonus:
                candidates.append(v)

        if not candidates:
            all_vals = list(self.validations.values())
            all_vals.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
            fallback_vals = all_vals[:5]
            
            fallback_assets = [self.assets[v.asset_id] for v in fallback_vals if v.asset_id in self.assets]
            return fallback_assets, "LOW_SCORE_FALLBACK"

        def score_evidence(val: ValidationResult):
            score = float(getattr(val, 'relevance_score', 0))
            score += self.asset_bonus_scores.get(val.asset_id, 0.0)
            return score

        sorted_vals = sorted(candidates, key=score_evidence, reverse=True)
        
        selected_assets = []
        current_tokens = 0
        
        for val in sorted_vals:
            asset = self.assets.get(val.asset_id)
            if not asset: continue 
            
            est_tokens = len(asset.content) / 2 if asset.type == 'text' else 1500 

            selected_assets.append(asset)
            current_tokens += est_tokens
            if len(selected_assets) >= 6: break

        hint = f"Selected {len(selected_assets)} assets."
        return selected_assets, hint

    # ==============================================================================
    #  4. situation update
    # ==============================================================================

    def store_meta_output(self, meta_output: MetaControllerOutput, step: int):
        status_map = {
            'REFERENCE_SUFFICIENT': 'COMPLETE',
            'FURTHER_SEARCH_NEEDED': 'PARTIAL',
            'NO_SEARCH_NEEDED': 'INSUFFICIENT'
        }
        confidence_map = {
            "HIGH": 0.3,
            "MEDIUM": 0.2,
            "LOW": 0.1
        }
        gaps = [g.description for g in meta_output.missing_info_gaps] if meta_output.missing_info_gaps else []

        draft_text = ""
        
        if hasattr(meta_output, 'belief_state') and meta_output.belief_state:
            beliefs = []
            cited_short_ids = []
            cited_confidence_list = []
            for b in meta_output.belief_state:
                if hasattr(b, 'fact'): beliefs.append(b.fact)
                elif isinstance(b, dict) and 'fact' in b: beliefs.append(b['fact'])
                else: beliefs.append(str(b))
                
                if hasattr(b, "confidence") and hasattr(b, "source_id"):
                    ori_id = b.source_id
                    confidence = b.confidence
                elif isinstance(b, dict) and 'source_id' in b and 'confidence' in b:
                    ori_id = b.get("source_id", "")
                    confidence = b.get("confidence", "")

                if ori_id and confidence:
                    cited_short_ids.append(ori_id)
                    cited_confidence_list.append(confidence)

            for a_ids, c in zip(cited_short_ids, cited_confidence_list):
                a_id_list = [a.strip() for a in a_ids.split(",")]
                for a_id in a_id_list:
                    raw_id = self.short_id_to_raw.get(a_id, "")
                    cited_score = confidence_map.get(c, 0)
                    if raw_id:
                        self.boost_assets([raw_id], bonus=cited_score)

            draft_text += "\n[Accumulated Facts]: " + " | ".join(beliefs)
        else:
            draft_text += "\n[Accumulated Facts]: No concrete facts extracted yet."

        new_draft = DraftAnswer(
            current_answer=draft_text,
            status=status_map.get(meta_output.decision),
            open_gaps=gaps
        )
        self.draft_history.append(new_draft)

    def log_step_details(self, 
                         step: int, 
                         search_actions: List[Any], 
                         new_assets_count: int,
                         meta_output: Optional[MetaControllerOutput],
                         elected_evidence: Optional[List[Asset]] = None,
                         validation_results: Optional[List[ValidationResult]] = None,
                         meta_think: str = "",
                         optimizer_think: str = "",
                         validator_thinks: Dict[str, str] = None,
                         current_short_id_map: Dict[str, str] = None): 
        
        actions_dump = [a.model_dump() if hasattr(a, 'model_dump') else str(a) for a in search_actions]
        validations_dump = [v.model_dump() for v in validation_results] if validation_results else []

        meta_input_snapshot = []
        if elected_evidence:
            for asset in elected_evidence:
                val_info = self.validations.get(asset.asset_id)
                bonus = self.asset_bonus_scores.get(asset.asset_id, 0.0)
                short_id = current_short_id_map.get(asset.asset_id, "?") if current_short_id_map else "?"
            
                meta_input_snapshot.append({
                    "short_id": short_id, 
                    "asset_id": asset.asset_id, 
                    "type": asset.type,
                    "relevance_score": getattr(val_info, 'relevance_score', 0) if val_info else 0,
                    "bonus_score": bonus,
                    "validator_suggestion": val_info.search_suggestion if val_info else "none", 
                    "content_snippet": asset.content[:300] + "..." if asset.type == 'text' else "[IMAGE]",
                })

        log_entry = {
            "step": step,
            "detailed_query_logs": [log for log in self.query_history_log if log['step'] == step],
            "search_actions": actions_dump,
            "new_assets_found": new_assets_count,
            "validations_performed_count": len(validations_dump),
            "meta_input_context": meta_input_snapshot, 
            "meta_analysis": meta_output.model_dump() if meta_output else None,
            "resulting_draft": self.latest_draft.model_dump() if self.latest_draft else None,
        }
        self.step_history.append(log_entry)

    def set_final_answer(self, text: str, think_trace: str = "", final_short_id_map: dict = {}):
        self.final_answer = text
        self.final_responder_think = think_trace
        self.final_short_id_map = final_short_id_map

    # ==============================================================================
    #  5. export
    # ==============================================================================

    def get_best_final_draft(self) -> Optional[DraftAnswer]:
        return self.draft_history[-1] if self.draft_history else None

    @property
    def latest_draft(self) -> Optional[DraftAnswer]:
        return self.draft_history[-1] if self.draft_history else None

    def export_full_state(self) -> dict:
        exported_assets = {}
        for aid, asset in self.assets.items():
            content = asset.content
            
            d = asset.model_dump()
            d['content'] = content
            d['meta_bonus'] = self.asset_bonus_scores.get(aid, 0.0)
            exported_assets[aid] = d

        return {
            "original_question": self.original_question,
            "final_responder_think": self.final_responder_think, 
            "final_answer": self.final_answer, 
            "final_short_id_map": self.final_short_id_map,
            "execution_trace": self.step_history,
            "final_assets_summary": exported_assets,
            "dataset_vibe_analysis_log": self.dataset_vibe_log,
            "final_validations_map": {k: v.model_dump() for k, v in self.validations.items()},
        }