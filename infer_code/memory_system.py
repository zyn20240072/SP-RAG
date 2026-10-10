import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import (
    INITIAL_ANCHORS_PER_MODALITY,
    MAX_REPEAT_HIT_BONUS,
    REPEAT_HIT_BONUS,
    REPEAT_RESCUE_MIN_VALIDATOR_SCORE,
    VALIDATOR_PASS_SCORE,
)
from data_schemas import (
    Asset,
    Belief,
    DraftAnswer,
    MetaControllerOutput,
    SearchAction,
    ValidationResult,
)


class MemorySystem:
    """Asymmetric memory for raw evidence, epistemic state, and trajectory history."""

    def __init__(self, question: str):
        self.original_question = question
        self.assets: Dict[str, Asset] = {}
        self.validations: Dict[str, ValidationResult] = {}
        self.draft_history: List[DraftAnswer] = []

        self.query_history_log: List[Dict[str, Any]] = []
        self.dataset_vibe_log: List[Dict[str, Any]] = []

        self.asset_bonus_scores: Dict[str, float] = {}
        self.promoted_asset_ids: Set[str] = set()

        self.initial_anchor_ids: List[str] = []
        self._initial_anchor_count_by_modality: Dict[str, int] = {
            "text": 0,
            "image": 0,
        }

        self.step_history: List[Dict[str, Any]] = []

        self.final_answer: Optional[str] = None
        self.final_responder_think: Optional[str] = None
        self.final_short_id_map: Dict[str, str] = {}

        self.short_id_to_raw: Dict[str, str] = {}
        self.raw_id_to_short: Dict[str, str] = {}

    # ==============================================================================
    # 1. Basic state management
    # ==============================================================================

    @staticmethod
    def _normalize_query(query: str) -> str:
        compact = " ".join(str(query or "").split()).casefold()
        # Ignore casing, whitespace, and punctuation-only rewrites so they do
        # not manufacture artificial repeat-hit support.
        return re.sub(r"[\W_]+", " ", compact, flags=re.UNICODE).strip()

    def get_unique_query_count(self, asset_id: str) -> int:
        asset = self.assets.get(asset_id)
        if asset is None:
            return 0
        return len(
            {
                self._normalize_query(query)
                for query in asset.source_queries
                if self._normalize_query(query)
            }
        )

    def get_repeat_bonus(self, asset_id: str) -> float:
        """Bounded retrieval-consistency bonus; never mutates validator scores."""
        validation = self.validations.get(asset_id)
        if validation is None:
            return 0.0
        if validation.relevance_score < REPEAT_RESCUE_MIN_VALIDATOR_SCORE:
            return 0.0

        unique_query_count = self.get_unique_query_count(asset_id)
        return min(
            MAX_REPEAT_HIT_BONUS,
            REPEAT_HIT_BONUS * max(0, unique_query_count - 1),
        )

    def get_admission_score(self, asset_id: str) -> float:
        """Score used only for soft admission into the evidence pipeline."""
        validation = self.validations.get(asset_id)
        if validation is None:
            return 0.0
        return float(validation.relevance_score) + self.get_repeat_bonus(asset_id)

    def get_utility_score(self, asset_id: str) -> float:
        """Ranking utility after admission: validator + repeat + belief support."""
        return self.get_admission_score(asset_id) + self.asset_bonus_scores.get(
            asset_id,
            0.0,
        )

    def is_effectively_valid(self, asset_id: str) -> bool:
        validation = self.validations.get(asset_id)
        if validation is None:
            return False
        # Score-1 assets remain raw protected context at most; they cannot be
        # promoted into beliefs or normal evidence by repeated retrieval.
        if validation.relevance_score < REPEAT_RESCUE_MIN_VALIDATOR_SCORE:
            return False
        return self.get_admission_score(asset_id) >= VALIDATOR_PASS_SCORE

    def _update_promotion_state(self, asset_id: str, was_valid: bool) -> None:
        validation = self.validations.get(asset_id)
        rescued_from_below_threshold = (
            validation is not None
            and validation.relevance_score < VALIDATOR_PASS_SCORE
            and not was_valid
            and self.is_effectively_valid(asset_id)
        )
        if rescued_from_below_threshold:
            self.promoted_asset_ids.add(asset_id)
            raw_score = validation.relevance_score
            print(
                f"♻️ [Memory] Rescued asset {asset_id}: "
                f"validator={raw_score}, "
                f"unique_queries={self.get_unique_query_count(asset_id)}, "
                f"repeat_bonus={self.get_repeat_bonus(asset_id):.2f}, "
                f"admission={self.get_admission_score(asset_id):.2f}"
            )

    def add_assets(self, new_assets: List[Asset]) -> int:
        count = 0
        for asset in new_assets:
            if not asset.asset_id:
                continue

            if asset.asset_id not in self.assets:
                # Normalize and de-duplicate source queries at ingestion.
                unique_queries: List[str] = []
                seen_queries: Set[str] = set()
                for query in asset.source_queries:
                    normalized = self._normalize_query(query)
                    if normalized and normalized not in seen_queries:
                        unique_queries.append(" ".join(str(query).split()))
                        seen_queries.add(normalized)
                asset.source_queries = unique_queries
                self.assets[asset.asset_id] = asset
                count += 1
                continue

            existing_asset = self.assets[asset.asset_id]
            was_valid = self.is_effectively_valid(asset.asset_id)
            existing_query_keys = {
                self._normalize_query(query)
                for query in existing_asset.source_queries
                if self._normalize_query(query)
            }

            for query in asset.source_queries:
                normalized = self._normalize_query(query)
                if not normalized or normalized in existing_query_keys:
                    continue
                existing_asset.source_queries.append(" ".join(str(query).split()))
                existing_query_keys.add(normalized)

            self._update_promotion_state(asset.asset_id, was_valid)
        return count

    def update_validation(self, results: List[ValidationResult]) -> None:
        """Store validator outputs and detect same-step repeat-hit promotion."""
        for result in results:
            was_valid = self.is_effectively_valid(result.asset_id)
            self.validations[result.asset_id] = result
            self._update_promotion_state(result.asset_id, was_valid)

    def consume_promoted_asset_ids(self) -> Set[str]:
        promoted = set(self.promoted_asset_ids)
        self.promoted_asset_ids.clear()
        return promoted

    def register_initial_anchor(
        self,
        asset_id: str,
        modality: str,
        max_per_modality: int = INITIAL_ANCHORS_PER_MODALITY,
    ) -> None:
        """Protect a tiny first-round top-ranked raw-evidence channel."""
        if asset_id not in self.assets or max_per_modality <= 0:
            return
        normalized_modality = "image" if modality == "image" else "text"
        if self._initial_anchor_count_by_modality[normalized_modality] >= max_per_modality:
            return
        if asset_id in self.initial_anchor_ids:
            return
        self.initial_anchor_ids.append(asset_id)
        self._initial_anchor_count_by_modality[normalized_modality] += 1

    def get_unvalidated_assets(self) -> List[Asset]:
        return [
            asset
            for asset in self.assets.values()
            if asset.asset_id not in self.validations
        ]

    def get_past_queries_set(self) -> Set[Tuple[str, str]]:
        return {
            (log.get("modality", "text"), log["query"])
            for log in self.query_history_log
        }

    def get_past_queries_list(self) -> List[str]:
        return [log["query"] for log in self.query_history_log]

    def add_query_log(
        self,
        step: int,
        modality: str,
        query: str,
        new_count: int,
        valid_count: int = 0,
        returned_count: Optional[int] = None,
        physical_modalities: Optional[List[str]] = None,
        requested_top_n: Optional[int] = None,
        physical_top_n: Optional[Dict[str, int]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.query_history_log.append(
            {
                "step": step,
                "modality": modality,
                "query": query,
                "new_count": new_count,
                "valid_count": valid_count,
                "returned_count": (
                    max(new_count, 0) if returned_count is None else returned_count
                ),
                "physical_modalities": physical_modalities or [],
                "requested_top_n": requested_top_n,
                "physical_top_n": physical_top_n or {},
                "error": error,
            }
        )

    def update_dataset_vibe(self, step: int, vibe: str) -> None:
        if vibe:
            self.dataset_vibe_log.append({"step": step, "vibe": vibe})

    def get_search_history_report(self) -> str:
        if not self.query_history_log:
            return "No searches have been performed yet."

        lines: List[str] = []
        for log in self.query_history_log:
            new_count = log["new_count"]
            valid_count = log.get("valid_count", 0)

            returned_count = log.get("returned_count", max(new_count, 0))
            if log.get("error"):
                status = f"⚠️ Retrieval error: {log['error']}"
            elif new_count < 0:
                status = "⏭️ Skipped duplicate query"
            elif returned_count == 0:
                status = "❌ No retrieval results"
            elif new_count == 0:
                status = (
                    f"↩️ Reconfirmed {returned_count} previously seen assets"
                )
            elif valid_count == 0:
                status = (
                    f"⚠️ Low-yield (Found {new_count} new assets, "
                    "but none scored above 2)"
                )
            else:
                status = f"✅ Effective (Found {valid_count} useful new assets)"

            lines.append(
                f"[Step {log['step']}] [{log['modality'].upper()}] "
                f"Query: \"{log['query']}\" -> {status}"
            )
        return "\n".join(lines)

    # ==============================================================================
    # 2. Short-ID routing
    # ==============================================================================

    def bind_short_ids_for_step(
        self,
        step: int,
        assets: List[Asset],
    ) -> Dict[str, str]:
        """Return a raw_asset_id -> step_short_id mapping."""
        raw_to_short: Dict[str, str] = {}
        next_index = 1
        for asset in assets:
            short_id = self.raw_id_to_short.get(asset.asset_id)
            if short_id is None:
                while f"S{step}-A{next_index}" in self.short_id_to_raw:
                    next_index += 1
                short_id = f"S{step}-A{next_index}"
                next_index += 1
                self.raw_id_to_short[asset.asset_id] = short_id
                self.short_id_to_raw[short_id] = asset.asset_id
            raw_to_short[asset.asset_id] = short_id
        return raw_to_short

    def resolve_short_ids(self, short_ids: List[str]) -> List[str]:
        """Resolve short IDs into raw asset IDs, preserving input order."""
        raw_ids: List[str] = []
        for short_id in short_ids:
            raw_id = self.short_id_to_raw.get(short_id)
            if raw_id is None:
                print(f"⚠️ [Memory] Invalid short-ID citation: '{short_id}'")
                continue
            raw_ids.append(raw_id)
        return raw_ids

    # ==============================================================================
    # 3. Evidence ranking and context windows
    # ==============================================================================

    @staticmethod
    def estimate_asset_tokens(asset: Asset) -> int:
        """Cheap deterministic budget estimate used consistently across models."""
        if asset.type == "image":
            return 1500
        return max(1, int(len(asset.content) / 2))

    def boost_assets(
        self,
        asset_ids: List[str],
        bonus: float = 0.5,
        max_bonus: float = 1.5,
    ) -> None:
        """Bound belief-citation bonuses so they cannot dominate evidence ranking."""
        for asset_id in asset_ids:
            updated = self.asset_bonus_scores.get(asset_id, 0.0) + bonus
            self.asset_bonus_scores[asset_id] = min(updated, max_bonus)

    def get_context_window_assets(
        self,
        max_tokens: int = 4000,
        max_assets: int = 8,
        exclude_ids: Optional[Set[str]] = None,
        priority_ids: Optional[Set[str]] = None,
    ) -> List[Asset]:
        exclude_ids = exclude_ids or set()
        priority_ids = priority_ids or set()

        candidates: List[Tuple[int, float, Asset]] = []
        for asset_id in self.validations:
            if asset_id in exclude_ids:
                continue
            if not self.is_effectively_valid(asset_id):
                continue

            asset = self.assets.get(asset_id)
            if asset is not None:
                candidates.append(
                    (
                        1 if asset_id in priority_ids else 0,
                        self.get_utility_score(asset_id),
                        asset,
                    )
                )

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected_assets: List[Asset] = []
        used_tokens = 0
        for _, _, asset in candidates:
            if len(selected_assets) >= max_assets:
                break
            cost = self.estimate_asset_tokens(asset)
            if used_tokens + cost > max_tokens:
                continue
            selected_assets.append(asset)
            used_tokens += cost

        if selected_assets:
            return selected_assets

        # High-recall fallback is intentionally small. It lets the cognitive
        # policy inspect retrieval failures, but low-score assets still cannot
        # survive belief sanitation unless later rescued.
        unseen_assets = [
            asset
            for asset in self.assets.values()
            if asset.asset_id not in exclude_ids
        ]

        def fallback_priority(asset: Asset) -> Tuple[int, float, float, int]:
            validation = self.validations.get(asset.asset_id)
            raw_validator_score = (
                float(validation.relevance_score) if validation else 0.0
            )
            metadata_score = float(asset.metadata.get("score", 0.0) or 0.0)
            metadata_hits = int(asset.metadata.get("hit", 0) or 0)
            return (
                1 if asset.asset_id in self.initial_anchor_ids else 0,
                raw_validator_score,
                metadata_score,
                metadata_hits,
            )

        unseen_assets.sort(key=fallback_priority, reverse=True)
        fallback_assets: List[Asset] = []
        used_tokens = 0
        for asset in unseen_assets:
            cost = self.estimate_asset_tokens(asset)
            if used_tokens + cost > max_tokens:
                continue
            fallback_assets.append(asset)
            used_tokens += cost
            if len(fallback_assets) >= min(3, max_assets):
                break
        return fallback_assets

    def elect_best_evidence_set(
        self,
        max_token_limit: int = 6000,
        max_assets: int = 6,
    ) -> Tuple[List[Asset], str]:
        if not self.validations:
            return [], "NO_VALIDATED_DATA"

        candidates: List[ValidationResult] = []
        for validation in self.validations.values():
            if self.is_effectively_valid(validation.asset_id):
                candidates.append(validation)

        fallback_mode = False
        if not candidates:
            fallback_mode = True
            candidates = sorted(
                self.validations.values(),
                key=lambda item: self.get_utility_score(item.asset_id),
                reverse=True,
            )[:5]

        def evidence_score(validation: ValidationResult) -> float:
            return self.get_utility_score(validation.asset_id)

        sorted_validations = sorted(
            candidates,
            key=evidence_score,
            reverse=True,
        )

        selected_assets: List[Asset] = []
        used_tokens = 0
        for validation in sorted_validations:
            if len(selected_assets) >= max_assets:
                break

            asset = self.assets.get(validation.asset_id)
            if asset is None:
                continue

            cost = self.estimate_asset_tokens(asset)
            if used_tokens + cost > max_token_limit:
                continue

            selected_assets.append(asset)
            used_tokens += cost

        if fallback_mode:
            hint = (
                f"LOW_SCORE_FALLBACK: selected {len(selected_assets)} assets "
                f"within {used_tokens}/{max_token_limit} estimated tokens."
            )
        else:
            hint = (
                f"Selected {len(selected_assets)} assets within "
                f"{used_tokens}/{max_token_limit} estimated tokens."
            )
        return selected_assets, hint

    # ==============================================================================
    # 4. State update and logs
    # ==============================================================================

    def store_meta_output(self, meta_output: MetaControllerOutput, step: int) -> None:
        status_map = {
            "REFERENCE_SUFFICIENT": "COMPLETE",
            "FURTHER_SEARCH_NEEDED": "PARTIAL",
            "NO_SEARCH_NEEDED": "COMPLETE",
        }
        confidence_bonus = {
            "HIGH": 0.3,
            "MEDIUM": 0.2,
            "LOW": 0.1,
        }

        gaps = [gap.description for gap in meta_output.missing_info_gaps]
        facts: List[str] = []

        for belief in meta_output.belief_state:
            facts.append(belief.fact)
            short_ids = re.findall(
                r"S\d+-A\d+",
                str(belief.source_id).upper(),
            )
            raw_ids = self.resolve_short_ids(short_ids)
            if raw_ids:
                self.boost_assets(
                    raw_ids,
                    bonus=confidence_bonus.get(belief.confidence, 0.0),
                )

        if facts:
            draft_text = "\n[Accumulated Facts]: " + " | ".join(facts)
        else:
            draft_text = "\n[Accumulated Facts]: No concrete facts extracted yet."

        self.draft_history.append(
            DraftAnswer(
                current_answer=draft_text,
                status=status_map.get(meta_output.decision, "PARTIAL"),
                open_gaps=gaps,
            )
        )

    def log_step_details(
        self,
        step: int,
        search_actions: List[Any],
        new_assets_count: int,
        meta_output: Optional[MetaControllerOutput],
        elected_evidence: Optional[List[Asset]] = None,
        validation_results: Optional[List[ValidationResult]] = None,
        meta_think: str = "",
        optimizer_think: str = "",
        validator_thinks: Optional[Dict[str, str]] = None,
        current_short_id_map: Optional[Dict[str, str]] = None,
    ) -> None:
        actions_dump = [
            action.model_dump() if hasattr(action, "model_dump") else str(action)
            for action in search_actions
        ]
        validations_dump = [
            validation.model_dump()
            for validation in (validation_results or [])
        ]

        meta_input_snapshot: List[Dict[str, Any]] = []
        for asset in elected_evidence or []:
            validation = self.validations.get(asset.asset_id)
            belief_bonus = self.asset_bonus_scores.get(asset.asset_id, 0.0)
            repeat_bonus = self.get_repeat_bonus(asset.asset_id)
            short_id = (
                current_short_id_map.get(asset.asset_id, "?")
                if current_short_id_map
                else "?"
            )
            meta_input_snapshot.append(
                {
                    "short_id": short_id,
                    "asset_id": asset.asset_id,
                    "type": asset.type,
                    "relevance_score": (
                        validation.relevance_score if validation else 0
                    ),
                    "repeat_bonus": repeat_bonus,
                    "belief_bonus": belief_bonus,
                    "admission_score": self.get_admission_score(asset.asset_id),
                    "utility_score": self.get_utility_score(asset.asset_id),
                    "unique_query_count": self.get_unique_query_count(asset.asset_id),
                    "effectively_valid": self.is_effectively_valid(asset.asset_id),
                    "is_initial_anchor": asset.asset_id in self.initial_anchor_ids,
                    "validator_suggestion": (
                        validation.search_suggestion if validation else "none"
                    ),
                    "content_snippet": (
                        asset.content[:300] + "..."
                        if asset.type == "text"
                        else "[IMAGE]"
                    ),
                }
            )

        self.step_history.append(
            {
                "step": step,
                "detailed_query_logs": [
                    log for log in self.query_history_log if log["step"] == step
                ],
                "search_actions": actions_dump,
                "new_assets_found": new_assets_count,
                "validations_performed_count": len(validations_dump),
                "meta_input_context": meta_input_snapshot,
                "meta_analysis": (
                    meta_output.model_dump() if meta_output else None
                ),
                "resulting_draft": (
                    self.latest_draft.model_dump() if self.latest_draft else None
                ),
                "think_traces": {
                    "meta_controller": meta_think,
                    "query_optimizer": optimizer_think,
                    "validators": validator_thinks or {},
                },
            }
        )

    def set_final_answer(
        self,
        text: str,
        think_trace: str = "",
        final_short_id_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.final_answer = text
        self.final_responder_think = think_trace
        self.final_short_id_map = final_short_id_map or {}

    # ==============================================================================
    # 5. Export
    # ==============================================================================

    def get_best_final_draft(self) -> Optional[DraftAnswer]:
        return self.draft_history[-1] if self.draft_history else None

    @property
    def latest_draft(self) -> Optional[DraftAnswer]:
        return self.draft_history[-1] if self.draft_history else None

    def export_full_state(self) -> Dict[str, Any]:
        exported_assets: Dict[str, Any] = {}
        for asset_id, asset in self.assets.items():
            dumped = asset.model_dump()
            dumped["validator_score"] = (
                self.validations[asset_id].relevance_score
                if asset_id in self.validations
                else None
            )
            dumped["unique_query_count"] = self.get_unique_query_count(asset_id)
            dumped["repeat_bonus"] = self.get_repeat_bonus(asset_id)
            dumped["belief_bonus"] = self.asset_bonus_scores.get(asset_id, 0.0)
            dumped["admission_score"] = self.get_admission_score(asset_id)
            dumped["utility_score"] = self.get_utility_score(asset_id)
            dumped["effectively_valid"] = self.is_effectively_valid(asset_id)
            dumped["is_initial_anchor"] = asset_id in self.initial_anchor_ids
            exported_assets[asset_id] = dumped

        return {
            "original_question": self.original_question,
            "final_responder_think": self.final_responder_think,
            "final_answer": self.final_answer,
            "final_short_id_map": self.final_short_id_map,
            "final_selected_asset_ids": list(self.final_short_id_map.keys()),
            "initial_anchor_ids": list(self.initial_anchor_ids),
            "execution_trace": self.step_history,
            "all_assets_summary": exported_assets,
            # Backward-compatible alias retained for downstream analysis scripts.
            "final_assets_summary": exported_assets,
            "dataset_vibe_analysis_log": self.dataset_vibe_log,
            "final_validations_map": {
                key: value.model_dump()
                for key, value in self.validations.items()
            },
        }
