import time
import re
import concurrent.futures
from typing import Dict, List, Optional, Tuple

from data_schemas import (
    Asset,
    Belief,
    SearchAction,
    Step2ActionOutput,
    MetaControllerOutput,
    MissingInformation,
)
from memory_system import MemorySystem
from llm_roles import LLMRoles
from tools import Toolbox
from config import (
    COGNITIVE_CONTEXT_TOKENS,
    COGNITIVE_MAX_ASSETS,
    FINAL_CONTEXT_TOKENS,
    FINAL_MAX_ASSETS,
    FINAL_MAX_BELIEFS,
    COGNITIVE_MAX_BELIEFS,
    MAX_RETRIEVAL_STEPS,
    VALIDATOR_MAX_WORKERS,
)

class IterativeAgent:
    def __init__(self, llm_roles: LLMRoles, dataset: str, retrieval_strategy: str = "independent"):
        self.roles = llm_roles
        self.top_n = 3
        self.toolbox = Toolbox(dataset=dataset, top_n=self.top_n) 
        self.max_steps = MAX_RETRIEVAL_STEPS
        
        # Global retrieval strategy: "union_rrf", "intersect", or "independent".
        allowed_strategies = {"union_rrf", "intersect", "independent"}
        normalized_strategy = str(retrieval_strategy).strip().lower()
        if normalized_strategy not in allowed_strategies:
            raise ValueError(
                f"Unsupported retrieval strategy: {retrieval_strategy}. "
                f"Expected one of {sorted(allowed_strategies)}."
            )
        self.retrieval_strategy = normalized_strategy

    def get_default_further_search_output(self) -> MetaControllerOutput:
        """兜底机制：当完全没有有效信息输入时使用"""
        return MetaControllerOutput(
            decision='FURTHER_SEARCH_NEEDED',
            missing_info_gaps=[
                MissingInformation(
                    description="Broader context and specific details related to the user's query.",
                    suggested_modality="both"
                )
            ],
            belief_state=[], 
            dataset_vibe_analysis="Initial exploration or dead end. No new signal detected.",
            next_search_actions=[],
            useful_asset_ids=[],
            extracted_answer=None
        )

    def _generate_bootstrap_actions(self, initial_question: str) -> List[SearchAction]:
        actions = []
        actions.append(SearchAction(modality="text", query=initial_question, top_n=self.top_n))
        actions.append(SearchAction(modality="image", query=initial_question, top_n=self.top_n))
        return actions

    def _normalize_base_id(self, raw_id: str) -> str:
        if not raw_id: return ""
        normalized = str(raw_id)
        while True:
            match = re.match(
                r"^(?:text|image|img|doc)_(.+)$",
                normalized,
                re.IGNORECASE,
            )
            if not match or match.group(1) == normalized:
                return normalized
            normalized = match.group(1)

    def _grounded_beliefs(self, beliefs: List, memory: MemorySystem) -> List:
        """Keep only beliefs whose provenance anchors resolve to stored raw assets."""
        grounded = []
        for belief in beliefs or []:
            source_ids = re.findall(
                r"S\d+-A\d+",
                str(getattr(belief, "source_id", "")).upper(),
            )
            if not source_ids:
                continue
            raw_ids = memory.resolve_short_ids(source_ids)
            if (
                len(raw_ids) == len(source_ids)
                and all(raw_id in memory.assets for raw_id in raw_ids)
                and all(memory.is_effectively_valid(raw_id) for raw_id in raw_ids)
            ):
                grounded.append(belief)
        return grounded

    def _sanitize_beliefs(
        self,
        beliefs: List[Belief],
        memory: MemorySystem,
        max_beliefs: int = COGNITIVE_MAX_BELIEFS,
    ) -> List[Belief]:
        """Provenance-filter, exact-deduplicate, and compact the belief state."""
        confidence_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        merged: Dict[str, Dict[str, object]] = {}

        for belief in self._grounded_beliefs(beliefs, memory):
            fact = " ".join(str(belief.fact).split()).strip()
            if not fact:
                continue
            key = re.sub(
                r"[\W_]+",
                " ",
                fact.casefold(),
                flags=re.UNICODE,
            ).strip() or fact.casefold()
            source_ids = re.findall(
                r"S\d+-A\d+",
                str(belief.source_id).upper(),
            )
            if not source_ids:
                continue

            entry = merged.setdefault(
                key,
                {
                    "fact": fact,
                    "source_ids": [],
                    "confidence": belief.confidence,
                },
            )
            for source_id in source_ids:
                if source_id not in entry["source_ids"]:
                    entry["source_ids"].append(source_id)
            if confidence_rank.get(belief.confidence, 0) > confidence_rank.get(
                str(entry["confidence"]),
                0,
            ):
                entry["confidence"] = belief.confidence

        compacted = [
            Belief(
                fact=str(entry["fact"]),
                source_id=", ".join(entry["source_ids"]),
                confidence=str(entry["confidence"]),
            )
            for entry in merged.values()
        ]

        def priority(belief: Belief):
            source_ids = re.findall(r"S\d+-A\d+", belief.source_id.upper())
            raw_ids = memory.resolve_short_ids(source_ids)
            best_score = max(
                (memory.get_utility_score(raw_id) for raw_id in raw_ids),
                default=0.0,
            )
            return (-confidence_rank.get(belief.confidence, 0), -best_score)

        compacted.sort(key=priority)
        return compacted[:max_beliefs]

    @staticmethod
    def _action_already_executed(
        action: SearchAction,
        executed_queries: set,
    ) -> bool:
        """Treat a `both` action as duplicate when both physical searches ran."""
        query_key = action.query.strip().casefold()
        if not query_key:
            return True
        if action.modality == "both":
            return (
                ("both", query_key) in executed_queries
                or (
                    ("text", query_key) in executed_queries
                    and ("image", query_key) in executed_queries
                )
            )
        return (
            (action.modality, query_key) in executed_queries
            or ("both", query_key) in executed_queries
        )

    @staticmethod
    def _expand_independent_actions(
        actions: List[SearchAction],
    ) -> List[SearchAction]:
        """Make `both` executable under independent retrieval."""
        expanded: List[SearchAction] = []
        for action in actions:
            if action.modality == "both":
                expanded.extend(
                    [
                        SearchAction(
                            modality="text",
                            query=action.query,
                            top_n=action.top_n,
                        ),
                        SearchAction(
                            modality="image",
                            query=action.query,
                            top_n=action.top_n,
                        ),
                    ]
                )
            else:
                expanded.append(action)
        return expanded

    @staticmethod
    def _select_intersect_with_ranked_backfill(
        doc_dict: Dict[str, dict],
        img_dict: Dict[str, dict],
        text_target: int,
        image_target: int,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """Select matched IDs first, then backfill each modality by its own rank.

        `intersect` is a *priority rule*, not a hard admission gate:

        1. Base IDs retrieved by both modalities are ranked by the sum of their
           text/image ranks and consumed first.
        2. Any unfilled text slots are backfilled from the remaining text
           ranking.
        3. Any unfilled image slots are backfilled from the remaining image
           ranking.

        Therefore, for text_target=image_target=k, the returned context contains
        up to k text assets and k image assets. A shortage of exact cross-modal
        matches never discards otherwise high-ranked modality-specific evidence.
        """
        text_target = max(0, int(text_target))
        image_target = max(0, int(image_target))

        matched_ids = sorted(
            set(doc_dict).intersection(img_dict),
            key=lambda base_id: (
                doc_dict[base_id]["rank"] + img_dict[base_id]["rank"],
                doc_dict[base_id]["rank"],
                img_dict[base_id]["rank"],
                base_id,
            ),
        )

        selected_text: List[str] = []
        selected_image: List[str] = []
        matched_selected: List[str] = []

        # Structural-prior stage: exact cross-modal matches have priority.
        for base_id in matched_ids:
            used = False
            if len(selected_text) < text_target:
                selected_text.append(base_id)
                used = True
            if len(selected_image) < image_target:
                selected_image.append(base_id)
                used = True
            if used:
                matched_selected.append(base_id)
            if (
                len(selected_text) >= text_target
                and len(selected_image) >= image_target
            ):
                break

        # Ranked modality-specific backfill.
        text_backfill: List[str] = []
        for base_id in sorted(
            doc_dict,
            key=lambda item: (doc_dict[item]["rank"], item),
        ):
            if len(selected_text) >= text_target:
                break
            if base_id in selected_text:
                continue
            selected_text.append(base_id)
            text_backfill.append(base_id)

        image_backfill: List[str] = []
        for base_id in sorted(
            img_dict,
            key=lambda item: (img_dict[item]["rank"], item),
        ):
            if len(selected_image) >= image_target:
                break
            if base_id in selected_image:
                continue
            selected_image.append(base_id)
            image_backfill.append(base_id)

        return (
            selected_text,
            selected_image,
            matched_selected,
            text_backfill,
            image_backfill,
        )

    def _fallback_actions_from_gaps(
        self,
        question: str,
        gaps: List[MissingInformation],
        executed_queries: set,
    ) -> List[SearchAction]:
        """Deterministic last-resort planning; never invents an answer or source."""
        actions: List[SearchAction] = []
        for gap in gaps[:2]:
            modalities = (
                ["text", "image"]
                if gap.suggested_modality == "both"
                else [gap.suggested_modality]
            )
            query_candidates = [
                f"{question} {gap.description}".strip(),
                gap.description.strip(),
            ]
            for modality in modalities:
                query = next(
                    (
                        candidate
                        for candidate in query_candidates
                        if candidate
                        and (modality, candidate.casefold())
                        not in executed_queries
                    ),
                    "",
                )
                if not query:
                    continue
                actions.append(
                    SearchAction(
                        modality=modality,
                        query=query,
                        top_n=4,
                    )
                )
                if len(actions) >= 2:
                    return actions
        return actions

    def _normalize_cognitive_decision(self, step1_out):
        """Enforce only structural consistency; never infer sufficiency from validator scores."""
        if step1_out.decision == "FURTHER_SEARCH_NEEDED":
            if not step1_out.missing_info_gaps:
                step1_out.missing_info_gaps = [
                    MissingInformation(
                        description=(
                            "Retrieve evidence that directly establishes the exact "
                            "answer-bearing entity, value, attribute, or relation "
                            "requested by the question."
                        ),
                        suggested_modality="both",
                    )
                ]
        else:
            # A terminal decision and an unresolved-gap list are contradictory.
            # Preserve the model's terminal decision and remove stale gaps.
            step1_out.missing_info_gaps = []
        return step1_out

    def _build_final_bundle(
        self,
        memory: MemorySystem,
        beliefs: List,
        max_assets: int = 6,
        max_tokens: int = 6000,
        max_beliefs: int = 6,
    ):
        """Build a provenance-closed final context under a fixed budget."""
        grounded_beliefs = [
            belief
            for belief in self._sanitize_beliefs(
                beliefs,
                memory,
                max_beliefs=max(max_beliefs * 2, max_beliefs),
            )
            if getattr(belief, "confidence", "LOW") in {"HIGH", "MEDIUM"}
        ]
        ranked_assets, _ = memory.elect_best_evidence_set(
            max_token_limit=max_tokens,
            max_assets=max_assets,
        )

        selected_assets = []
        selected_ids = set()
        retained_beliefs = []
        used_tokens = 0

        confidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

        def belief_priority(belief):
            source_ids = re.findall(
                r"S\d+-A\d+",
                str(getattr(belief, "source_id", "")).upper(),
            )
            raw_ids = memory.resolve_short_ids(source_ids)
            best_source_score = max(
                (memory.get_utility_score(raw_id) for raw_id in raw_ids),
                default=0.0,
            )
            return (
                confidence_rank.get(getattr(belief, "confidence", "LOW"), 2),
                -best_source_score,
            )

        grounded_beliefs = sorted(grounded_beliefs, key=belief_priority)

        # Reserve a tiny channel for the strongest first-round raw retrievals.
        # These assets are not promoted into beliefs unless they independently
        # satisfy the grounded-belief rules.
        for anchor_id in memory.initial_anchor_ids:
            if len(selected_assets) >= max_assets:
                break
            asset = memory.assets.get(anchor_id)
            if asset is None or anchor_id in selected_ids:
                continue
            cost = memory.estimate_asset_tokens(asset)
            if used_tokens + cost > max_tokens:
                continue
            selected_assets.append(asset)
            selected_ids.add(anchor_id)
            used_tokens += cost

        # Close every retained belief with all of its cited raw assets.
        for belief in grounded_beliefs:
            source_ids = re.findall(
                r"S\d+-A\d+",
                str(getattr(belief, "source_id", "")).upper(),
            )
            raw_ids = memory.resolve_short_ids(source_ids)
            new_assets = [
                memory.assets[raw_id]
                for raw_id in raw_ids
                if raw_id in memory.assets and raw_id not in selected_ids
            ]
            new_cost = sum(
                memory.estimate_asset_tokens(asset) for asset in new_assets
            )

            if len(selected_assets) + len(new_assets) > max_assets:
                continue
            if used_tokens + new_cost > max_tokens:
                continue

            selected_assets.extend(new_assets)
            selected_ids.update(asset.asset_id for asset in new_assets)
            used_tokens += new_cost
            retained_beliefs.append(belief)
            if len(retained_beliefs) >= max_beliefs:
                break

        # Fill the remaining budget with globally high-value validated assets.
        for asset in ranked_assets:
            if len(selected_assets) >= max_assets:
                break
            if asset.asset_id in selected_ids:
                continue

            cost = memory.estimate_asset_tokens(asset)
            if used_tokens + cost > max_tokens:
                continue

            selected_assets.append(asset)
            selected_ids.add(asset.asset_id)
            used_tokens += cost

        return selected_assets, retained_beliefs

    def run(self, initial_question: str) -> Tuple[Optional[str], dict]:
        print(f"\n🌀 [Cognitive Agent] Online Rollout Started: {initial_question}")
        print(f"⚙️ [Config] Retrieval Strategy: {self.retrieval_strategy.upper()}")
        start_time = time.time()
        
        self.roles.clear_memory()
        memory = MemorySystem(question=initial_question)
        next_actions = self._generate_bootstrap_actions(initial_question)
        
        prev_beliefs = []
        current_vibe = ""
        
        executed_queries_set = set()
        
        historical_short_ids = {} 
        
        step = 0
        while step < self.max_steps:
            step += 1
            print(f"\n--- 🔄 Cognitive Cycle {step}/{self.max_steps} ---")
            
            validator_thinks_this_step = {}
            cognitive_think_str = ""
            optimizer_think_str = ""
            
            current_actions = list(next_actions) 
            new_assets_this_step = []
            queries_info_this_step = [] 
            
            # ==============================================================================
            # A.(Execution Layer)
            # ==============================================================================
            
            if self.retrieval_strategy == "independent":
                # ------------------------------------------------------------------
                # Strategy 1: independent modality-specific retrieval.
                # ------------------------------------------------------------------
                current_actions = self._expand_independent_actions(current_actions)
                for action in current_actions:
                    action_key = (action.modality, action.query.strip().casefold())
                    if self._action_already_executed(action, executed_queries_set):
                        print(
                            f"⏭️ [Skipped] Duplicate query: "
                            f"({action.modality}) {action.query}"
                        )
                        memory.add_query_log(
                            step=step,
                            modality=action.modality,
                            query=action.query,
                            new_count=-1,
                            valid_count=0,
                            returned_count=0,
                            physical_modalities=[],
                            requested_top_n=action.top_n,
                            physical_top_n={},
                        )
                        continue

                    top_n_requested = action.top_n
                    print(
                        f"🔎 [Search] ({action.modality.upper()} | "
                        f"Top-{top_n_requested}) '{action.query}'"
                    )
                    temp_assets: List[Asset] = []
                    try:
                        if action.modality == "image":
                            results = self.toolbox.image_search(
                                action.query,
                                top_n=top_n_requested,
                            )
                        else:
                            results = self.toolbox.text_search(
                                action.query,
                                top_n=top_n_requested,
                            )

                        for result in results:
                            raw_id = (
                                result.get("asset_id")
                                or result.get("text_id")
                                or result.get("image_id")
                                or result.get("metadata", {}).get("doc_id")
                                or ""
                            )
                            if not raw_id:
                                continue
                            temp_assets.append(
                                Asset(
                                    asset_id=raw_id,
                                    type=result.get("type", action.modality),
                                    content=result.get("content", ""),
                                    source_queries=[action.query],
                                    metadata=result.get("metadata", {}),
                                )
                            )
                    except Exception as exc:
                        print(f"⚠️ Search error for '{action.query}': {exc}")
                        memory.add_query_log(
                            step=step,
                            modality=action.modality,
                            query=action.query,
                            new_count=0,
                            valid_count=0,
                            returned_count=0,
                            physical_modalities=[action.modality],
                            requested_top_n=action.top_n,
                            physical_top_n={action.modality: action.top_n},
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        # Infrastructure failures remain retryable.
                        continue

                    novel_assets = [
                        asset
                        for asset in temp_assets
                        if asset.asset_id not in memory.assets
                    ]
                    queries_info_this_step.append(
                        {
                            "action": action,
                            "new_count": len(novel_assets),
                            "novel_assets": novel_assets,
                            "all_assets": temp_assets,
                            "physical_modalities": [action.modality],
                            "requested_top_n": action.top_n,
                            "physical_top_n": {action.modality: action.top_n},
                        }
                    )
                    executed_queries_set.add(action_key)
                    memory.add_assets(temp_assets)
                    if step == 1 and temp_assets:
                        memory.register_initial_anchor(
                            temp_assets[0].asset_id,
                            action.modality,
                        )
                    new_assets_this_step.extend(novel_assets)

            else:
                # ------------------------------------------------------------------
                # Strategy 2: global late fusion.
                # `intersect` prioritizes cross-modal ID agreement, then backfills
                # each requested modality by its own ranking; `union_rrf` is union.
                # ------------------------------------------------------------------
                effective_actions: List[SearchAction] = []
                for action in current_actions:
                    action_key = (action.modality, action.query.strip().casefold())
                    if self._action_already_executed(action, executed_queries_set):
                        memory.add_query_log(
                            step=step,
                            modality=action.modality,
                            query=action.query,
                            new_count=-1,
                            valid_count=0,
                            returned_count=0,
                            physical_modalities=[],
                            requested_top_n=action.top_n,
                            physical_top_n={},
                        )
                        continue
                    effective_actions.append(action)

                requested_modalities = set()
                for action in effective_actions:
                    if action.modality == "both":
                        requested_modalities.update({"text", "image"})
                    else:
                        requested_modalities.add(action.modality)

                # Keep a separate target budget for each modality. For example,
                # bootstrap text top-3 + image top-3 means exactly T=3 / I=3,
                # not six matched base IDs (which could expand to 12 assets).
                text_target_n = min(
                    10,
                    sum(
                        action.top_n
                        for action in effective_actions
                        if action.modality in {"text", "both"}
                    ),
                )
                image_target_n = min(
                    10,
                    sum(
                        action.top_n
                        for action in effective_actions
                        if action.modality in {"image", "both"}
                    ),
                )
                total_top_n = min(
                    10,
                    sum(action.top_n for action in effective_actions),
                )
                is_single_modality = len(requested_modalities) == 1
                primary_modality = (
                    next(iter(requested_modalities))
                    if is_single_modality
                    else None
                )
                target_desc = (
                    f"text={text_target_n}, image={image_target_n}"
                    if self.retrieval_strategy == "intersect"
                    else str(total_top_n)
                )
                print(
                    f"🔎 [Global {self.retrieval_strategy.upper()}] "
                    f"Mode: {'Single-Modal Focus' if is_single_modality else 'Multi-Modal'} "
                    f"| Target Size: {target_desc}"
                )

                doc_dict: Dict[str, dict] = {}
                img_dict: Dict[str, dict] = {}

                def get_raw_id(result):
                    metadata = result.get("metadata", {}) or {}
                    return (
                        metadata.get("doc_id")
                        or result.get("asset_id")
                        or result.get("text_id")
                        or result.get("image_id")
                        or ""
                    )

                def merge_result(target, result, rank, query):
                    base_id = self._normalize_base_id(get_raw_id(result))
                    if not base_id:
                        return
                    if base_id not in target:
                        target[base_id] = {
                            "data": result,
                            "rank": rank,
                            "queries": [query],
                        }
                    else:
                        if query not in target[base_id]["queries"]:
                            target[base_id]["queries"].append(query)
                        if rank < target[base_id]["rank"]:
                            target[base_id]["data"] = result
                            target[base_id]["rank"] = rank

                successful_actions: List[SearchAction] = []
                for action in effective_actions:
                    query_str = action.query.strip()
                    pool_size = max(10, action.top_n * 3)
                    action_key = (action.modality, query_str.casefold())
                    cross_modal_probe = (
                        self.retrieval_strategy == "intersect"
                        and is_single_modality
                    )
                    physical_modalities = (
                        ["text", "image"]
                        if cross_modal_probe or action.modality == "both"
                        else [action.modality]
                    )
                    try:
                        local_docs = []
                        local_images = []
                        # Intersect may probe both pools even for a single-modality
                        # request so cross-modal agreement can be used as a ranking prior;
                        # union RRF follows the requested modality.
                        if cross_modal_probe or action.modality in {"text", "both"}:
                            local_docs = self.toolbox.text_search(
                                query_str,
                                top_n=pool_size,
                            )
                        if cross_modal_probe or action.modality in {"image", "both"}:
                            local_images = self.toolbox.image_search(
                                query_str,
                                top_n=pool_size,
                            )
                        for index, result in enumerate(local_docs, start=1):
                            merge_result(doc_dict, result, index, query_str)
                        for index, result in enumerate(local_images, start=1):
                            merge_result(img_dict, result, index, query_str)
                    except Exception as exc:
                        print(f"⚠️ Search error for '{query_str}': {exc}")
                        memory.add_query_log(
                            step=step,
                            modality=action.modality,
                            query=query_str,
                            new_count=0,
                            valid_count=0,
                            returned_count=0,
                            physical_modalities=physical_modalities,
                            requested_top_n=action.top_n,
                            physical_top_n={
                                modality: pool_size
                                for modality in physical_modalities
                            },
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        continue
                    executed_queries_set.add(action_key)
                    if (
                        action.modality == "both"
                        or (
                            self.retrieval_strategy == "intersect"
                            and is_single_modality
                        )
                    ):
                        executed_queries_set.add(("text", query_str.casefold()))
                        executed_queries_set.add(("image", query_str.casefold()))
                    else:
                        executed_queries_set.add(
                            (action.modality, query_str.casefold())
                        )
                    successful_actions.append(action)

                temp_assets: List[Asset] = []

                def append_asset_from_dict(
                    base_id: str,
                    source_dict: Dict[str, dict],
                    default_type: str,
                ) -> None:
                    if base_id not in source_dict:
                        return
                    entry = source_dict[base_id]
                    result = entry["data"]
                    temp_assets.append(
                        Asset(
                            asset_id=get_raw_id(result),
                            type=result.get("type", default_type),
                            content=result.get("content", ""),
                            source_queries=entry["queries"],
                            metadata=result.get("metadata", {}),
                        )
                    )

                if self.retrieval_strategy == "intersect":
                    (
                        selected_text_ids,
                        selected_image_ids,
                        matched_selected,
                        text_backfill,
                        image_backfill,
                    ) = self._select_intersect_with_ranked_backfill(
                        doc_dict=doc_dict,
                        img_dict=img_dict,
                        text_target=text_target_n,
                        image_target=image_target_n,
                    )

                    print(
                        "🔗 [Intersect] "
                        f"matched-first={len(matched_selected)} | "
                        f"text={len(selected_text_ids)}/{text_target_n} "
                        f"(+{len(text_backfill)} backfill) | "
                        f"image={len(selected_image_ids)}/{image_target_n} "
                        f"(+{len(image_backfill)} backfill)"
                    )

                    # Preserve structural matches at the front of the context.
                    selected_text_set = set(selected_text_ids)
                    selected_image_set = set(selected_image_ids)
                    emitted_text = set()
                    emitted_image = set()

                    for base_id in matched_selected:
                        if base_id in selected_text_set:
                            append_asset_from_dict(base_id, doc_dict, "text")
                            emitted_text.add(base_id)
                        if base_id in selected_image_set:
                            append_asset_from_dict(base_id, img_dict, "image")
                            emitted_image.add(base_id)

                    for base_id in selected_text_ids:
                        if base_id not in emitted_text:
                            append_asset_from_dict(base_id, doc_dict, "text")
                    for base_id in selected_image_ids:
                        if base_id not in emitted_image:
                            append_asset_from_dict(base_id, img_dict, "image")

                else:  # union_rrf
                    candidate_ids = set(doc_dict).union(img_dict)
                    rrf_scores = {}
                    for base_id in candidate_ids:
                        score = 0.0
                        if base_id in doc_dict:
                            score += 1.0 / (60.0 + doc_dict[base_id]["rank"])
                        if base_id in img_dict:
                            score += 1.0 / (60.0 + img_dict[base_id]["rank"])
                        rrf_scores[base_id] = score
                    sorted_union = sorted(
                        candidate_ids,
                        key=lambda base_id: rrf_scores[base_id],
                        reverse=True,
                    )

                    if is_single_modality:
                        primary_dict = (
                            doc_dict if primary_modality == "text" else img_dict
                        )
                        selected_base_ids = [
                            base_id
                            for base_id in sorted_union
                            if base_id in primary_dict
                        ][:total_top_n]
                    else:
                        selected_base_ids = sorted_union[:total_top_n]

                    for base_id in selected_base_ids:
                        if (
                            not is_single_modality
                            or primary_modality == "text"
                        ) and base_id in doc_dict:
                            append_asset_from_dict(base_id, doc_dict, "text")
                        if (
                            not is_single_modality
                            or primary_modality == "image"
                        ) and base_id in img_dict:
                            append_asset_from_dict(base_id, img_dict, "image")

                novel_assets = [
                    asset
                    for asset in temp_assets
                    if asset.asset_id not in memory.assets
                ]
                for action in successful_actions:
                    associated_all = [
                        asset
                        for asset in temp_assets
                        if action.query in asset.source_queries
                    ]
                    associated_novel = [
                        asset
                        for asset in novel_assets
                        if action.query in asset.source_queries
                    ]
                    physical_modalities = (
                        ["text", "image"]
                        if (
                            action.modality == "both"
                            or (
                                self.retrieval_strategy == "intersect"
                                and is_single_modality
                            )
                        )
                        else [action.modality]
                    )
                    queries_info_this_step.append(
                        {
                            "action": action,
                            "new_count": len(associated_novel),
                            "novel_assets": associated_novel,
                            "all_assets": associated_all,
                            "physical_modalities": physical_modalities,
                            "requested_top_n": action.top_n,
                            "physical_top_n": {
                                modality: max(10, action.top_n * 3)
                                for modality in physical_modalities
                            },
                        }
                    )

                memory.add_assets(temp_assets)
                if step == 1:
                    for modality in ("text", "image"):
                        top_asset = next(
                            (asset for asset in temp_assets if asset.type == modality),
                            None,
                        )
                        if top_asset is not None:
                            memory.register_initial_anchor(
                                top_asset.asset_id,
                                modality,
                            )
                new_assets_this_step.extend(novel_assets)

            print(f"📥 Environmental perception updated. Added {len(new_assets_this_step)} new assets.")

            # ==============================================================================
            # B. (Validation)
            # ==============================================================================
            unvalidated = memory.get_unvalidated_assets()
            validation_results_this_step = []
            
            if unvalidated:
                print(f"🕵️ [Validator] Auditing {len(unvalidated)} raw assets for signal-to-noise ratio...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=VALIDATOR_MAX_WORKERS) as executor:
                    futures = {executor.submit(self.roles.validate_asset, asset, initial_question): asset for asset in unvalidated}
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            res, v_think = f.result() 
                            validation_results_this_step.append(res)
                            validator_thinks_this_step[res.asset_id] = v_think 
                        except Exception as e:
                            print(f"⚠️ Validator error: {e}")

                memory.update_validation(validation_results_this_step)

            for q_info in queries_info_this_step:
                all_assets_for_query = q_info.get("all_assets", q_info["novel_assets"])
                valid_count = sum(
                    1
                    for asset in all_assets_for_query
                    if (
                        asset.asset_id in memory.validations
                        and memory.is_effectively_valid(asset.asset_id)
                    )
                )
                memory.add_query_log(
                    step=step,
                    modality=q_info["action"].modality,
                    query=q_info["action"].query,
                    new_count=q_info["new_count"],
                    valid_count=valid_count,
                    returned_count=len(all_assets_for_query),
                    physical_modalities=q_info.get("physical_modalities", []),
                    requested_top_n=q_info.get("requested_top_n"),
                    physical_top_n=q_info.get("physical_top_n", {}),
                )

            # ==============================================================================
            # C. (Cognitive Core)
            # ==============================================================================
            promoted_ids = memory.consume_promoted_asset_ids()
            if promoted_ids:
                # A rescued false negative gets exactly one new opportunity to
                # enter the cognitive state with its stable provenance ID.
                self.roles._seen_asset_ids.difference_update(promoted_ids)

            assets_to_review = memory.get_context_window_assets(
                max_tokens=COGNITIVE_CONTEXT_TOKENS,
                max_assets=COGNITIVE_MAX_ASSETS,
                exclude_ids=self.roles._seen_asset_ids,
                priority_ids=promoted_ids,
            )
            current_short_id_map = memory.bind_short_ids_for_step(step, assets_to_review)
            
            for raw_id, short_id in current_short_id_map.items():
                historical_short_ids.setdefault(raw_id, short_id)
            
            actually_viewed_assets = [] 

            if not assets_to_review and not prev_beliefs:
                print("⚠️ [Cognitive Core] No viable assets in context; planning a broader search.")
                cognitive_output = self.get_default_further_search_output()
                cognitive_think_str = "No incoming evidence; external evidence is still required."
                try:
                    step2_out, optimizer_think_str = self.roles.plan_next_actions(
                        question=initial_question,
                        dataset_vibe=cognitive_output.dataset_vibe_analysis,
                        step1_gaps=cognitive_output.missing_info_gaps,
                        beliefs=[],
                        past_queries_report=memory.get_search_history_report(),
                    )
                    cognitive_output.next_search_actions = step2_out.next_search_actions
                except Exception as exc:
                    print(f"⚠️ Planner error after empty retrieval: {exc}")
                    optimizer_think_str = "Planner failed after empty retrieval."
                if not cognitive_output.next_search_actions:
                    cognitive_output.next_search_actions = self._fallback_actions_from_gaps(
                        initial_question,
                        cognitive_output.missing_info_gaps,
                        executed_queries_set,
                    )
                    optimizer_think_str += " Deterministic gap fallback was used."
            else:
                print(f"🧠 [Cognitive Core - Step 1] Evaluating Epistemic State...")
                
                try:
                    step1_out, cognitive_think_str, actually_viewed_assets = (
                        self.roles.evaluate_cognitive_state(
                            question=initial_question,
                            assets_in_view=assets_to_review,
                            prev_beliefs=prev_beliefs,
                            past_queries_report=memory.get_search_history_report(),
                            short_id_map=current_short_id_map,
                            dataset_vibe=current_vibe,
                            validation_map=memory.validations,
                            evidence_signal_map={
                                asset.asset_id: {
                                    "unique_query_count": memory.get_unique_query_count(asset.asset_id),
                                    "repeat_bonus": memory.get_repeat_bonus(asset.asset_id),
                                    "admission_score": memory.get_admission_score(asset.asset_id),
                                    "utility_score": memory.get_utility_score(asset.asset_id),
                                    "is_initial_anchor": asset.asset_id in memory.initial_anchor_ids,
                                }
                                for asset in assets_to_review
                            },
                        )
                    )
                except Exception as exc:
                    # Preserve the last grounded state and continue evidence seeking.
                    # The same raw assets keep stable short IDs if reviewed again.
                    print(f"⚠️ Cognitive-state error: {exc}")
                    step1_out = self.get_default_further_search_output()
                    step1_out.belief_state = list(prev_beliefs)
                    step1_out.dataset_vibe_analysis = current_vibe
                    cognitive_think_str = (
                        "Cognitive-state generation failed; the prior grounded "
                        "belief state was preserved and retrieval continued."
                    )
                    actually_viewed_assets = []

                step1_out = self._normalize_cognitive_decision(step1_out)
                # The persistent state must contain only resolvable, non-duplicated beliefs.
                step1_out.belief_state = self._sanitize_beliefs(
                    step1_out.belief_state,
                    memory,
                )

                extracted_short_ids = []
                for belief in step1_out.belief_state:
                    extracted_short_ids.extend(
                        re.findall(r"S\d+-A\d+", belief.source_id.upper())
                    )

                # Confidence-aware belief bonuses are applied once in store_meta_output.
                prev_beliefs = step1_out.belief_state
                
                if hasattr(step1_out, 'dataset_vibe_analysis') and step1_out.dataset_vibe_analysis:
                    current_vibe = step1_out.dataset_vibe_analysis
                    memory.update_dataset_vibe(step, current_vibe)
                
                step2_out = None
                if step1_out.decision == 'FURTHER_SEARCH_NEEDED':
                    print(f"🎯 [Cognitive Core - Step 2] Action Planning...")
                    try:
                        step2_out, optimizer_think_str = self.roles.plan_next_actions(
                            question=initial_question,
                            dataset_vibe=step1_out.dataset_vibe_analysis,
                            step1_gaps=step1_out.missing_info_gaps,
                            beliefs=step1_out.belief_state,
                            past_queries_report=memory.get_search_history_report()
                        )
                    except Exception as e:
                        print(f"⚠️ Planner error: {e}")
                    if step2_out is None or not step2_out.next_search_actions:
                        fallback_actions = self._fallback_actions_from_gaps(
                            initial_question,
                            step1_out.missing_info_gaps,
                            executed_queries_set,
                        )
                        step2_out = Step2ActionOutput(
                            next_search_actions=fallback_actions
                        )
                        optimizer_think_str += " Deterministic gap fallback was used."

                cognitive_output = MetaControllerOutput(
                    meta_controller_think=cognitive_think_str,
                    decision=step1_out.decision,
                    belief_state=step1_out.belief_state,
                    useful_asset_ids=list(set(extracted_short_ids)),
                    missing_info_gaps=step1_out.missing_info_gaps,
                    dataset_vibe_analysis=getattr(step1_out, 'dataset_vibe_analysis', ""),
                    next_search_actions=step2_out.next_search_actions if step2_out else []
                )
                    
            memory.store_meta_output(cognitive_output, step)

            print(f"📝 [Decision] {cognitive_output.decision}")

            memory.log_step_details(
                step=step,
                search_actions=current_actions,
                new_assets_count=len(new_assets_this_step),
                meta_output=cognitive_output,
                elected_evidence=actually_viewed_assets, 
                validation_results=validation_results_this_step,
                meta_think=cognitive_think_str, 
                optimizer_think=optimizer_think_str, 
                validator_thinks=validator_thinks_this_step,
                current_short_id_map=current_short_id_map 
            )

            if step >= self.max_steps or cognitive_output.decision in ['REFERENCE_SUFFICIENT', 'NO_SEARCH_NEEDED']:
                break

            next_actions = getattr(cognitive_output, "next_search_actions", [])
            if not next_actions:
                next_actions = self._fallback_actions_from_gaps(
                    initial_question,
                    cognitive_output.missing_info_gaps,
                    executed_queries_set,
                )
            if not next_actions:
                print("⚠️ No executable non-duplicate search action remains.")
                break
            
        # ==============================================================================
        # 4. (Final Response Formulation)
        # ==============================================================================
        final_draft = memory.get_best_final_draft()
        final_evidence, final_beliefs = self._build_final_bundle(
            memory,
            prev_beliefs,
            max_assets=FINAL_MAX_ASSETS,
            max_tokens=FINAL_CONTEXT_TOKENS,
            max_beliefs=FINAL_MAX_BELIEFS,
        )

        final_decision = cognitive_output.decision
        final_short_id_map = {}

        if final_decision == "NO_SEARCH_NEEDED":
            print("↪️ Direct-answer branch: external evidence is unnecessary.")
            try:
                final_answer, final_think = self.roles.generate_fallback_response(
                    initial_question
                )
            except Exception as exc:
                print(f"⚠️ Direct-answer failure: {exc}")
                final_answer = "Information insufficient."
                final_think = "Direct-answer branch failed."

        elif not final_evidence:
            # An evidence-seeking state must not silently fall back to internal knowledge.
            print("⚠️ No validated evidence is available for evidence-grounded answering.")
            final_answer = "Information insufficient."
            final_think = "No final evidence bundle was available."

        else:
            print("✍️ Drafting final evidence-grounded response...")
            sf_counter = 1

            for asset in final_evidence:
                assigned_short_id = historical_short_ids.get(asset.asset_id)
                if assigned_short_id is None:
                    assigned_short_id = f"SF-A{sf_counter}"
                    sf_counter += 1
                # generate_final_response expects raw asset ID -> short ID.
                final_short_id_map[asset.asset_id] = assigned_short_id

            final_answer, final_think = self.roles.generate_final_response(
                question=initial_question,
                draft=final_draft,
                final_assets=final_evidence,
                short_id_map=final_short_id_map,
                prev_beliefs=final_beliefs,
            )
        
        memory.set_final_answer(text=final_answer, think_trace=final_think, final_short_id_map=final_short_id_map) 
        print(f"⏱️ Done. Time: {(time.time() - start_time):.2f}s")
        
        return final_answer, memory.export_full_state()