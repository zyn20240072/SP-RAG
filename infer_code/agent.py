import time
import re
import concurrent.futures
from typing import Tuple, Optional, List, Dict

from data_schemas import Asset, SearchAction, MetaControllerOutput, MissingInformation
from memory_system import MemorySystem
from llm_roles import LLMRoles
from tools import Toolbox

class IterativeAgent:
    def __init__(self, llm_roles: LLMRoles, dataset: str, retrieval_strategy: str = "independent"):
        self.roles = llm_roles
        self.top_n = 3
        self.toolbox = Toolbox(dataset=dataset, top_n=self.top_n) 
        self.max_steps = 4 
        
        # "intersect", "independent"
        self.retrieval_strategy = retrieval_strategy

    def get_default_further_search_output(self) -> MetaControllerOutput:
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
        actions.append(SearchAction(modality="text", query=initial_question, purpose="EXPLORATION", top_n=self.top_n))
        actions.append(SearchAction(modality="image", query=initial_question, purpose="EXPLORATION", top_n=self.top_n))
        return actions

    def _normalize_base_id(self, raw_id: str) -> str:
        if not raw_id: return ""
        s = str(raw_id)
        match = re.match(r'^(?:text|image|img|doc)_(.+)$', s, re.IGNORECASE)
        if match:
            return match.group(1)
        return s

    def run(self, initial_question: str) -> Tuple[Optional[str], dict]:
        print(f"\n[Cognitive Agent] Online Rollout Started: {initial_question}")
        print(f"[Config] Retrieval Strategy: {self.retrieval_strategy.upper()}")
        start_time = time.time()
        
        self.roles.clear_memory()
        memory = MemorySystem(question=initial_question)
        next_actions = self._generate_bootstrap_actions(initial_question)
        
        prev_beliefs = [] 
        past_queries_tracker = [f"[Step 0] TEXT (Top-3): '{initial_question}'", f"[Step 0] IMAGE (Top-3): '{initial_question}'"] 
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
            # A. (Execution Layer)
            # ==============================================================================
            
            if self.retrieval_strategy == "independent":
                # ------------------------------------------------------------------
                # (Independent)
                # ------------------------------------------------------------------
                for action in current_actions:
                    action_key = (action.modality, action.query.strip())
                    if action_key in executed_queries_set: 
                        print(f"⏭️ [Skipped] Duplicate query: ({action.modality}) {action.query}")
                        memory.add_query_log(step=step, modality=action.modality, query=action.query, new_count=-1, valid_count=0)
                        continue
                    
                    top_n_requested = getattr(action, 'top_n', 3) 
                    print(f"🔎 [Search] ({action.modality.upper()} | Top-{top_n_requested}) '{action.query}'")
                    past_queries_tracker.append(f"[Step {step}] {action.modality.upper()} (Top-{top_n_requested}): '{action.query}'")
                    
                    temp_assets = []
                    try:
                        if action.modality == 'image':
                            results = self.toolbox.image_search(action.query, top_n=top_n_requested)
                        else:
                            results = self.toolbox.text_search(action.query, top_n=top_n_requested)
                            
                        for r in results:
                            raw_id = r.get('asset_id') or r.get('text_id') or r.get('image_id') or r.get('metadata', {}).get('doc_id') or ""
                            temp_assets.append(Asset(
                                asset_id=raw_id, 
                                type=r.get('type', action.modality), 
                                content=r.get('content', ''), 
                                source_queries=[action.query], 
                                metadata=r.get('metadata', {})
                            ))
                    except Exception as e: 
                        print(f"⚠️ Search error for '{action.query}': {e}")

                    novel_assets = [a for a in temp_assets if a.asset_id not in memory.assets]
                    queries_info_this_step.append({
                        "action": action, "new_count": len(novel_assets), "novel_assets": novel_assets
                    })

                    executed_queries_set.add(action_key)
                    memory.add_assets(temp_assets)
                    new_assets_this_step.extend(temp_assets)

            else:
                # ------------------------------------------------------------------
                # (Global Late Fusion)
                # ------------------------------------------------------------------
                requested_modalities = set()
                total_top_n = 0
                for action in current_actions:
                    mod = str(action.modality).strip().lower()
                    if mod in ["text", "image"]: requested_modalities.add(mod)
                    elif mod == "both": requested_modalities.update(["text", "image"])
                    total_top_n += getattr(action, 'top_n', 3)
                total_top_n = min(10, total_top_n)
                is_single_modality = (len(requested_modalities) == 1)
                primary_modality = list(requested_modalities)[0] if is_single_modality else None
                
                print(f"[Global {self.retrieval_strategy.upper()}] Mode: {'Single-Modal Focus' if is_single_modality else 'Multi-Modal Exact'} | Target Size: {total_top_n}")
                
                doc_dict = {}
                img_dict = {}

                def get_raw_id(r):
                    meta = r.get('metadata', {})
                    if meta and 'doc_id' in meta: return meta['doc_id']
                    return r.get('asset_id') or r.get('text_id') or r.get('image_id') or ''

                for action in current_actions:
                    query_str = action.query.strip()
                    modality_req = str(action.modality).strip().lower()
                    pool_size = max(10, getattr(action, 'top_n', 3) * 3)

                    if not query_str: continue

                    action_key = (modality_req, query_str)
                    if action_key in executed_queries_set: continue
                    
                    executed_queries_set.add(action_key)
                    past_queries_tracker.append(f"[Step {step}] {modality_req.upper()} (Pool-{pool_size}): '{query_str}'")

                    try:
                        if is_single_modality:
                            docs_res = self.toolbox.text_search(query_str, top_n=pool_size)
                            imgs_res = self.toolbox.image_search(query_str, top_n=pool_size)
                            
                            for idx, d in enumerate(docs_res):
                                bid = self._normalize_base_id(get_raw_id(d))
                                if bid not in doc_dict or (idx + 1) < doc_dict[bid]["rank"]:
                                    doc_dict[bid] = {"data": d, "rank": idx + 1, "query": query_str}
                            for idx, img in enumerate(imgs_res):
                                bid = self._normalize_base_id(get_raw_id(img))
                                if bid not in img_dict or (idx + 1) < img_dict[bid]["rank"]:
                                    img_dict[bid] = {"data": img, "rank": idx + 1, "query": query_str}
                        else:
                            if modality_req in ["text", "both"]:
                                docs_res = self.toolbox.text_search(query_str, top_n=pool_size)
                                for idx, d in enumerate(docs_res):
                                    bid = self._normalize_base_id(get_raw_id(d))
                                    if bid not in doc_dict or (idx + 1) < doc_dict[bid]["rank"]:
                                        doc_dict[bid] = {"data": d, "rank": idx + 1, "query": query_str}
                            if modality_req in ["image", "both"]:
                                imgs_res = self.toolbox.image_search(query_str, top_n=pool_size)
                                for idx, img in enumerate(imgs_res):
                                    bid = self._normalize_base_id(get_raw_id(img))
                                    if bid not in img_dict or (idx + 1) < img_dict[bid]["rank"]:
                                        img_dict[bid] = {"data": img, "rank": idx + 1, "query": query_str}
                    except Exception as e:
                        print(f"⚠️ Search error: {e}")

                if self.retrieval_strategy == "intersect":
                    intersect_ids = set(doc_dict.keys()).intersection(set(img_dict.keys()))
                    
                    sorted_intersect = sorted(list(intersect_ids), key=lambda x: doc_dict[x]["rank"] + img_dict[x]["rank"])
                    
                    remaining_img_ids = [k for k in img_dict.keys() if k not in intersect_ids]
                    remaining_doc_ids = [k for k in doc_dict.keys() if k not in intersect_ids]
                    
                    remaining_img_ids.sort(key=lambda x: img_dict[x]["rank"])
                    remaining_doc_ids.sort(key=lambda x: doc_dict[x]["rank"])

                    backfill_ids = []
                    max_len = max(len(remaining_img_ids), len(remaining_doc_ids))
                    for i in range(max_len):
                        if i < len(remaining_img_ids): backfill_ids.append(remaining_img_ids[i])
                        if i < len(remaining_doc_ids): backfill_ids.append(remaining_doc_ids[i])
                    
                    sorted_union = sorted_intersect + [x for x in backfill_ids if x not in sorted_intersect]

                elif self.retrieval_strategy == "union_rrf":
                    all_ids = set(doc_dict.keys()).union(set(img_dict.keys()))
                    rrf_scores = {}
                    for b_id in all_ids:
                        score = 0.0
                        if b_id in doc_dict: score += 1.0 / (60.0 + doc_dict[b_id]["rank"])
                        if b_id in img_dict: score += 1.0 / (60.0 + img_dict[b_id]["rank"])
                        rrf_scores[b_id] = score
                    sorted_union = sorted(list(all_ids), key=lambda x: rrf_scores[x], reverse=True)
                else:
                    sorted_union = []

                selected_base_ids = []
                
                if is_single_modality:
                    primary_dict = doc_dict if primary_modality == "text" else img_dict
                    count = 0
                    for sid in sorted_union:
                        if sid in primary_dict:
                            selected_base_ids.append(sid)
                            count += 1
                            if count >= total_top_n: break
                    
                    if len(selected_base_ids) < total_top_n:
                        needed = total_top_n - len(selected_base_ids)
                        remaining_ids = [k for k in primary_dict.keys() if k not in selected_base_ids]
                        remaining_ids = sorted(remaining_ids, key=lambda x: primary_dict[x]["rank"])
                        selected_base_ids.extend(remaining_ids[:needed])
                else:
                    selected_base_ids = sorted_union[:total_top_n]
                    
                temp_assets = []
                for sid in selected_base_ids:
                    if sid in doc_dict: 
                        d = doc_dict[sid]
                        temp_assets.append(Asset(
                            asset_id=get_raw_id(d["data"]), 
                            type=d["data"].get('type', 'text'), 
                            content=d["data"].get('content', ''), 
                            source_queries=[d["query"]], 
                            metadata=d["data"].get('metadata', {})
                        ))
                    if sid in img_dict: 
                        img = img_dict[sid]
                        temp_assets.append(Asset(
                            asset_id=get_raw_id(img["data"]), 
                            type=img["data"].get('type', 'image'), 
                            content=img["data"].get('content', ''), 
                            source_queries=[img["query"]], 
                            metadata=img["data"].get('metadata', {})
                        ))

                novel_assets = [a for a in temp_assets if a.asset_id not in memory.assets]
                joint_action = SearchAction(modality="both", query="[Global Fusion]", purpose="EXPLORATION", top_n=total_top_n)
                queries_info_this_step.append({
                    "action": joint_action, "new_count": len(novel_assets), "novel_assets": novel_assets
                })

                memory.add_assets(temp_assets)
                new_assets_this_step.extend(temp_assets)

            print(f"Environmental perception updated. Added {len(new_assets_this_step)} new assets.")

            # ==============================================================================
            # B. (Validation)
            # ==============================================================================
            unvalidated = memory.get_unvalidated_assets()
            validation_results_this_step = []
            
            if unvalidated:
                print(f"🕵️ [Validator] Auditing {len(unvalidated)} raw assets for signal-to-noise ratio...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
                valid_count = sum(1 for asset in q_info["novel_assets"] if memory.validations.get(asset.asset_id, None) and memory.validations[asset.asset_id].relevance_score > 2)
                memory.add_query_log(step=step, modality=q_info["action"].modality, query=q_info["action"].query, new_count=q_info["new_count"], valid_count=valid_count)

            # ==============================================================================
            # C. (Cognitive Core)
            # ==============================================================================
            assets_to_review = memory.get_context_window_assets(max_tokens=4800, exclude_ids=self.roles._seen_asset_ids)
            current_short_id_map = memory.bind_short_ids_for_step(step, assets_to_review)
            
            for short_id, item in current_short_id_map.items():
                raw_id = getattr(item, 'asset_id', item) 
                if short_id not in historical_short_ids:
                    historical_short_ids[short_id] = raw_id
            
            actually_viewed_assets = [] 

            if not assets_to_review and not prev_beliefs:
                print(f"⚠️ [Cognitive Core] Sensory deprivation. No viable assets in context.")
                cognitive_output = self.get_default_further_search_output()
                cognitive_think_str = "No incoming data."
                optimizer_think_str = "Fallback."
            else:
                print(f"[Cognitive Core - Step 1] Evaluating Epistemic State...")
                
                step1_out, cognitive_think_str, actually_viewed_assets = self.roles.evaluate_cognitive_state(
                    question=initial_question, 
                    assets_in_view=assets_to_review,
                    prev_beliefs=prev_beliefs,  
                    past_queries_report="\n".join(past_queries_tracker),
                    short_id_map=current_short_id_map,
                    dataset_vibe=current_vibe, 
                    validation_map=memory.validations  
                )
                
                extracted_short_ids = []
                for b in step1_out.belief_state:
                    if b.source_id:
                        found_ids = re.findall(r'S\d+-A\d+', str(b.source_id).upper())
                        extracted_short_ids.extend(found_ids)
                
                if extracted_short_ids:
                    raw_bonus_ids = memory.resolve_short_ids(list(set(extracted_short_ids)))
                    if raw_bonus_ids: memory.boost_assets(raw_bonus_ids)
                
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
                            past_queries_report="\n".join(past_queries_tracker)
                        )
                    except Exception as e:
                        print(f"⚠️ Planner error: {e}")

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

            if cognitive_output.decision in ['REFERENCE_SUFFICIENT', 'NO_SEARCH_NEEDED'] or step >= self.max_steps:
                break

            next_actions = getattr(cognitive_output, 'next_search_actions', [])
            if not next_actions:
                break
            
        # ==============================================================================
        # 4. (Final Response Formulation)
        # ==============================================================================
        final_draft = memory.get_best_final_draft()
        final_evidence, _ = memory.elect_best_evidence_set()
        
        needs_fallback = not final_evidence or (final_draft and final_draft.status == 'INSUFFICIENT')
            
        if needs_fallback:
            print("⚠️ Switching to Fallback...")
            final_short_id_map = {}
            try:
                final_answer = self.roles.generate_fallback_response(initial_question)
            except AttributeError:
                final_answer = "Information insufficient."
            final_think = "Fallback protocol."
        else:
            print("✍️ Drafting final response...")
            
            final_short_id_map = {}
            sf_counter = 1
            
            for asset in final_evidence:
                if asset.asset_id in historical_short_ids:
                    assigned_short_id = historical_short_ids[asset.asset_id]
                else:
                    assigned_short_id = f"SF-A{sf_counter}"
                    sf_counter += 1
                final_short_id_map[assigned_short_id] = asset

            final_answer, final_think = self.roles.generate_final_response(
                question=initial_question, 
                draft=final_draft, 
                final_assets=final_evidence,
                short_id_map=final_short_id_map, 
                prev_beliefs=prev_beliefs 
            )
        
        memory.set_final_answer(text=final_answer, think_trace=final_think, final_short_id_map=final_short_id_map) 
        print(f"Done. Time: {(time.time() - start_time):.2f}s")
        
        return final_answer, memory.export_full_state()