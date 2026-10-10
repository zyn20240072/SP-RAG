# main.py
import json
import os
import time
from tqdm import tqdm
from typing import Dict, Any, Set, Optional
import multiprocessing as mp
from functools import partial
import traceback

from pydantic import BaseModel

from config import (
    ALL_DATASET_CONFIGS,
    ROLE_MODEL_CONFIGS,
    RETRIEVAL_STRATEGY,
    NUM_PROCESSES,
)
from agent import IterativeAgent
from llm_roles import LLMRoles


class PydanticJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode='json')
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def log_error(question_id: str, error_msg: str, dataset_name: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    error_line = f"[{timestamp}] [{dataset_name}] ID: {question_id} | Error: {error_msg}\n"
    try:
        with open("processing_errors.log", "a", encoding="utf-8") as f:
            f.write(error_line)
    except Exception:
        print(f"!!! CRITICAL: Failed to write error log for ID {question_id}")

def process_single_item(
    item: Dict[str, Any],
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Worker entry point with per-sample state isolation and error handling."""
    question_id = str(item.get("question_id", "unknown"))
    initial_question = item.get("question")
    dataset_name = dataset_config["name"]
    image_root_path = dataset_config.get("prefix_image", "")

    if not initial_question:
        log_error(question_id, "Missing question text", dataset_name)
        return None

    try:
        llm_roles = LLMRoles(
            model_config=model_config,
            image_root_path=image_root_path,
        )
        agent = IterativeAgent(
            llm_roles=llm_roles,
            dataset=dataset_name,
            retrieval_strategy=RETRIEVAL_STRATEGY,
        )

        final_answer_str, final_memory = agent.run(
            initial_question=initial_question
        )
        if final_answer_str is None:
            log_error(question_id, "Agent returned None", dataset_name)
            return None

        final_output_item = item.copy()
        final_output_item.update(
            {
                "final_answer": final_answer_str,
                "agent_trace": final_memory,
            }
        )
        return final_output_item
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"❌ [{dataset_name}] {question_id}: {exc}")
        log_error(question_id, error_msg, dataset_name)
        return None


def read_processed_ids(file_path: str) -> Set[str]:
    processed_ids = set()
    if not os.path.exists(file_path):
        return processed_ids
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and 'question_id' in data:
                    processed_ids.add(str(data['question_id']))
            except json.JSONDecodeError:
                continue
    return processed_ids

def process_dataset(dataset_config: Dict[str, Any], model_config: Dict[str, Any], num_processes: int = 4):
    dataset_name = dataset_config['name']
    input_path = dataset_config["input_jsonl"]
    output_path = dataset_config["output_jsonl"]
    
    print(f"\n===== 🚀 Dataset: {dataset_name} =====")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    processed_ids = read_processed_ids(output_path)
    print(f"Found {len(processed_ids)} already processed items.")

    all_data = []
    input_ids: Set[str] = set()
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                log_error(
                    f"line-{line_number}",
                    f"Invalid JSONL record: {exc}",
                    dataset_name,
                )
                continue
            if not isinstance(item, dict):
                log_error(
                    f"line-{line_number}",
                    "JSONL record is not an object",
                    dataset_name,
                )
                continue

            raw_question_id = item.get("question_id")
            question_id = str(raw_question_id).strip() if raw_question_id is not None else ""
            if not question_id:
                question_id = f"line-{line_number}"
                item["question_id"] = question_id

            if question_id in input_ids:
                log_error(
                    question_id,
                    "Duplicate question_id in input; later record skipped",
                    dataset_name,
                )
                continue
            input_ids.add(question_id)
            all_data.append(item)

    data_to_process = [
        item
        for item in all_data
        if str(item.get('question_id', 'unknown')) not in processed_ids
    ]
    print(f"Total: {len(all_data)} | Remaining: {len(data_to_process)}")

    if not data_to_process:
        print(f"✅ Dataset {dataset_name} is already finished.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)
    num_processes = max(1, int(num_processes))

    print(f"⚙️  Parallelism: {num_processes} processes (Spawn mode)")

    process_func = partial(
        process_single_item,
        model_config=model_config,
        dataset_config=dataset_config
    )

    with open(output_path, "a", encoding='utf-8') as f_out:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap_unordered(process_func, data_to_process, chunksize=1), 
                             total=len(data_to_process), 
                             desc=f"Running {dataset_name}"):
                if result is not None:
                    try:
                        line = json.dumps(result, ensure_ascii=False, cls=PydanticJSONEncoder)
                        f_out.write(line + '\n')
                        f_out.flush() 
                    except TypeError as e:
                        err_msg = f"JSON Serialization Error: {e}"
                        print(f"❌ {err_msg}")
                        log_error(result.get('question_id'), err_msg, dataset_name)

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("--- 🛠️  Multi-Agent Visual RAG Pipeline (Sub-Question Architecture) ---")
    # print(f"Selected Model: {SELECTED_MODEL_CONFIG.get('model_name', 'Unknown')}")

    for dataset_config in ALL_DATASET_CONFIGS:
        process_dataset(dataset_config, ROLE_MODEL_CONFIGS, num_processes=NUM_PROCESSES)

    print("\n--- ✨ All tasks completed. ---")

if __name__ == '__main__':
    main()