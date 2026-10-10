import os
import json
import re
import math
from typing import List, Dict, Any, Union, Tuple
import functools

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from swift.utils import get_logger

try:
    from tools import Toolbox
except ImportError:
    raise ImportError("Could not import 'Toolbox' from tools.py. Please ensure the file exists.")

try:
    import json_repair
except ImportError:
    raise ImportError("Please install json_repair: pip install json_repair")

logger = get_logger()

_TOOLBOX_CACHE = {}

try:
    from rouge import Rouge
    rouge_scorer = Rouge()
except ImportError:
    rouge_scorer = None


def _ensure_float(score: Any) -> float:
    try:
        if hasattr(score, 'item'):
            return float(score.item())
        elif isinstance(score, (list, tuple, set)):
            score_list = list(score)
            while isinstance(score_list, (list, tuple)):
                if len(score_list) == 0:
                    return 0.0
                score_list = score_list[0]
            return float(score_list)
        return float(score)
    except Exception as e:
        logger.error(f"Failed to convert score to float: {score}, error: {e}")
        return 0.0


def get_toolbox(dataset_name: str, top_n: int = 3) -> Toolbox:
    if dataset_name not in _TOOLBOX_CACHE:
        logger.info(f"🚀 Initializing Toolbox for dataset: {dataset_name}")
        _TOOLBOX_CACHE[dataset_name] = Toolbox(dataset=dataset_name, top_n=top_n)
    return _TOOLBOX_CACHE[dataset_name]


def _get_active_toolbox():
    if _TOOLBOX_CACHE:
        return next(iter(_TOOLBOX_CACHE.values()))
    return None


def extract_think_and_json(text: str) -> Tuple[str, str, bool]:
    json_block_count = text.count("```json")
    think_block_count = text.count("<think>")
    
    is_format_violation = (think_block_count > 1) 
    
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""
    
    remaining = text.replace(think_match.group(0), '') if think_match else text
    json_match = re.search(r'```json\n?(.*?)\n?```', remaining, re.DOTALL | re.IGNORECASE)
    json_str = json_match.group(1).strip() if json_match else remaining.strip()
    
    return think, json_str, is_format_violation


def parse_json(json_str: str) -> Union[Dict, List, None]:
    try:
        parsed = json_repair.loads(json_str)
        if isinstance(parsed, list):
            parsed = next((item for item in parsed if isinstance(item, dict)), parsed[0] if parsed else {})
        return parsed
    except Exception: 
        return None


def normalize_and_extract_numbers(text: str) -> set:
    text_lower = text.lower()
    nums = set(re.findall(r'\b\d+(?:[,.]\d+)*\b', text_lower))
    
    eng_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    for word, digit in eng_map.items():
        if re.search(rf'\b{word}\b', text_lower): nums.add(digit)
            
    chn_map = {
        '零': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4', 
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
    }
    for char, digit in chn_map.items():
        if char in text_lower: nums.add(digit)
            
    return nums


def is_query_semantically_similar(q1: str, q2: str, threshold: float = 0.88) -> bool:
    if not q1 or not q2: return False
    
    tb = _get_active_toolbox()
    if tb and hasattr(tb, 'calculate_text_sim_pair'):
        try:
            cosine_score = tb.calculate_text_sim_pair(q1, q2)
            cosine_score = _ensure_float(cosine_score)
            return cosine_score > threshold
        except Exception as e:
            logger.error(f"\n[BGE-M3 Error] Semantic model failed during similarity check: {e}")
            pass
    
    q1_clean = re.sub(r'[^\w\s]', '', q1.lower()).strip()
    q2_clean = re.sub(r'[^\w\s]', '', q2.lower()).strip()
    if q1_clean == q2_clean: return True
    w1, w2 = set(q1_clean.split()), set(q2_clean.split())
    if not w1 or not w2: return False
    return len(w1 & w2) / len(w1 | w2) >= 0.75


def calculate_text_similarity(pred: str, ref: str) -> float:
    if not pred or not ref: return 0.0
        
    pred_lower = pred.lower().strip()
    ref_lower = ref.lower().strip()
    if pred_lower == ref_lower: return 1.0
        
    pred_words = pred_lower.split()
    ref_words = ref_lower.split()

    is_concise = len(pred_words) <= len(ref_words) + 1

    if len(ref_words) <= 5 and ref_lower in pred_lower: 
        if is_concise:
            return 0.9
        else:
            logger.info(f"⚠️ [Reward Hack Guard] Model included GT '{ref_lower}' but rambled ({len(pred_words)} words). Bypassing shortcut.")

    if len(ref_words) <= 2:
        ref_nums = normalize_and_extract_numbers(ref_lower) 
        if ref_nums:
            for n in ref_nums:
                if re.search(rf'(?<!\d){n}(?!\d)', pred_lower): 
                    if is_concise:
                        return 0.9
                    else:
                        logger.info(f"⚠️ [Reward Hack Guard] Model included Num '{n}' but rambled. Bypassing shortcut.")

    tb = _get_active_toolbox()
    if tb and hasattr(tb, 'calculate_text_sim_pair'):
        try:
            cosine_score = tb.calculate_text_sim_pair(pred_lower, ref_lower)
            cosine_score = _ensure_float(cosine_score)
            logger.info(f"Final Answer similarity score: {cosine_score:.3f}")
            return max(0.0, cosine_score)
        except Exception as e:
            logger.error(f"\n[BGE-M3 Error] failed: {e}. Falling back to lexical match.")
            pass
            
    if 'rouge_scorer' not in globals() or not rouge_scorer:
        s1, s2 = set(pred_words), set(ref_words)
        return len(s1 & s2) / len(s1 | s2) if s1 or s2 else 0.0
    try: 
        return rouge_scorer.get_scores(pred_lower, ref_lower)[0]['rouge-l']['f']
    except Exception as e:
        logger.error(f"!!!{e}")
        return 0.0


def evaluate_answer_factuality(pred_text: str, gt_text: str) -> float:
    score_mod = 0.0
    gt_numbers = normalize_and_extract_numbers(gt_text)
    pred_numbers = normalize_and_extract_numbers(pred_text)
    
    if not gt_numbers: return 0.0
        
    hit_nums = gt_numbers & pred_numbers
    num_recall = len(hit_nums) / len(gt_numbers)
    
    if num_recall == 1.0: score_mod += 0.8  
    elif num_recall > 0.0: score_mod += 0.2  
    else: score_mod -= 1.5  
        
    hallucinated_nums = pred_numbers - gt_numbers
    if hallucinated_nums and num_recall < 1.0:
        score_mod -= 0.2 * len(hallucinated_nums)
    elif hallucinated_nums and num_recall == 1.0:
        score_mod -= 0.05 * min(len(hallucinated_nums), 3) 
        
    return score_mod


def score_reasoning_quality(reasoning: str, belief_state: List[Dict], boilerplate_phrases: List[str], forbidden_penalty: float, source_tracing_bonus: float) -> float:
    if not reasoning or len(reasoning) < 10: return -1.0
    reason_lower = reasoning.lower()
    
    if any(phrase in reason_lower for phrase in boilerplate_phrases):
        return forbidden_penalty
        
    score = 0.2 
    if belief_state:
        hit_count = sum(1 for b in belief_state if isinstance(b, dict) and b.get("source_id", "unknown") != "unknown")
        score += hit_count * source_tracing_bonus
    return score


def normalize_score(think_score: float, task_score: float) -> float:
    total = think_score + task_score
    return 3.0 * math.tanh(total / 3.0)


def calculate_thinking_tax(think_content: str, base_score: float, role: str) -> float:
    if not think_content: return -5.0
    if len(think_content)<150: return (-150.0 / len(think_content))
    if role == "Validator": threshold = 300
    elif role in ["Planner", "Cognition"]: threshold = 600
    else: threshold = 800
        
    word_count = len(think_content.split())
    if word_count <= threshold: return base_score
        
    excess_words = word_count - threshold
    penalty = (excess_words / threshold) * 0.2  
    return max(0.0, base_score - penalty)


def calculate_linguistic_penalty(query: str) -> float:
    if not query: return -3.0
        
    penalty = 0.0
    words = query.lower().split()
    num_words = len(words)
    
    if num_words == 0: return -3.0
    
    if num_words > 20:
        penalty -= ((num_words - 20) / 5.0) * 0.2
        
    max_word_len = max(len(w) for w in words)
    if max_word_len > 25:
        penalty -= (max_word_len - 25) * 0.1
        
    if num_words >= 5:

        ttr = len(set(words)) / num_words
        if ttr < 0.7:
            penalty -= (0.7 - ttr) * 5.0
            
        consecutive_dupes = sum(1 for i in range(num_words - 1) if words[i] == words[i+1])
        if consecutive_dupes > 0:
            penalty -= (consecutive_dupes ** 1.5) * 0.3
            
    if num_words >= 4:
        bigrams = [tuple(words[i:i+2]) for i in range(num_words - 1)]
        bigram_ratio = len(set(bigrams)) / len(bigrams)
        if bigram_ratio < 0.85:
            penalty -= (0.85 - bigram_ratio) * 6.0

    if penalty < 0:
        return 3.0 * math.tanh(penalty / 3.0)
        
    return 0.0


def verify_numeric_consistency(gt_str: str, fact_str: str) -> bool:
    if not gt_str or not fact_str: return True
    gt_clean = gt_str.lower().replace(',', '')
    fact_clean = fact_str.lower().replace(',', '')
    
    gt_nums_str = re.findall(r'\b\d+(?:\.\d+)?\b', gt_clean)
    if not gt_nums_str: return True 
    
    fact_nums_str = re.findall(r'\b\d+(?:\.\d+)?\b', fact_clean)
    try: fact_vals = [float(f) for f in fact_nums_str]
    except ValueError: fact_vals = []
    
    for g_s in gt_nums_str:
        g_val = float(g_s)
        if g_val in fact_vals: continue
        if 1900 <= g_val <= 2099 and (g_val % 100) in fact_vals: continue
        if 0 <= g_val <= 99 and ((2000 + g_val) in fact_vals or (1900 + g_val) in fact_vals): continue
        if g_s in fact_clean: continue
        
        return False
    return True