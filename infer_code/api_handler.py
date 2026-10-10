import base64
import io
import os
import threading
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from PIL import Image

from config import (
    API_MAX_RETRIES,
    API_PROXY,
    API_RETRY_DELAY,
    API_TIMEOUT,
    IMAGE_MAX_SIDE,
)


# ============================================================
# Global client cache
# ============================================================

_CLIENT_CACHE: Dict[Tuple[str, str, str], OpenAI] = {}
_CLIENT_LOCK = threading.Lock()


# ============================================================
# Qwen3.5 generation defaults
# ============================================================

# Final answer: native thinking ON
QWEN35_THINKING_TEMPERATURE = 0.0
QWEN35_THINKING_TOP_P = 0.95
QWEN35_THINKING_TOP_K = 20
QWEN35_THINKING_MIN_P = 0.0
QWEN35_THINKING_PRESENCE_PENALTY = 1.5
QWEN35_THINKING_REPETITION_PENALTY = 1.0

# Rewrite / planner / control generation: native thinking OFF
QWEN35_NON_THINKING_TEMPERATURE = 0.0


# ============================================================
# System prompt
# ============================================================

@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    """
    Load the optional system prompt without depending on the launch directory.

    Priority:
    1. RAG_SYSTEM_PROMPT environment variable
    2. ./prompt.txt
    3. prompt.txt next to this Python file
    """
    candidates: List[str] = []

    env_path = os.getenv("RAG_SYSTEM_PROMPT")
    if env_path:
        candidates.append(env_path)

    candidates.extend(
        [
            os.path.join(os.getcwd(), "prompt.txt"),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "prompt.txt",
            ),
        ]
    )

    for prompt_path in candidates:
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as file:
                return file.read()

    return "You are a helpful assistant."


# ============================================================
# Image processing
# ============================================================

def process_image_for_api(
    image_path: str,
    image_root_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Read, resize, and encode an image as an OpenAI-compatible data URL.

    The image aspect ratio is preserved. The longest side is capped by
    IMAGE_MAX_SIDE from config.py.
    """
    full_path = (
        os.path.join(image_root_path, image_path)
        if image_root_path and not os.path.isabs(image_path)
        else image_path
    )

    if not full_path or not os.path.exists(full_path):
        print(f"  [Warning] Image file not found at {full_path}")
        return None

    try:
        with Image.open(full_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid image dimensions: {image.size}"
                )

            if width > IMAGE_MAX_SIDE or height > IMAGE_MAX_SIDE:
                scale = min(
                    IMAGE_MAX_SIDE / width,
                    IMAGE_MAX_SIDE / height,
                )

                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )

                resampling = getattr(
                    Image,
                    "Resampling",
                    Image,
                )

                image = image.resize(
                    new_size,
                    resampling.LANCZOS,
                )

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=95,
                subsampling=0,
            )

            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            return {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + encoded
                    )
                },
            }

    except Exception as exc:
        print(
            f"  [Warning] Error processing image "
            f"{full_path}: {exc}"
        )
        return None


# ============================================================
# OpenAI-compatible client
# ============================================================

def _client_cache_key(
    model_config: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    Build a cache key for one endpoint.

    Proxy behavior is preserved from the original code:
    only GPT/Gemini model names use API_PROXY.
    """
    base_url = str(model_config["api_base"])
    api_key = str(model_config.get("api_key", ""))
    model_name = str(
        model_config.get("model_name", "")
    ).lower()

    proxy = (
        API_PROXY
        if ("gpt" in model_name or "gemini" in model_name)
        else ""
    )

    return base_url, api_key, proxy


def _get_client(
    model_config: Dict[str, Any],
) -> OpenAI:
    """
    Reuse one OpenAI/httpx client per process and endpoint.
    """
    key = _client_cache_key(model_config)

    with _CLIENT_LOCK:
        cached = _CLIENT_CACHE.get(key)

        if cached is not None:
            return cached

        base_url, api_key, proxy = key

        http_client = None

        if proxy:
            client_kwargs = {
                "transport": httpx.HTTPTransport(
                    local_address="0.0.0.0"
                ),
                "timeout": API_TIMEOUT,
            }

            try:
                http_client = httpx.Client(
                    proxy=proxy,
                    **client_kwargs,
                )
            except TypeError:
                # Compatibility with older httpx versions.
                http_client = httpx.Client(
                    proxies=proxy,
                    **client_kwargs,
                )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=API_TIMEOUT,
            max_retries=0,
        )

        _CLIENT_CACHE[key] = client

        return client


# ============================================================
# Response helpers
# ============================================================

def _safe_usage_dict(
    completion: Any,
) -> Optional[Dict[str, Any]]:
    """
    Convert OpenAI-compatible usage metadata into a JSON-serializable dict.
    """
    usage = getattr(
        completion,
        "usage",
        None,
    )

    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass

    try:
        return dict(usage)
    except Exception:
        return None


def _extract_reasoning(
    message: Any,
) -> str:
    """
    Extract Qwen3.5 native reasoning returned by vLLM.

    Preferred field:
        message.reasoning

    Compatibility fallback:
        message.reasoning_content
    """
    if message is None:
        return ""

    # First inspect model_dump because some OpenAI SDK versions keep
    # server-specific extra fields there.
    if hasattr(message, "model_dump"):
        try:
            data = message.model_dump()

            reasoning = (
                data.get("reasoning")
                or data.get("reasoning_content")
            )

            if reasoning:
                return str(reasoning).strip()

        except Exception:
            pass

    reasoning = getattr(
        message,
        "reasoning",
        None,
    )

    if reasoning:
        return str(reasoning).strip()

    reasoning = getattr(
        message,
        "reasoning_content",
        None,
    )

    if reasoning:
        return str(reasoning).strip()

    return ""


# ============================================================
# Request content construction
# ============================================================

def _build_user_content(
    prompt: Optional[str],
    image_paths: Optional[List[str]],
    image_root_path: Optional[str],
    interleaved_content: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible multimodal user content.

    If interleaved_content is supplied, it is used directly so the caller
    can explicitly control text/image ordering.
    """
    if interleaved_content is not None:
        return interleaved_content

    user_content: List[Dict[str, Any]] = []

    for image_path in image_paths or []:
        image_obj = process_image_for_api(
            image_path,
            image_root_path or "",
        )

        if image_obj:
            user_content.append(image_obj)

    if prompt:
        user_content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

    return user_content


# ============================================================
# Full Qwen3.5 API
# ============================================================

def call_qwen_api_full(
    model_config: Dict[str, Any],
    prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_root_path: Optional[str] = None,
    interleaved_content: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    enable_thinking: bool = False,
    stop: Optional[List[str]] = None,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call Qwen3.5 through an OpenAI-compatible endpoint.

    This utility assumes the endpoint is Qwen3.5.
    No model-family detection is performed.

    Recommended usage
    -----------------
    Query rewrite / planner / control:
        enable_thinking=False

    Final answer:
        enable_thinking=True

    For native reasoning extraction, start vLLM with:
        --reasoning-parser qwen3

    Returns
    -------
    {
        "content": str,
        "reasoning": str,
        "finish_reason": Optional[str],
        "usage": Optional[dict],
        "truncated": bool,
    }
    """
    required = {
        "model_name",
        "api_base",
        "max_new_tokens",
    }

    missing = required - set(model_config)

    if missing:
        raise KeyError(
            "Model configuration is missing fields: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": get_system_prompt(),
        }
    ]

    user_content = _build_user_content(
        prompt=prompt,
        image_paths=image_paths,
        image_root_path=image_root_path,
        interleaved_content=interleaved_content,
    )

    if not user_content:
        raise ValueError(
            "The API request contains no user content."
        )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    # --------------------------------------------------------
    # Generation configuration
    # --------------------------------------------------------

    client = _get_client(model_config)

    tokens = int(
        max_new_tokens
        if max_new_tokens is not None
        else model_config["max_new_tokens"]
    )

    if temperature is None:
        if enable_thinking:
            resolved_temperature = (
                QWEN35_THINKING_TEMPERATURE
            )
        else:
            resolved_temperature = (
                QWEN35_NON_THINKING_TEMPERATURE
            )
    else:
        resolved_temperature = float(
            temperature
        )

    last_error: Optional[Exception] = None

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    retry_count = max(
        1,
        int(API_MAX_RETRIES),
    )

    for attempt in range(
        1,
        retry_count + 1,
    ):
        try:
            # ------------------------------------------------
            # Base request
            # ------------------------------------------------

            request_kwargs: Dict[str, Any] = {
                "model": model_config["model_name"],
                "messages": messages,
                "max_tokens": tokens,
                "temperature": resolved_temperature,
                "timeout": API_TIMEOUT,
            }

            if stop is not None:
                request_kwargs["stop"] = stop

            # ------------------------------------------------
            # Qwen3.5 chat-template control
            # ------------------------------------------------

            extra_body: Dict[str, Any] = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(
                        enable_thinking
                    ),
                }
            }

            # ------------------------------------------------
            # Qwen3.5 thinking-mode sampling
            # ------------------------------------------------

            if enable_thinking:
                request_kwargs[
                    "top_p"
                ] = QWEN35_THINKING_TOP_P

                request_kwargs[
                    "presence_penalty"
                ] = (
                    QWEN35_THINKING_PRESENCE_PENALTY
                )

                extra_body.update(
                    {
                        "top_k": (
                            QWEN35_THINKING_TOP_K
                        ),
                        "min_p": (
                            QWEN35_THINKING_MIN_P
                        ),
                        "repetition_penalty": (
                            QWEN35_THINKING_REPETITION_PENALTY
                        ),
                    }
                )

            request_kwargs[
                "extra_body"
            ] = extra_body

            # ------------------------------------------------
            # API call
            # ------------------------------------------------

            completion = (
                client.chat.completions.create(
                    **request_kwargs
                )
            )

            if not completion.choices:
                raise ValueError(
                    "The API returned no choices."
                )

            choice = completion.choices[0]
            message = choice.message

            # ------------------------------------------------
            # Parse visible answer + native reasoning
            # ------------------------------------------------

            content = (
                message.content or ""
            ).strip()

            reasoning = (
                _extract_reasoning(message)
                if enable_thinking
                else ""
            )

            finish_reason = getattr(
                choice,
                "finish_reason",
                None,
            )

            truncated = (
                finish_reason == "length"
            )

            # If native reasoning consumes the whole token budget,
            # content may legitimately be empty while reasoning is not.
            if not content and not reasoning:
                raise ValueError(
                    "The API returned neither visible "
                    "content nor reasoning."
                )

            return {
                "content": content,
                "reasoning": reasoning,
                "finish_reason": finish_reason,
                "usage": _safe_usage_dict(
                    completion
                ),
                "truncated": truncated,
            }

        except Exception as exc:
            last_error = exc

            print(
                f"  API Error "
                f"(Attempt {attempt}/{retry_count}) "
                f"[{model_config['model_name']}]: "
                f"{exc}"
            )

            if attempt < retry_count:
                time.sleep(
                    API_RETRY_DELAY * attempt
                )

    raise RuntimeError(
        "API call failed after "
        f"{retry_count} attempts: "
        f"{last_error}"
    ) from last_error


# ============================================================
# Backward-compatible string API
# ============================================================

def call_qwen_api(
    model_config: Dict[str, Any],
    prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_root_path: Optional[str] = None,
    interleaved_content: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    enable_thinking: bool = False,
    stop: Optional[List[str]] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """
    Backward-compatible wrapper that returns visible content only.

    Existing code can continue to use:

        output = call_qwen_api(...)

    For query rewrite:
        output = call_qwen_api(
            ...,
            enable_thinking=False,
            max_new_tokens=128,
        )

    For final answers where native reasoning must be saved, use
    call_qwen_api_full(...) instead.
    """
    result = call_qwen_api_full(
        model_config=model_config,
        prompt=prompt,
        image_paths=image_paths,
        image_root_path=image_root_path,
        interleaved_content=interleaved_content,
        temperature=temperature,
        enable_thinking=enable_thinking,
        stop=stop,
        max_new_tokens=max_new_tokens,
    )

    content = result["content"]

    if content:
        return content

    # Native thinking may consume all output tokens before the final
    # visible answer is generated.
    if result["truncated"]:
        raise RuntimeError(
            "Generation reached max_new_tokens before "
            "a visible final answer was produced. "
            "Increase max_new_tokens."
        )

    raise ValueError(
        "The API returned empty visible content."
    )
