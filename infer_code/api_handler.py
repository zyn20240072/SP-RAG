# api_handler.py
import os
import io
import base64
from openai import OpenAI
from typing import List, Dict, Any, Optional
from openai import APITimeoutError, APIConnectionError
import time
from PIL import Image
import httpx

def get_system_prompt():
    prompt_path = "prompt.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a helpful assistant." 

def process_image_for_api(image_path: str, image_root_path: str) -> Optional[Dict[str, Any]]:
    full_path = os.path.join(image_root_path, image_path) if image_root_path else image_path
    
    if not os.path.exists(full_path):
        print(f"  ⚠️ Warning: Image file not found at {full_path}")
        return None

    try:
        with Image.open(full_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            MAX_SIDE = 1024

            if original_width > MAX_SIDE or original_height > MAX_SIDE:
                scale = min(MAX_SIDE / original_width, MAX_SIDE / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95, subsampling=0)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
    except Exception as e:
        print(f"  ⚠️ Error processing image {full_path}: {e}")
        return None

def call_qwen_api(model_config: dict, 
                  prompt: str = None, 
                  image_paths: List[str] = None, 
                  image_root_path: str = None, 
                  
                  interleaved_content: List[Dict] = None, 
                  temperature: float = 0.0) -> str:
    
    if "gpt" in model_config['model_name'] or "gemini" in model_config['model_name']:
        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['api_base'],
            http_client=httpx.Client(
                proxy="http://172.20.137.65:40171",
                transport=httpx.HTTPTransport(local_address="0.0.0.0")
            )
        )
    else:
        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['api_base']
        )

    system_content = get_system_prompt()
    messages = [{"role": "system", "content": system_content}]

    if interleaved_content is not None:
        messages.append({"role": "user", "content": interleaved_content})
        
    else:
        content = []
        if image_paths:
            for img_path in image_paths:
                img_obj = process_image_for_api(img_path, image_root_path)
                if img_obj:
                    content.append(img_obj)
        
        if prompt:
            content.append({"type": "text", "text": prompt})
            
        messages.append({"role": "user", "content": content})
        

    max_retries = 1
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                max_tokens=model_config["max_new_tokens"],
                temperature=temperature,
                timeout=300.0
            )
            # print("******************************************************")
            # print(messages)
            # print(completion.choices[0].message.content.strip())
            # print("******************************************************")
            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"  API Error (Attempt {attempt+1}/{max_retries}): {e}")
            print(model_config)
            if attempt + 1 == max_retries: raise e
            time.sleep(retry_delay)

    raise Exception("API call failed after retries.")