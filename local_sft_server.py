#!/usr/bin/env python3
"""Serve the fine-tuned Qwen adapter with a small OpenAI-compatible API."""

from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled during startup
    PeftModel = None


BASE_MODEL = os.getenv("LOCAL_SFT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_PATH = os.getenv(
    "LOCAL_SFT_ADAPTER_PATH",
    "training/outputs/qwen25-3b-edu-qlora-v21",
)
MODEL_ID = os.getenv("LOCAL_SFT_MODEL_ID", "qwen25-3b-edu-qlora-v21")
DEVICE_MAP = os.getenv("LOCAL_SFT_DEVICE_MAP", "auto")

app = FastAPI(title="Local SFT OpenAI-Compatible Server")
generation_lock = Lock()
tokenizer = None
model = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 512
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False


def _torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16 if torch.cuda.is_available() else torch.float32


@app.on_event("startup")
def load_model() -> None:
    global tokenizer, model

    if PeftModel is None:
        raise RuntimeError("peft is required to load the LoRA adapter.")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_torch_dtype(),
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
    if request.stream:
        raise ValueError("Streaming responses are not implemented.")

    messages = [message.model_dump() for message in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]
    temperature = max(float(request.temperature or 0.0), 0.0)
    max_new_tokens = int(request.max_tokens or 512)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": float(request.top_p or 1.0),
        "temperature": temperature if temperature > 0 else None,
        "repetition_penalty": float(request.repetition_penalty or 1.0),
        "pad_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {
        key: value
        for key, value in generation_kwargs.items()
        if value is not None
    }

    with generation_lock:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

    completion_ids = output_ids[0, input_length:]
    content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    prompt_tokens = int(inputs.input_ids.numel())
    completion_tokens = int(completion_ids.numel())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "loading",
        "model": MODEL_ID,
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
        "cuda": torch.cuda.is_available(),
    }
