#!/usr/bin/env python3
"""
vllm_chat_server.py
Serveur vLLM OpenAI-compatible pour les Chat Completions avec gestion via lifespan.
"""

import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- 1. Lecture de l'argument modèle ---
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Nom du modèle à charger")
args = parser.parse_args()

# --- 2. Chargement des hyperparamètres depuis JSON ---

project_dir = Path(__file__).resolve().parents[2]
params_path = project_dir / "src" / "vllm_server" / f"{args.model_name}.json"

if not params_path.is_file():
    raise SystemExit(f"[ERROR] Fichier de config introuvable : {params_path}")

raw = params_path.read_text(encoding="utf-8")
filtered = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("//"))
cfg = json.loads(filtered)

# --- 3. Configuration modèle et chemins ---

model_name = cfg["model_name"]
abs_path = cfg.get("absolute_path")
rel_path = cfg.get("relative_path")

if abs_path:
    model_dir = Path(abs_path)
elif rel_path:
    model_dir = (project_dir / rel_path).resolve()
else:
    raise SystemExit("[ERROR] Aucun chemin 'absolute_path' ou 'relative_path' défini dans le JSON.")

if not model_dir.is_dir():
    raise SystemExit(f"[ERROR] Modèle introuvable : {model_dir}")

model_path = str(model_dir)



# --- 4. Extraction des autres paramètres ---

dtype = cfg.get("dtype")
quantization = cfg.get("quantization")
kv_cache_dtype = cfg.get("kv_cache_dtype")
gpu_memory_utilization = cfg.get("gpu_memory_utilization")
max_model_len = cfg.get("max_model_len")
tensor_parallel_size = cfg.get("tensor_parallel_size", 1)
pipeline_parallel_size = cfg.get("pipeline_parallel_size", 1)
max_num_seqs = cfg.get("max_num_seqs", 1)
max_num_batched_tokens = cfg.get("max_num_batched_tokens", 0)
swap_space_gb = cfg.get("swap_space_gb", 0)
cpu_offload_gb = cfg.get("cpu_offload_gb", 0)

# --- 5. Déclarations globales ---

llm: Optional[LLM] = None
tokenizer: Optional[AutoTokenizer] = None
model_ready = False

# --- 6. Gestion via lifespan FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, tokenizer, model_ready
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        dtype=dtype,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=True,
        swap_space=swap_space_gb,
        cpu_offload_gb=cpu_offload_gb,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_ready = True
    print("[INFO] Model and tokenizer loaded.")
    yield

app = FastAPI(lifespan=lifespan, openapi_url="/v1/openapi.json", docs_url="/v1/docs")

# --- 7. Endpoint statut modèle ---
@app.get("/v1/status")
async def status():
    return {"ready": model_ready}

# --- 8. Endpoint Chat Completions ---
@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_completion_tokens", 1024)

    if not messages:
        return {"error": "No messages provided."}

    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    out = llm.generate([chat_prompt], sampling_params)[0]

    return {
        "id": "chatcmpl-generated",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": choice.text
                }
            }
            for choice in out.outputs
        ],
    }

# --- 9. Lancement du serveur ---
if __name__ == "__main__":
    uvicorn.run(
        "vllm_chat_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
        access_log=False,
    )
