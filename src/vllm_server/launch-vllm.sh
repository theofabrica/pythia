#!/bin/bash
set -e

# Détermine le chemin du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/vllm-server"
REQ_FILE="$SCRIPT_DIR/../../docker/requirements.txt"

# 0) Vérifie qu’un argument est fourni
if [ -z "$1" ]; then
    echo "[ERROR] Aucun model_name fourni."
    echo "Usage: $0 <model_name>"
    exit 1
fi
MODEL_NAME="$1"

# 1) Création du venv si absent
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Création du venv vllm-server..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
else
    echo "[INFO] venv déjà présent."
fi

# 2) Activation et lancement du serveur vLLM
echo "[INFO] Lancement de vllm_chat_server.py avec --model_name $MODEL_NAME ..."
source "$VENV_DIR/bin/activate"
exec python "$SCRIPT_DIR/vllm_chat_server.py" --model_name "$MODEL_NAME"
