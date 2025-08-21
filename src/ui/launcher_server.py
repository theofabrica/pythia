# src/ui/launcher_server.py
import sys
import subprocess
import threading
from flask import Flask, request

app = Flask(__name__)

# ---------------- Milvus ----------------
@app.route("/milvus/up", methods=["POST"])
def start_milvus():
    subprocess.run(
        ["docker", "compose", "--env-file", ".env", "-f", "docker-compose.yml", "up", "-d"],
        cwd="/home/theoub02/ai_gen/Projets/pythia/src/tools/RAG/VectorStore/milvus_local"
    )
    return "Milvus started\n", 200


@app.route("/milvus/down", methods=["POST"])
def stop_milvus():
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "down"],
        cwd="/home/theoub02/ai_gen/Projets/pythia/src/tools/RAG/VectorStore/milvus_local"
    )
    return "Milvus stopped\n", 200


# ---------------- vLLM ----------------
VLLM_PROCESS = None

def stream_logs(process):
    """Relaye les logs de vLLM vers la console Flask en direct."""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[vLLM] {line}", end='')
    process.stdout.close()

@app.route("/vllm/up", methods=["POST"])
def start_vllm():
    """Lance vLLM via launch-vllm.sh avec le modèle choisi (si pas déjà en cours)."""
    global VLLM_PROCESS
    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is None:
        return "vLLM already running\n", 200

    # Récupérer model_name depuis le JSON envoyé par l'UI
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    if not model_name:
        return "Missing model_name in request\n", 400

    script_path = "/home/theoub02/ai_gen/Projets/pythia/src/vllm_server/launch-vllm.sh"
    VLLM_PROCESS = subprocess.Popen(
        [script_path, model_name],   # <-- passe model_name en argument
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Thread qui lit les logs et les affiche
    threading.Thread(target=stream_logs, args=(VLLM_PROCESS,), daemon=True).start()

    return f"vLLM starting with model: {model_name}\n", 200



@app.route("/vllm/down", methods=["POST"])
def stop_vllm():
    """Arrête vLLM et décharge le modèle (kill du process)."""
    global VLLM_PROCESS
    if VLLM_PROCESS and VLLM_PROCESS.poll() is None:
        VLLM_PROCESS.terminate()
        try:
            VLLM_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            VLLM_PROCESS.kill()
        VLLM_PROCESS = None
        return "vLLM stopped\n", 200
    return "vLLM not running\n", 200


# ---------------- Shutdown Flask ----------------
@app.route("/shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Flask server shutting down...\n", 200


if __name__ == "__main__":
    host_ip = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    app.run(host="0.0.0.0", port=5055)
