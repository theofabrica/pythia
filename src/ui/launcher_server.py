import sys
from flask import Flask, request
import subprocess

app = Flask(__name__)

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
