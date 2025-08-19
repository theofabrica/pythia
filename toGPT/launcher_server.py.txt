import sys
from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/milvus/up", methods=["POST"])
def start_milvus():
    subprocess.run(
        ["docker", "compose", "--env-file", ".env", "-f", "docker-compose.yml", "up", "-d"],
        cwd="/home/user/ai_gen/Projets/pythia/src/tools/RAG/VectorStore/milvus_local"
    )
    return "Milvus started\n", 200

@app.route("/milvus/down", methods=["POST"])
def stop_milvus():
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "down"],
        cwd="/home/user/ai_gen/Projets/pythia/src/tools/RAG/VectorStore/milvus_local"
    )
    return "Milvus stopped\n", 200

if __name__ == "__main__":
    host_ip = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    app.run(host=host_ip, port=5055)
