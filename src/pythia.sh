#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# -------------------------------------------------------------
# Configuration X11 pour l’UI Qt (permet l’affichage depuis Docker)
# Si DISPLAY n’est pas défini, on force :0 (affichage local)
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "[INFO] DISPLAY non défini, utilisation de :0"
fi
# Autoriser les conteneurs locaux à utiliser le serveur X11
if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || true
fi
# -------------------------------------------------------------


# 0. Autoriser le port 5055 côté firewall
sudo ufw allow 5055/tcp

# 1. Détecter l'IP LAN (réseau local)
DOCKER_HOST_IP=$(hostname -I | awk '{print $1}')
[ -z "$DOCKER_HOST_IP" ] && DOCKER_HOST_IP=$(ip addr show docker0 | grep -Po 'inet \K[\d.]+')
[ -z "$DOCKER_HOST_IP" ] && DOCKER_HOST_IP=$(ip route | grep default | awk '{print $3}')

# 2. Sauvegarder l'IP
echo "$DOCKER_HOST_IP" > src/ui/docker_host_ip.txt
echo "[INFO] IP hôte détectée : $DOCKER_HOST_IP"

# 3. Fonction de nettoyage
cleanup() {
    echo "[INFO] Arrêt de Pythia et Flask..."
    docker compose -f docker/pythia-compose.yml down || true
    kill $FLASK_PID 2>/dev/null || true
    exit 0
}

# 4. Trap sur CTRL+C et EXIT
trap cleanup SIGINT SIGTERM EXIT

# 5. Lancer Flask
echo "[INFO] Lancement du launcher Milvus sur $DOCKER_HOST_IP..."
python3 src/ui/launcher_server.py "$DOCKER_HOST_IP" &
FLASK_PID=$!
sleep 2
echo "[INFO] Launcher Milvus démarré (PID: $FLASK_PID)."

# 6. Lancer Pythia
echo "[INFO] Lancement de Pythia..."
docker compose -f docker/pythia-compose.yml up -d
echo "[INFO] Tout est lancé."

# 7. Attendre que le container s'arrête
docker wait pythia
