#!/bin/bash
set -e

# Aller à la racine du projet (peu importe d'où le script est lancé)
cd "$(dirname "$0")/.."

# 1. Détecter l'IP de la gateway Docker sur l’hôte
DOCKER_HOST_IP=$(ip route | grep -m1 docker0 | awk '{print $9}')
if [ -z "$DOCKER_HOST_IP" ]; then
    # fallback : méthode classique
    DOCKER_HOST_IP=$(ip route | grep default | awk '{print $3}')
fi

# 2. Écrire cette IP dans un fichier accessible par Pythia
echo "$DOCKER_HOST_IP" > src/ui/docker_host_ip.txt
echo "[INFO] IP hôte Docker détectée : $DOCKER_HOST_IP"

# 3. Lancer le serveur Flask sur CETTE IP
echo "[INFO] Lancement du launcher Milvus sur $DOCKER_HOST_IP..."
python3 src/ui/launcher_server.py "$DOCKER_HOST_IP" &
FLASK_PID=$!

sleep 2
echo "[INFO] Launcher Milvus démarré (PID: $FLASK_PID)."

# 4. Lancer Pythia
echo "[INFO] Lancement de Pythia..."
docker compose -f docker/pythia-compose.yml up -d

echo "[INFO] Tout est lancé."
echo "[INFO] Pour arrêter :"
echo "  kill $FLASK_PID   # pour arrêter le serveur Flask"
echo "  docker compose -f docker/pythia-compose.yml down   # pour arrêter Pythia"
