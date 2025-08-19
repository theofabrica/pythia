import subprocess
import json
from pymilvus import connections, Collection
from pathlib import Path

MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_demo"

def run_cmd(cmd):
    """Exécute une commande shell et retourne stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def inspect_container(container_name):
    """Retourne l'inspect complet d'un conteneur Docker en JSON."""
    result = run_cmd(["docker", "inspect", container_name])
    if not result:
        print(f"[ERREUR] Impossible d'inspecter {container_name}")
        return None
    return json.loads(result)[0]

def du_size(path):
    """Retourne la taille humaine du chemin."""
    if not Path(path).exists():
        return "[inexistant]"
    size = run_cmd(["du", "-sh", path])
    return size if size else "[erreur]"

def get_upperdir_and_mounts(container_name):
    """Affiche UpperDir, WorkDir, Mounts et tailles associées."""
    data = inspect_container(container_name)
    if not data:
        return

    print(f"\n=== Infos stockage pour {container_name} ===")

    # UpperDir & WorkDir
    graphdriver_data = data.get("GraphDriver", {}).get("Data", {})
    if "UpperDir" in graphdriver_data:
        upath = graphdriver_data['UpperDir']
        print(f"UpperDir: {upath}  (taille: {du_size(upath)})")
    if "WorkDir" in graphdriver_data:
        wpath = graphdriver_data['WorkDir']
        print(f"WorkDir: {wpath}  (taille: {du_size(wpath)})")

    # Mounts
    mounts = data.get("Mounts", [])
    if mounts:
        print("\n--- Mounts ---")
        for m in mounts:
            src = m.get('Source')
            print(f"Type: {m.get('Type')}, Source: {src}, Dest: {m.get('Destination')}, taille: {du_size(src)}")

def show_collection_info():
    """Affiche le nombre d'entités dans la collection Milvus."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    print(f"\n=== Collection Milvus ===")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Nombre d'entités : {collection.num_entities}")

if __name__ == "__main__":
    # 1. Infos Milvus
    show_collection_info()

    # 2. Infos Docker sur MinIO et Milvus
    get_upperdir_and_mounts("milvus-minio")
    get_upperdir_and_mounts("milvus-standalone")
