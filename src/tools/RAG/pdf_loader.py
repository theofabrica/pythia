import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_milvus.vectorstores.milvus import Milvus
from pymilvus import connections, utility, Collection

log = logging.getLogger(__name__)


# Taille max par lot d'envoi vers TEI (doit respecter MAX_CLIENT_BATCH_SIZE côté TEI)
MAX_BATCH_SIZE = 512

def ingest_pdf(path: str) -> dict:
    """
    Ingestion d'un PDF dans Milvus via TEI avec batching.
    Retourne un dict: {ok: bool, chunks: int, collection: str, msg: str}
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        return {
            "ok": False,
            "chunks": 0,
            "collection": "rag_demo",
            "msg": f"PDF introuvable: {pdf_path}"
        }

    collection_name = "rag_demo"

    # 1) Connexion Milvus
    log.info(f"[RAG] Connexion à Milvus sur milvus-standalone:19530…")
    try:
        connections.connect(alias="default", host="milvus-standalone", port="19530")
        log.info(f"[RAG] Milvus connecté. Version: {utility.get_server_version()}")
    except Exception as e:
        log.error(f"[RAG] Erreur connexion Milvus: {e}")
        return {
            "ok": False,
            "chunks": 0,
            "collection": collection_name,
            "msg": f"Milvus unreachable: {e}"
        }

    # 2) Embeddings TEI
    log.info(f"[RAG] Initialisation embeddings TEI @ http://tei:80")
    embedding = HuggingFaceEndpointEmbeddings(model="http://tei:80")

    # 3) Chargement PDF
    log.info(f"[RAG] Chargement PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # 4) Chunking
    log.info(f"[RAG] Chunking en cours…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    log.info(f"[RAG] Nombre de chunks générés: {len(chunks)}")

    if not chunks:
        return {
            "ok": False,
            "chunks": 0,
            "collection": collection_name,
            "msg": "Aucun chunk généré"
        }

    for d in chunks:
        d.metadata = dict(d.metadata or {})
        d.metadata["source_path"] = str(pdf_path)

    # 5) VectorStore Milvus (réutilise la connexion alias "default")
    log.info(f"[RAG] Création VectorStore Milvus collection={collection_name} (alias=default)")
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={"alias": "default"}
    )

    # 6) Insertion par batch
    try:
        log.info(f"[RAG] Insertion des chunks en batch de {MAX_BATCH_SIZE} max…")
        total_inserted = 0
        for i in range(0, len(chunks), MAX_BATCH_SIZE):
            batch = chunks[i:i + MAX_BATCH_SIZE]
            vectorstore.add_documents(batch)
            total_inserted += len(batch)
            log.info(f"[RAG] Batch {i//MAX_BATCH_SIZE + 1} inséré ({len(batch)} chunks)")

        log.info(f"[RAG] Insertion terminée. Total inséré: {total_inserted}")
    except Exception as e:
        log.error(f"[RAG] Erreur insertion Milvus: {e}")
        return {
            "ok": False,
            "chunks": total_inserted,
            "collection": collection_name,
            "msg": f"Ingestion failed: {e}"
        }

        # 7) Vérification collection + compactage
    try:
        if not utility.has_collection(collection_name):
            log.error(f"[RAG] Collection absente après ingestion.")
            return {
                "ok": False,
                "chunks": total_inserted,
                "collection": collection_name,
                "msg": "Collection missing post-ingest"
            }

        col = Collection(collection_name)
        col.load()
        total_entities = col.num_entities
        log.info(f"[RAG] Vérif Milvus: {total_entities} entités présentes dans {collection_name}")

        # 8) Flush + compact systématiques
        log.info(f"[RAG] Flush de la collection {collection_name}…")
        col.flush()
        log.info(f"[RAG] Compactage de la collection {collection_name}…")
        col.compact()
        log.info(f"[RAG] Flush + compact terminés pour {collection_name}")

        # <<< ICI LE RETURN FINAL >>>
        return {
            "ok": True,
            "chunks": total_inserted,
            "collection": collection_name,
            "msg": f"Ingestion terminée, {total_entities} entités présentes"
        }

    except Exception as e:
        return {
            "ok": False,
            "chunks": total_inserted,
            "collection": collection_name,
            "msg": f"Milvus check/flush/compact failed: {e}"
        }
