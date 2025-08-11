from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_milvus.vectorstores.milvus import Milvus

def ingest_pdf(path: str):
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    # 📥 Embeddings TEI (réseau docker)
    embedding = HuggingFaceEndpointEmbeddings(model="http://tei:8010")

    # 🧠 Connexion à Milvus (déplacée ici pour attendre son lancement)
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name="rag_demo",
        connection_args={"host": "milvus-standalone", "port": 19530},
    )

    print(f"[INFO] Chargement du PDF : {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    print(f"[INFO] Chunking…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"[INFO] Ingestion dans Milvus…")
    vectorstore.add_documents(chunks)

    print(f"[OK] PDF indexé dans Milvus ({len(chunks)} chunks)")
    print(f"[PDF_LOADER] Fichier indexé avec succès dans Milvus : {pdf_path}")
