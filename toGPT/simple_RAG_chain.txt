from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Milvus
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from operator import itemgetter

# 📥 Embeddings TEI local (réseau docker -> nom de service)
embedding = HuggingFaceEndpointEmbeddings(model="http://tei:8010")

# 🧠 Vector store Milvus (collection créée par RAG_tools)
vectorstore = Milvus(
    embedding_function=embedding,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="rag_demo",
)
retriever = vectorstore.as_retriever()

# 🧾 Prompt de réponse fondée sur le contexte
template = """Tu es un expert. Utilise le contexte suivant pour répondre à la question.

Contexte :
{context}

Question :
{question}

Réponse :"""
prompt = PromptTemplate.from_template(template)

# 🤖 LLM via vLLM exposé en HTTP (attention : chemin complet !)
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1/chat/completions",
    api_key="not-needed",
    model="auto"
)

# 🔁 Chaîne RAG LCEL : {question} -> retriever -> prompt -> llm -> str
rag_chain: RunnableSequence = (
    RunnableMap({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }) | prompt | llm | StrOutputParser()
)

# ---------------------------------------------------------------------
# Interface imposée par l'UI : run_chain(prompt, history) -> str
def run_chain(user_input: str, _history=None, model_path=None) -> str:
    return rag_chain.invoke({"question": user_input})
# ---------------------------------------------------------------------

# 🧪 Exécution isolée
if __name__ == "__main__":
    q = "Qui est le personnage principal ?"
    print("[TEST] Question :", q)
    print("[TEST] Réponse :", run_chain(q))
