from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Prompt LCEL : intègre toute la mémoire dans le contexte
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant poli et cordial."),
    # Boucle sur l’historique (user/assistant)
    *[
        ("user", "{history_user}"),
        ("assistant", "{history_assistant}")
    ],
    ("user", "{input}")
])

# LLM (API vLLM)
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="auto"
)

# LCEL Chain : prompt -> llm
chain: RunnableSequence = prompt | llm

def run_chain(user_input, history,  model_path=None):
    """
    Args:
        user_input (str): prompt utilisateur courant
        history (list[dict]): [{"role": "user"/"assistant", "content": "..."}]
    Returns:
        str: Réponse générée
    """
    # Sépare les tours pour injection dans le prompt (pattern openai)
    history_user = [h["content"] for h in history if h["role"] == "user"]
    history_assistant = [h["content"] for h in history if h["role"] == "assistant"]

    # Appel LCEL
    result = chain.invoke({
        "input": user_input,
        "history_user": history_user,
        "history_assistant": history_assistant,
    })
    return result.content

if __name__ == "__main__":
    import sys
    user_input = sys.argv[1] if len(sys.argv) > 1 else "Dis bonjour !"
    # Exemple : historique factice
    history = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour, comment puis-je vous aider ?"},
    ]
    print(run_chain(user_input, history))
