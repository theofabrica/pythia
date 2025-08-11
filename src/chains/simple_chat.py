# src/chains/simple_chat.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Prompt LCEL : intègre toute la mémoire dans le contexte
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant poli et cordial."),
    *[("user", "{history_user}"), ("assistant", "{history_assistant}")],
    ("user", "{input}")
])

# LLM (API vLLM)
llm = ChatOpenAI(base_url="http://localhost:8000/v1",
                 api_key="not-needed",
                 model="auto")

# LCEL Chain : prompt -> llm
chain: RunnableSequence = prompt | llm


def run_chain(user_input: str,
              history: list[dict],
              **kwargs):                # ← accepte model_path ou autres
    """
    Args:
        user_input: prompt utilisateur
        history   : [{"role":"user"/"assistant", "content": "..."}]
        **kwargs  : ignorés (ex. model_path)

    Returns:
        str : réponse générée
    """
    history_user = [h["content"] for h in history if h["role"] == "user"]
    history_assistant = [h["content"] for h in history if h["role"] == "assistant"]

    result = chain.invoke({
        "input": user_input,
        "history_user": history_user,
        "history_assistant": history_assistant,
    })
    return result.content
