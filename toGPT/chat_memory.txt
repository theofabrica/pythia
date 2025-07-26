# src/chains/chat_memory.py

class ChatMemory:
    def __init__(self):
        self.history = []

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})

    def add_ai_message(self, content):
        self.history.append({"role": "assistant", "content": content})

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []
