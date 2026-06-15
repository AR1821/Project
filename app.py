from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time


# IMPORT YOUR RAG FUNCTIONS
from rag_pipeline import (
    create_embedding_model,
    load_vector_db,
    retrieve_documents,
    load_llm,
    ask_llm
)

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI()

# ==========================================
# ENABLE CORS
# ==========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ==========================================
# LOAD MODELS ONCE
# ==========================================

embeddings = create_embedding_model()
print("FASTAPI WORKING DIRECTORY:")
print(os.getcwd())

vector_db = load_vector_db(embeddings)
print("VECTOR DB COUNT:", vector_db._collection.count())

llm = load_llm()

# ==================================
# TEMP CHAT MEMORY
# ==================================

chat_memory = {}
session_last_used = {}
SESSION_TIMEOUT = 1800

def remove_expired_sessions():

    current_time = time.time()

    expired = []

    for sid, last_time in session_last_used.items():

        if current_time - last_time > SESSION_TIMEOUT:

            expired.append(sid)

    for sid in expired:

        del chat_memory[sid]
        del session_last_used[sid]

        print(f"Removed session {sid}")

# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    message: str
    session_id: str

# ==========================================
# CHAT API
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    # Remove inactive sessions
    remove_expired_sessions()

    try:

        user_message = request.message
        session_id = request.session_id

        session_last_used[session_id] = time.time()

        if session_id not in chat_memory:

            chat_memory[session_id] = []

        history = chat_memory[session_id]

        print("\n====================")
        print("USER MESSAGE:", user_message)

        # RETRIEVE DOCS

        docs = retrieve_documents(
            vector_db,
            user_message
        )

        # ASK LLM
        response = ask_llm(
            llm,
            docs,
            user_message,
            history
        )

        history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Keep only last 10 messages
        chat_memory[session_id] = history[-10:]

        return {
            "response": response
        }

    except Exception as e:

        import traceback

        print("\n========== ERROR ==========")
        traceback.print_exc()

        return {
            "response": str(e)
        }