# ============================================================
# COMPLETE RAG PIPELINE
# PDF + WEBSITE → CHUNKING → EMBEDDING → VECTOR DB → LLM
# ============================================================

# ============================================================
# STEP 1 : INSTALL REQUIRED LIBRARIES
# ============================================================

# pip install langchain
# pip install langchain-community
# pip install langchain-core
# pip install langchain-text-splitters
# pip install langchain-huggingface
# pip install sentence-transformers
# pip install chromadb
# pip install pypdf
# pip install beautifulsoup4
# pip install requests
# pip install ollama


# ============================================================
# STEP 2 : IMPORT LIBRARIES
# ============================================================

import os
import json
import requests

from bs4 import BeautifulSoup

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding Model
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from langchain_community.vectorstores import Chroma

# Document Object
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# STEP 3 : PDF EXTRACTION

def extract_pdf(pdf_path):

    print("Extracting PDF Data...")

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    documents = []

    for page in pages:

        documents.append(
            {
                "source": pdf_path,
                "page": page.metadata.get("page"),
                "content": page.page_content
            }
        )

    return documents


# ============================================================
# STEP 4 : WEBSITE EXTRACTION
# ============================================================

def extract_website(url):

    print("Extracting Website Data...")

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    documents = []

    documents.append(
        {
            "source": url,
            "content": text
        }
    )

    return documents


# ============================================================
# STEP 5 : SAVE RAW DATA INTO JSON
# ============================================================

def save_json(data, filename):

    print("Saving JSON File...")

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON Saved : {filename}")


# ============================================================
# STEP 6 : LOAD JSON DATA
# ============================================================

def load_json(filename):

    print("Loading JSON Data...")

    with open(filename, "r", encoding="utf-8") as f:

        data = json.load(f)

    return data


# ============================================================
# STEP 7 : CONVERT JSON TO DOCUMENTS
# ============================================================

def convert_to_documents(json_data):

    print("Converting JSON to LangChain Documents...")

    docs = []

    for item in json_data:

        doc = Document(

            page_content=item["content"],

            metadata={
                "source": item.get("source", ""),
                "page": item.get("page", "")
            }
        )

        docs.append(doc)

    return docs


# ============================================================
# STEP 8 : CHUNKING
# ============================================================

def chunk_documents(documents):

    print("Chunking Documents...")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks Created : {len(chunks)}")

    return chunks


# ============================================================
# STEP 9 : CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    print("Loading Embedding Model...")

    embeddings = HuggingFaceEmbeddings(

        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings




# ============================================================
# STEP 10 : STORE EMBEDDINGS IN VECTOR DB
# ============================================================

def create_vector_db(chunks, embeddings):

    print("Creating Vector Database...")
    vector_db = Chroma.from_documents(
         documents=chunks,
         embedding=embeddings,
         persist_directory=CHROMA_PATH
        )

  

    vector_db.persist()

    print("Vector DB Created Successfully")

    return vector_db


# ============================================================
# STEP 11 : LOAD EXISTING VECTOR DB
# ============================================================

def load_vector_db(embeddings):

    print("Loading Existing Vector Database...")

    print("CURRENT WORKING DIRECTORY:")
    print(os.getcwd())
    vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

    
    print("VECTOR DB LOADED")

    return vector_db


# ============================================================
# STEP 12 : RETRIEVAL FUNCTION
# ============================================================
def retrieve_documents(vector_db, query):

    print("Searching Relevant Documents...")

    results = vector_db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    filtered_docs = []

    for i, doc in enumerate(results):

        print(f"\n===== DOC {i+1} =====")
        print(doc.page_content[:500])

        filtered_docs.append(doc)

    print("RETRIEVED:", len(filtered_docs))

    return filtered_docs

# ============================================================
# STEP 13 : LOAD LLM
# ============================================================
def load_llm():

    print("Loading OpenAI...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=300
    )

    return llm


def ask_llm(llm, retrieved_docs, question, history):

    print("NEW ASK_LLM EXECUTING")
    print("Generating AI Response...")

    # ==========================================
    # 1. FALLBACK ONLY IF NO DOCS ARE RETRIEVED
    # ==========================================
    if len(retrieved_docs) == 0:
        return "I can only answer questions related to Sabudh Foundation and its offerings."

    # ==========================================
    # 2. BUILD CONTEXT (Pass full content, do not slice at :700)
    # ==========================================
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("\n========== CONTEXT ==========\n")
    print(context[:1500])

    history_text = ""
    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    # ==========================================
    # 3. SYSTEM PROMPT (Enforces strict domain control)
    # ==========================================
    prompt = f"""
You are Sabudh AI, the official assistant of Sabudh Foundation.
Your role is to answer questions strictly regarding Sabudh Foundation and its offerings based on the provided context.

CRITICAL INSTRUCTIONS:
1. The user input might contain spelling mistakes (e.g., 'data analitics' instead of 'Data Analytics') or short abbreviations (e.g., 'AI'). Use the CONTEXT to infer what they mean and answer anyway.
2. If the user asks a short follow-up question like "tell me more", "explain further", or "what is the duration?", look at the PREVIOUS CONVERSATION to see which program was being discussed, look it up in the CONTEXT, and provide a helpful, natural response.
3. Do NOT repeat the user's question back to them.
4. Do NOT combine multiple points into a single paragraph block. Use a line break between items if necessary to keep them separated.
5.If and ONLY if the question is completely unrelated to Sabudh Foundation (e.g., "how to bake a cake" or "who is the prime minister"), reply EXACTLY with this line and nothing else:
I can only answer questions related to Sabudh Foundation and its offerings.

PREVIOUS CONVERSATION
{history_text}

CONTEXT
{context}

QUESTION
{question}

ANSWER
"""

    try:
        response = llm.invoke(prompt)
        print("\n========== FINAL RESPONSE ==========\n")
        print(response.content)
        return response.content

    except Exception as e:
        print("LLM ERROR:", e)
        return "Model generation failed."
# ============================================================
# STEP 15 : MAIN PIPELINE
# ============================================================

def main():

    # ========================================================
    # PDF EXTRACTION
    # ========================================================

    pdf_data = extract_pdf("FAQ.pdf")

    # ========================================================
    # WEBSITE EXTRACTION
    # ========================================================

    website_data = extract_website("https://sabudh.org/")

    # ========================================================
    # COMBINE DATA
    # ========================================================

    combined_data = pdf_data + website_data

    # ========================================================
    # SAVE JSON
    # ========================================================

    save_json(combined_data, "knowledge_base.json")

    # ========================================================
    # LOAD JSON
    # ========================================================

    json_data = load_json("knowledge_base.json")

    # ========================================================
    # CONVERT TO DOCUMENTS
    # ========================================================

    documents = convert_to_documents(json_data)

    # ========================================================
    # CHUNKING
    # ========================================================

    chunks = chunk_documents(documents)

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embeddings = create_embedding_model()

    # ========================================================
    # VECTOR DATABASE
    # ========================================================

    vector_db = create_vector_db(

        chunks,
        embeddings
    )

    # ========================================================
    # LOAD LLM
    # ========================================================

    llm = load_llm()

    # ========================================================
    # CHAT LOOP
    # ========================================================

    while True:

        question = input("\nAsk Question : ")

        if question.lower() == "exit":
            break

        # RETRIEVE DOCUMENTS

        retrieved_docs = retrieve_documents(

            vector_db,
            question
        )

# GENERATE RESPONSE

        response = ask_llm(

            llm,
            retrieved_docs,
            question,
            []
        )

        print("\nAI RESPONSE:\n")

        print(response)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()