# ssh people_logic@192.168.1.112      --->1234plbs
# same as 2.1 but with streaming!
import os
import pdfplumber
import logging
import hashlib
import json
import nltk
import sqlite3
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi
from functools import lru_cache
import re
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="WARNING! top_p is not default parameter.")
warnings.filterwarnings("ignore", category=UserWarning, message="WARNING! presence_penalty is not default parameter.")
warnings.filterwarnings("ignore", category=UserWarning, message="WARNING! frequency_penalty is not default parameter.")

# Let's store all Q&A
DB_FILE = "qa_logs.db"  # SQLite database file

def init_db():
    """Creates the Q&A table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        retrieved_docs TEXT,
        final_answer TEXT,
        feedback TEXT,  -- New column for feedback
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

# ✅ Call this at startup to ensure the DB is ready
init_db()

def save_qa_to_db(question, retrieved_docs, final_answer, feedback=None):
    """Stores a Q&A pair in SQLite with optional feedback."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO qa_history (question, retrieved_docs, final_answer, feedback) 
        VALUES (?, ?, ?, ?)
        """, (question, retrieved_docs, final_answer, feedback))
        conn.commit()
        conn.close()
        logging.info(f"✅ Q&A stored in SQLite: {question} -> {final_answer[:50]}...")
    except Exception as e:
        logging.error(f"❌ Error saving Q&A to DB: {e}")

# Load environment variables
load_dotenv()
nltk.download("punkt")

# --- 1. Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hr-knowledge-base"
POLICIES_FOLDER = "HR_docs/"

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- 3. Initialize BM25 for Keyword Search ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Adjusted chunk size
    chunk_overlap=50  # Adjusted chunk overlap
)

# ✅ Define embeddings at the top
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Embedding Dimension: 384

# --- 4. Pinecone Initialization ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        logging.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        from pinecone import ServerlessSpec
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    index_stats = index.describe_index_stats()
    num_vectors = index_stats.get("total_vector_count", 0)
    if num_vectors == 0:
        logging.warning("⚠️ Pinecone index is empty! Re-inserting embeddings...")
        # --- Step 3: Reload documents & reinsert embeddings ---
        documents = []  # Store text chunks
        table_chunks = []  # Store table chunks separately
        for filename in os.listdir(POLICIES_FOLDER):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(POLICIES_FOLDER, filename)
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        tables = page.extract_tables()  # ✅ Extract tables separately
                        # ✅ Split text into chunks & store
                        text_chunks = text_splitter.split_text(text)
                        documents.extend(text_chunks)
                        # ✅ Process extracted tables
                        for table in tables:
                            df = pd.DataFrame(table[1:], columns=table[0])  # Convert table to DataFrame
                            table_markdown = df.to_markdown()  # Convert to Markdown
                            table_chunks.append(table_markdown)  # Store as Markdown table
        # ✅ Insert both text and table chunks into Pinecone
        all_chunks = documents + table_chunks  # Merge text & tables
        vectorstore = LangchainPinecone.from_texts(all_chunks, embeddings, index_name=PINECONE_INDEX_NAME)
        logging.info(f"✅ Successfully inserted {len(all_chunks)} chunks (text + tables) into Pinecone.")
    else:
        logging.info(f"✅ Pinecone index already contains {num_vectors} vectors.")
    logging.info("Pinecone initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Pinecone: {e}")
    raise

def format_bold_numbered_points(text):
    """Ensure numbered points (1., 2., etc.) have their entire heading bold while preserving Markdown formatting."""
    pattern = r'(\d+\.\s)([^\n]+)'  # Match "1. Heading Text"
    formatted_text = re.sub(pattern, r'**\1\2**', text)  # Make the number and heading bold
    return formatted_text

# --- 4. Initialize Embeddings & Vectorstore ---
vectorstore = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

def build_bm25_index(folder_path):
    """Builds BM25 index from policy documents while preserving tables."""
    global bm25_index, bm25_corpus
    all_texts = []  # Store text chunks
    table_chunks = []  # Store table markdown chunks
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    tables = page.extract_tables()  # ✅ Extract tables separately
                    # ✅ Split text into chunks & store
                    text_chunks = text_splitter.split_text(text)
                    all_texts.extend(text_chunks)
                    # ✅ Process extracted tables
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])  # Convert table to DataFrame
                        table_markdown = df.to_markdown()  # Convert to Markdown
                        table_chunks.append(table_markdown)  # Store as Markdown table
    # ✅ Combine text & tables for indexing
    all_chunks = all_texts + table_chunks
    bm25_corpus = [text.split() for text in all_chunks]
    bm25_index = BM25Okapi(bm25_corpus)
    logging.info(f"✅ BM25 index built with {len(bm25_corpus)} document chunks.")

# --- 6. Query Expansion with LLM ---
def expand_query_with_llm(question):
    """Expands user query using LLM to include synonyms but retains original meaning."""
    expansion_prompt = f"""
    Provide alternative phrasings and related terms for: '{question}', 
    ensuring the original word is always included. Include HR-specific terms if applicable.
    """
    try:
        expanded_query = llm.invoke(expansion_prompt).content
        logging.info(f"🔍 Query Expansion: {expanded_query}")
        return expanded_query
    except Exception as e:
        logging.error(f"❌ Query Expansion Failed: {e}")
        return question  # Fall back to the original question

# --- 7. Hybrid Search (BM25 + Pinecone) ---
def hybrid_search(question):
    """Performs hybrid retrieval using BM25 and Pinecone, ensuring tables are preserved."""
    global bm25_index, bm25_corpus
    expanded_query = expand_query_with_llm(question)
    # ✅ Step 1: BM25 Keyword Search
    bm25_texts = []
    if bm25_index and bm25_corpus:
        bm25_results = bm25_index.get_top_n(expanded_query.split(), bm25_corpus, n=5)  # Increased n to 5
        bm25_texts = [" ".join(text[:200]) for text in bm25_results]  # Limit text per doc
        logging.info(f"🔍 BM25 Retrieved {len(bm25_texts)} results.")
    # ✅ Step 2: Pinecone Semantic Search
    pinecone_results = retriever.invoke(expanded_query)
    pinecone_texts = [doc.page_content for doc in pinecone_results[:5]]  # Increased n to 5
    # ✅ Step 3: Identify if retrieved text contains tables
    table_texts = [text for text in pinecone_texts if "|" in text]  # Detect Markdown tables
    # ✅ Step 4: Merge Results (Prioritize Tables)
    combined_results = table_texts + bm25_texts + pinecone_texts  # Keep tables first
    final_text = "\n\n".join(combined_results)[:5000]  # Limit text size
    return final_text

# --- 8. Setup LLM & Retrieval Chain ---
llm = ChatGroq(
    model_name="qwen-2.5-32b",  # "llama-3.1-8b-instant", "deepseek-r1-distill-qwen-32b"
    api_key=GROQ_API_KEY,
    temperature=0.3,  # Lower temperature for deterministic responses
    max_tokens=1024,  # Ensure sufficient length for detailed answers
    top_p=0.9,  # Focus on high-probability tokens
    presence_penalty=0.6,  # Encourage diverse content
    frequency_penalty=0.6,  # Avoid repetitive phrases
    streaming=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# --- 9. Flask API Setup ---
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@lru_cache(maxsize=100)
def cached_qa(question):
    return qa.invoke({"query": question})

ACRONYM_MAP = {
    "wfh": "work from home policy",
    "pto": "paid time off policy",
    "loa": "leave of absence policy",
    "nda": "non-disclosure agreement",
    "od": "on duty policy",
    "hrbp": "human resources business partner",
    "kra": "KRA Policy - Promoting Transparency",
    "regularization": "Time change Request/ Regularization",
    "regularisation": "Time change Request/ Regularization",
    "posh": "Policy On Prevention of Sexual Harassment",
    "appraisal": "PERFORMANCE APPRAISAL & PROMOTION POLICY",
    "promotion": "PERFORMANCE APPRAISAL & PROMOTION POLICY",
    "prep": "Performance Review & Enhancement Program",
    "Grade": "GRADE STRUCTURE & FLEXIBILITY",
    "leave": "LEAVE POLICY",
    "nda": "Non Compete and Non Disclosure",
    "Office timings": "Office Timing and Attendance Policy",
    "pet": "pet policy",
    "sprint": "Weekly Sprint Policy",
    "work ethics": "WORK PLACE ETHICS"
}

@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "").strip().lower()
        mode = data.get("mode", "RAG").upper()

        words = question.split()
        expanded_question = " ".join([ACRONYM_MAP.get(word, word) for word in words])
        logging.info(f"🔍 Expanded Query: {expanded_question}")

        if not question:
            return jsonify({"error": "Missing 'question' parameter"}), 400

        logging.info(f"📩 User Question: {question}")

        # Greetings, bot info, and creator checks remain the same

        retrieved_docs = hybrid_search(expanded_question)

        # Check for relevant documents
        if len(retrieved_docs.strip()) < 50:
            rag_answer = "No matching documents found in the knowledge base."
            external_answer = ""
        else:
            llm_input = f"""
            Answer the following question using the provided context. If the answer is not fully contained within the context, supplement it with your knowledge. Clearly separate your response into sections:

            **RAG (Retrieved Documents):** [Answer from retrieved documents only]
            **External Knowledge:** [Answer supplementing from external knowledge]

            Context: {retrieved_docs[:5000]}
            Question: {question}
            Answer:
            """

            def generate():
                try:
                    full_response = ""
                    for chunk in llm.stream(llm_input):
                        chunk_content = chunk.content
                        full_response += chunk_content
                        yield chunk_content

                    try:
                        rag_section, external_section = full_response.split("**External Knowledge:**")
                        rag_answer = rag_section.split("**RAG (Retrieved Documents):**")[1].strip()
                        external_answer = external_section.strip()
                    except IndexError:
                        rag_answer = ""
                        external_answer = full_response

                    final_answer = {
                        "rag": rag_answer,
                        "external": external_answer
                    }
                    save_qa_to_db(question, retrieved_docs[:500], json.dumps(final_answer))
                    yield json.dumps(final_answer)
                except Exception as e:
                    logging.error(f"❌ Streaming Error: {e}")
                    yield "An error occurred while generating the response."

            return Response(stream_with_context(generate()), mimetype="application/json")

        # Return JSON response even if no docs are found
        final_answer = {
            "rag": rag_answer,
            "external": external_answer
        }
        save_qa_to_db(question, retrieved_docs[:500], json.dumps(final_answer))
        return jsonify(final_answer)

    except Exception as e:
        logging.error(f"❌ API Error: {e}")
        return jsonify({"error": "An error occurred"}), 500
    





@app.route("/api/update_index", methods=["POST"])
def update_index_api():
    """Manually refresh the Pinecone & BM25 index."""
    try:
        build_bm25_index(POLICIES_FOLDER)
        return jsonify({"message": "Indexes updated successfully"}), 200
    except Exception as e:
        logging.error(f"Index Update Error: {e}")
        return jsonify({"error": "Failed to update indexes"}), 500

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Handles user feedback for responses."""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        feedback = data.get("feedback", "").lower()  # "positive" or "negative"
        if not question or not answer or feedback not in ["positive", "negative"]:
            return jsonify({"error": "Invalid feedback data"}), 400
        # Store feedback in the database
        save_qa_to_db(question, "N/A", answer, feedback)
        logging.info(f"✅ Feedback stored: {question} -> {feedback}")
        return jsonify({"message": "Feedback received. Thank you!"}), 200
    except Exception as e:
        logging.error(f"❌ Feedback Submission Error: {e}")
        return jsonify({"error": "Failed to store feedback"}), 500

# --- 10. Run Flask Server ---
if __name__ == "__main__":
    build_bm25_index(POLICIES_FOLDER)  # ✅ Build BM25 at startup
    app.run(debug=False, port=5000)