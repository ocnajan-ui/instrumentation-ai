import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ================= LOAD ENV =================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY missing")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

# ================= CHROMA =================
client = chromadb.PersistentClient(path="vector_store")

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = client.get_collection(
    name="instrumentation_docs",
    embedding_function=embedding_function
)

# ================= MEMORY =================
chat_history = []

# ================= HELPERS =================
def clean_query(q):
    return q.strip().lower()

def is_casual(q):
    return any(x in q for x in ["hi", "hello", "hey", "how are you"])

def is_general(q):
    general_words = [
        "what are", "fruit", "food", "fan", "animal",
        "vegetable", "orange", "banana", "dry fruits"
    ]
    return any(word in q for word in general_words)

def enhance_query(query):
    keywords = "instrumentation sensor troubleshooting AVL CPC APC489 emission measurement device"
    return f"{query} {keywords}"

# ================= ASK AI =================
def ask_ai(query):
    global chat_history

    query_clean = clean_query(query)

    # ================= 1. CASUAL =================
    if is_casual(query_clean):
        res = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        return res.choices[0].message.content

    # ================= 2. GENERAL QUESTIONS =================
    if is_general(query_clean):
        prompt = f"""
Answer simply in 3-4 lines.

Question:
{query}
"""
        res = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = res.choices[0].message.content
        return "🌐 Answer from INTERNET:\n\n" + answer

    # ================= 3. VECTOR SEARCH =================
    improved_query = enhance_query(query)

    results = collection.query(
        query_texts=[improved_query],
        n_results=5
    )

    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs)

    # ================= 4. SOURCE DECISION =================
    if len(context) > 500:
        source_type = "MANUAL"
    elif len(context) > 100:
        source_type = "HYBRID"
    else:
        source_type = "INTERNET"

    # ================= 5. PROMPT =================
    prompt = f"""
You are an expert instrumentation engineer.

Conversation history:
{chat_history[-3:]}

Rules:
- Answer SHORT and CLEAR
- "what is" → max 3-4 lines
- "why/explain" → max 5-6 lines
- "simple" → easy language
- Continue previous context

Manual Data:
{context}

Question:
{query}
"""

    res = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = res.choices[0].message.content

    # ================= 6. LABEL =================
    if source_type == "MANUAL":
        answer = "📘 Answer from MANUAL:\n\n" + answer
    elif source_type == "HYBRID":
        answer = "📗 Answer from MANUAL + INTERNET:\n\n" + answer
    else:
        answer = "🌐 Answer from INTERNET:\n\n" + answer

    # ================= 7. MEMORY =================
    chat_history.append(f"Q: {query}\nA: {answer}")

    return answer


# ================= MAIN =================
if __name__ == "__main__":
    print("🤖 AI READY (SMART ENGINE MODE)\n")

    while True:
        q = input("💬 Ask: ")

        if q.lower() == "exit":
            break

        ans = ask_ai(q)

        print("\n💡 Answer:\n")
        print(ans)
        print("\n" + "=" * 60 + "\n")