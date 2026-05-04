from fastapi import FastAPI
from query import ask_ai

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Server Running"}

@app.get("/ask")
def ask(question: str):
    try:
        answer = ask_ai(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}