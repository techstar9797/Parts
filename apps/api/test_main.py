from fastapi import FastAPI

app = FastAPI(title="PartSync-Llama API", version="0.1.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "PartSync-Llama API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
