# render_proxy.py - Deploy to Render's free tier
from fastapi import FastAPI
from transformers import pipeline
import time

app = FastAPI()

# Load model on demand (sleeps after 15 min idle)
model_cache = {}
def get_model(name):
    if name not in model_cache or time.time() - model_cache[name]['time'] > 900:
        model_cache[name] = {
            'model': pipeline(name, model=f"distilbert-base-uncased"),
            'time': time.time()
        }
    return model_cache[name]['model']

@app.post("/hf/{task}")
async def hf_query(task: str, text: str):
    model = get_model(task)  # task = "sentiment-analysis", "ner", etc.
    result = model(text)[0]
    return {"output": result}

# Run: uvicorn render_proxy:app --port 8000
