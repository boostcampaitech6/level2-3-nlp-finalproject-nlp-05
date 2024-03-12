from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class LineRequest(BaseModel):
    emotion: Optional[str] = None

class PoemRequest(BaseModel):
    line: Optional[str] = None

app = FastAPI()

template = FileResponse('templates/index.html')

model_path = "./ML/Metaphor_Gen/code/outputs"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.get("/", response_class=HTMLResponse)
async def home():
    return template

@app.post("/api/line")
async def generate_line(request: LineRequest):
    emotion = request.emotion
    inputs = tokenizer(emotion, return_tensors="pt")
    lines = []

    for i in range(3):
        output = model.generate(**inputs, do_sample=True)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        lines.append(decoded_output)

    return { "lines": lines}
    
@app.post("/api/poem")
async def generate_poem(request: PoemRequest):
    line = request.line
    poem = line

    return { "poem": poem}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)