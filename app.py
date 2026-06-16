# fastapi -> python based web-framework
# uvicorn -> light wieght webserver
# client(UI) ----dialogue----> server(has model -> summary)
# server     -----summary----> client(UI)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarizer using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

#templating
templates = Jinja2Templates(directory=".")

# Input Schema for the dialogue => String
class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = str(text)
    text = re.sub(r'\r\n', ' ', text) # lines
    text = re.sub(r'\s+', ' ', text)   # spaces
    text = re.sub(r'<.*?>', ' ', text)  # html tags
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        min_length=35,
        max_length=100,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=False
    )

    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary

app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")