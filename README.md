# Text Summarizer

A dialogue summarization project built with a fine-tuned **T5-small** model trained on the samsum dataset. The model takes a multi-turn conversation as input and outputs a concise summary. It is served through a FastAPI backend with a simple web UI.

## Project Flow

```
text_summarizer.ipynb
        │
        ├── 1. Load samsum dataset (train / test / validation CSVs)
        ├── 2. Clean and tokenize dialogues
        ├── 3. Fine-tune T5-small using HuggingFace Transformers
        └── 4. Save best model checkpoint → saved_summary_model/
                                                    │
                                              app.py (FastAPI)
                                                    │
                                             index.html  (UI)
```

## Files

| File | Description |
|---|---|
| `text_summarizer.ipynb` | Training notebook — data loading, preprocessing, model fine-tuning |
| `app.py` | FastAPI server — loads the saved model and serves the `/summarize/` endpoint |
| `index.html` | Frontend UI — sends dialogue to the API and renders the summary |
| `static/style.css` | Styles for the frontend |

## Dataset

samsum is a dialogue summarization dataset containing ~16k messenger-style conversations paired with human-written summaries.

[samsum dataset — Kaggle](https://www.kaggle.com/datasets/nileshmalode1/samsum-dataset-text-summarization)

Download and place the CSVs inside a `Text Summarizer/` folder before running the notebook:

```
Text Summarizer/
  samsum-train.csv
  samsum-test.csv
  samsum-validation.csv
```

## Setup

Install dependencies:

```bash
pip install torch transformers fastapi uvicorn jinja2 python-multipart pandas
```

**Step 1 — Train the model**

Run `text_summarizer.ipynb` top-to-bottom. The best model checkpoint will be saved to `saved_summary_model/`.

**Step 2 — Start the server**

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser, paste a conversation, and click Summarize.

## Model

T5-small fine-tuned on samsum. The saved weights are not included in this repo (too large). Run the notebook to generate them locally, or drop any compatible T5 checkpoint into `saved_summary_model/`.
