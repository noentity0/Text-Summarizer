# Text Summarizer

Fine-tuned **T5-small** on the SAMSum dialogue dataset to summarize conversations. Served through a FastAPI backend with a minimal web UI.

## How it works

```
text_summarizer.ipynb  →  load SAMSum data  →  fine-tune T5  →  save best model
                                                                        ↓
                                                              app.py (FastAPI)
                                                                        ↓
                                                              index.html (UI)
```

1. **Notebook** — loads the SAMSum dataset, tokenizes dialogues, fine-tunes T5-small, and saves the best checkpoint to `saved_summary_model/`
2. **`app.py`** — FastAPI server that loads the saved model and exposes a `/summarize/` endpoint
3. **`index.html` + `static/style.css`** — minimal frontend that sends dialogue to the API and displays the summary

## Dataset

[SAMSum Corpus — Kaggle](https://www.kaggle.com/datasets/nileshmalode1/samsum-dataset-text-summarization)

Place the CSV files inside a `Text Summarizer/` folder:

```
Text Summarizer/
  samsum-train.csv
  samsum-test.csv
  samsum-validation.csv
```

## Setup

```bash
pip install torch transformers fastapi uvicorn jinja2 python-multipart pandas
```

Run the notebook top-to-bottom to train and save the model, then start the server:

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

## Model

The saved model is not included in this repo due to file size. Train it by running `text_summarizer.ipynb` or load any compatible T5 checkpoint into `saved_summary_model/`.
