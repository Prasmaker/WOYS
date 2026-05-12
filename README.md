\---

title: WOYS
emoji: 👀
colorFrom: purple
colorTo: green
sdk: gradio
sdk\_version: 6.14.0
python\_version: 3.13
app\_file: app.py
pinned: false
---

# WOYS – What's On Your Screen

An AI-powered screenshot reader that extracts, describes, and answers questions about anything visible in an image.

**Live demo →** [**huggingface.co/spaces/Prasmaker/WOYS**](https://huggingface.co/spaces/Prasmaker/WOYS)

\---

## WHAT does it do?

Upload any screenshot and WOYS will tell you what's on it. It combines two Vision-Language Models running in a pipeline — one specialised in reading text accurately, the other in understanding context and answering questions.

There are three modes:

**Analyse** — runs the full pipeline and returns a structured summary of what's on screen, all text extracted, and a visual description of the application or page shown.

**Quick mode** — runs only Florence-2 for fast OCR and a caption, skipping the heavier PaliGemma model. Useful when you just need the text pulled out quickly.

**Ask a question** — type any natural language question about the screenshot and PaliGemma answers it directly. You can also ask multiple questions at once, one per line.

\---

## Working

WOYS uses a two-model pipeline:

**Florence-2** (Microsoft, MIT licence) handles all text extraction. It has a dedicated `<OCR>` task trained specifically to read text in images — including dense UI layouts, code, menus, and error messages — with high accuracy. At 0.23B parameters it is fast and lightweight.

**PaliGemma 3B** (Google) handles scene understanding and question answering. It uses the `answer en` prompt format it was trained on for Visual Question Answering, allowing it to reason about what application is shown, what the user is doing, and answer specific questions about the content.

Both models run on CPU, making WOYS usable on standard hardware without a GPU.

\---

## To run locally on your machine

Clone the repo and install dependencies:

```bash
git clone https://github.com/Prasmaker/WOYS.git
cd WOYS
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser. On first run, the model weights (\~4 GB total) will download automatically and cache locally. Subsequent runs start in under 2 minutes.

\---

## Tech stack used

|Component|Technology|
|-|-|
|OCR model|Microsoft Florence-2-base-ft|
|Vision-language model|Google PaliGemma 3B|
|Framework|PyTorch (CPU)|
|Model loading|HuggingFace Transformers|
|UI|Gradio|
|Deployment|HuggingFace Spaces|
|CI/CD|GitHub Actions|

\---

## Project structure

```
WOYS/
├── app.py                  # Full application — models, inference pipeline, Gradio UI
├── requirements.txt        # Pinned dependencies for HF Spaces CPU tier
├── README.md               # This file
└── .github/
    └── workflows/
        └── sync-to-hf.yml  # Auto-syncs GitHub → HuggingFace Spaces on every push
```

\---

## Limitations

Inference runs on CPU, so each analysis takes 15–45 seconds depending on image complexity. PaliGemma answers are best for factual, visual questions — it is not a general-purpose chatbot. Very low-resolution or heavily compressed screenshots may reduce OCR accuracy.

\---

*Built with HuggingFace Transformers and Gradio. Models are loaded from the HuggingFace Hub at runtime — no weights are stored in this repository.*

