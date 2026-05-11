import gradio as gr
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    PaliGemmaForConditionalGeneration,
)
from PIL import Image
import re
import os

# ── CPU-only safety: disable flash_attn before anything loads ────────────────
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_FLASH_ATTN"]    = "1"

# ── Device setup ─────────────────────────────────────────────────────────────
DEVICE = "cpu"
DTYPE  = torch.float32
print("Running on: CPU")

# ── Florence-2 (OCR + detection) ─────────────────────────────────────────────
print("Loading Florence-2...")
FLORENCE_ID = "microsoft/Florence-2-base-ft"
florence_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="eager",
).to(DEVICE).eval()
florence_processor = AutoProcessor.from_pretrained(
    FLORENCE_ID,
    trust_remote_code=True,
)
print("Florence-2 ready.")

# ── PaliGemma 3B (scene understanding + summary) ────────────────────────────
print("Loading PaliGemma...")
PALI_ID = "google/paligemma-3b-mix-448"
pali_model = PaliGemmaForConditionalGeneration.from_pretrained(
    PALI_ID,
    torch_dtype=DTYPE,
    attn_implementation="eager",
).to(DEVICE).eval()
pali_processor = AutoProcessor.from_pretrained(PALI_ID)
print("PaliGemma ready.")


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_florence(image: Image.Image, task: str) -> str:
    """Run a Florence-2 task prompt and return decoded text."""
    inputs = florence_processor(
        text=task,
        images=image,
        return_tensors="pt",
    ).to(DEVICE, DTYPE)
    with torch.no_grad():
        output_ids = florence_model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=False,
        )
    raw = florence_processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = florence_processor.post_process_generation(
        raw,
        task=task,
        image_size=(image.width, image.height),
    )
    return result.get(task, "")


def run_paligemma(image: Image.Image, prompt: str) -> str:
    """Run a PaliGemma prompt and return the generated text."""
    inputs = pali_processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = pali_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    return pali_processor.decode(output[0][input_len:], skip_special_tokens=True)


def clean_ocr_text(raw: str) -> str:
    """Tidy up Florence OCR output."""
    if isinstance(raw, dict):
        # OCR_WITH_REGION returns a dict with quad_boxes + labels
        labels = raw.get("labels", [])
        return "\n".join(labels) if labels else ""
    text = str(raw).strip()
    # Remove leftover XML-style tags Florence sometimes emits
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()



def answer_question(image: Image.Image, question: str) -> str:
    """Answer a free-form question about the image using PaliGemma."""
    if not question or not question.strip():
        return "⚠️ Please type a question first."
    if image is None:
        return "⚠️ Please upload a screenshot first."

    # PaliGemma answer en format — matches the prompting style from the docs
    prompt = f"answer en {question.strip()}\n"
    inputs = pali_processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = pali_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    answer = pali_processor.decode(output[0][input_len:], skip_special_tokens=True)
    return answer.strip()


def batch_questions(image: Image.Image, questions_text: str) -> str:
    """Answer multiple newline-separated questions about the image."""
    if image is None:
        return "⚠️ Please upload a screenshot first."
    questions = [q.strip() for q in questions_text.strip().splitlines() if q.strip()]
    if not questions:
        return "⚠️ Please enter at least one question."

    results = []
    for q in questions:
        prompt = f"answer en {q}\n"
        inputs = pali_processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output = pali_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        answer = pali_processor.decode(output[0][input_len:], skip_special_tokens=True)
        results.append(f"**Q: {q}**\n{answer.strip()}")

    return "\n\n---\n\n".join(results)

# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyse_screen(image: Image.Image, mode: str) -> tuple[str, str, str]:
    """
    Full WOYS pipeline.
    Returns (ocr_text, visual_description, summary)
    """
    if image is None:
        return "", "", "⚠️ Please upload a screenshot or image."

    # 1. Extract all text with Florence-2 OCR
    ocr_raw    = run_florence(image, "<OCR>")
    ocr_text   = clean_ocr_text(ocr_raw)

    # 2. Caption / visual description with Florence-2
    if mode == "Quick (OCR + caption only)":
        caption = run_florence(image, "<MORE_DETAILED_CAPTION>")
        summary = (
            "**Extracted text:**\n\n"
            + (ocr_text if ocr_text else "_No text detected_")
            + "\n\n---\n\n**Visual description:**\n\n"
            + str(caption).strip()
        )
        return ocr_text, str(caption).strip(), summary

    # 3. Full mode: also run PaliGemma for rich understanding
    pali_prompt = (
        "This is a screenshot. Describe what application or webpage is shown, "
        "what the user is doing, and summarise any important information visible. "
        "Be concise and structured."
    )
    visual_desc = run_paligemma(image, pali_prompt)

    # 4. Build structured output
    parts = []
    if ocr_text:
        parts.append(f"**Extracted text:**\n\n{ocr_text}")
    else:
        parts.append("**Extracted text:**\n\n_No text detected_")

    parts.append(f"**Visual summary (PaliGemma):**\n\n{visual_desc}")
    summary = "\n\n---\n\n".join(parts)

    return ocr_text, visual_desc, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0d0f12;
    --surface:  #13161b;
    --border:   #1e2530;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --text:     #e8eaf0;
    --muted:    #e6eaf0;
    --radius:   12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.woys-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.woys-logo {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.woys-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.4rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Panels */
.gr-panel, .gr-box, .gradio-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Image upload zone */
.gr-image {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    transition: border-color 0.2s;
}
.gr-image:hover { border-color: var(--accent) !important; }

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.04em;
    transition: opacity 0.15s, transform 0.1s;
}
.gr-button-primary:hover { opacity: 0.88; transform: translateY(-1px); }
.gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}

/* Text areas & labels */
label, .gr-input-label { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
textarea, .gr-text-input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}

/* Radio / dropdown */
.gr-radio-row, .gr-dropdown { color: var(--text) !important; }

/* Tabs */
.gr-tab-nav button { font-family: 'Syne', sans-serif !important; color: var(--muted) !important; }
.gr-tab-nav button.selected { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Footer */
.woys-footer {
    text-align: center;
    padding: 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
"""

HEADER_HTML = """
<div class="woys-header">
    <div class="woys-logo">WOYS</div>
    <div class="woys-sub">What's On Your Screen · AI Image Reader by Prasad Mokal</div>
</div>
"""

FOOTER_HTML = """
<div class="woys-footer">
    Florence-2 (Microsoft · MIT) + PaliGemma 3B (Google) &nbsp;·&nbsp;
    Built with HuggingFace Transformers + Gradio &nbsp;·&nbsp;
    Runs on CPU · no data stored
</div>
"""

EXAMPLES = [
    # Add paths to local example screenshots here if you have them
    # ["examples/terminal.png", "Full (OCR + PaliGemma summary)"],
]

def build_ui():
    with gr.Blocks(css=CSS, title="WOYS – What's On Your Screen") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row():
            # ── Left: inputs ──────────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Drop a screenshot here",
                    height=320,
                )
                mode_radio = gr.Radio(
                    choices=[
                        "Quick (OCR + caption only)",
                        "Full (OCR + PaliGemma summary)",
                    ],
                    value="Full (OCR + PaliGemma summary)",
                    label="Analysis mode",
                )
                with gr.Row():
                    clear_btn  = gr.ClearButton(
                        components=[image_input],
                        value="Clear",
                    )
                    submit_btn = gr.Button("Analyse screen ↗", variant="primary")

            # ── Right: outputs ────────────────────────────────────────────
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        summary_out = gr.Markdown(
                            label="Full analysis",
                            value="_Upload a screenshot and click **Analyse screen** to begin._",
                        )
                    with gr.Tab("Raw OCR text"):
                        ocr_out = gr.Textbox(
                            label="Extracted text (Florence-2)",
                            lines=12,
                            placeholder="OCR output will appear here...",
                        )
                    with gr.Tab("Visual description"):
                        desc_out = gr.Textbox(
                            label="Scene understanding (PaliGemma)",
                            lines=12,
                            placeholder="Visual description will appear here...",
                        )

                    with gr.Tab("❓ Ask a question"):
                        gr.Markdown("Ask anything about the uploaded screenshot. PaliGemma will answer directly.")
                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Your question",
                                placeholder="e.g. What app is shown?  /  What does this error mean?  /  What is the user doing?",
                                lines=2,
                            )
                        ask_btn = gr.Button("Ask ↗", variant="primary")
                        answer_out = gr.Markdown(
                            label="Answer",
                            value="_Type a question and click **Ask** to get an answer._",
                        )
                        gr.Markdown("##### Ask multiple questions at once")
                        multi_input = gr.Textbox(
                            label="Multiple questions (one per line)",
                            placeholder="What app is this?\nWhat error is shown?\nWhat language is the code in?",
                            lines=4,
                        )
                        multi_btn  = gr.Button("Ask all ↗", variant="secondary")
                        multi_out  = gr.Markdown(label="Answers")

        # Wire up
        submit_btn.click(
            fn=analyse_screen,
            inputs=[image_input, mode_radio],
            outputs=[ocr_out, desc_out, summary_out],
        )

        # Wire up Q&A
        ask_btn.click(
            fn=answer_question,
            inputs=[image_input, question_input],
            outputs=[answer_out],
        )
        multi_btn.click(
            fn=batch_questions,
            inputs=[image_input, multi_input],
            outputs=[multi_out],
        )

        gr.HTML(FOOTER_HTML)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        share=False,          # set True for a temporary public link
        show_error=True,
    )
