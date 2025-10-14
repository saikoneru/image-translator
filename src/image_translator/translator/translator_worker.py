from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import traceback

app = FastAPI(title="Translation Worker")

# ---------------------------------------------------
# 🧠 Model Initialization
# ---------------------------------------------------
CACHE_DIR = "/export/data1/skoneru/hf_cache"
MODEL_NAME = "ByteDance-Seed/Seed-X-PPO-7B"

lang_map = {
    "English": "en", "Chinese": "zh",  "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl",
    "Romanian": "ro", "Turkish": "tr"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


# ---------------------------------------------------
# 🔧 Utility Functions
# ---------------------------------------------------
def create_batches(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def generate(texts, src_lang, tgt_lang):
    text_batches = create_batches(texts, batch_size=8)
    hyps = []

    for batch in text_batches:
        prompts = [
            f"Translate the sentence from {src_lang} to {tgt_lang}: {text} <{lang_map[tgt_lang]}> "
            for text in batch
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=5,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        hyps.extend(decoded)

    filtered_hyps = []
    for text, hyp in zip(texts, hyps):
        if "-" not in text:
            hyp = hyp.replace("-", " ")
        filtered_hyps.append(hyp)

    return filtered_hyps


# ---------------------------------------------------
# ⚡ FastAPI Endpoint
# ---------------------------------------------------
@app.post("/translate")
async def translate_endpoint(
    texts_json: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...)
):
    """
    Input:
      - texts_json: JSON array of texts or OCR-merged results.
      - src_lang / tgt_lang: Language names matching lang_map keys.
    """
    try:
        texts = json.loads(texts_json)

        if not isinstance(texts, list):
            return JSONResponse(
                content={"error": "texts_json must be a list of strings."},
                status_code=400,
            )

        translations = generate(texts, src_lang, tgt_lang)
        return JSONResponse(content={"translations": translations})

    except Exception as e:
        print("❌ Translation Worker Error:")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004, reload=False)
