from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
import numpy as np
import json
import traceback
import os
import uvicorn

# ===================================================
# 🧩 Base Worker Interface
# ===================================================
class BaseWorker(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return standardized output.
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input format."""
        pass

    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output format."""
        pass


# ===================================================
# 🧠 Translation Worker Implementation
# ===================================================
class TranslationWorker(BaseWorker):
    def __init__(self):
        self.CACHE_DIR = "/app/models/"
        self.MODEL_NAME = "ByteDance-Seed/Seed-X-PPO-7B"

        self.lang_map = {
            "English": "en", "Chinese": "zh",  "French": "fr", "German": "de",
            "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
            "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
            "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
            "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
            "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl",
            "Romanian": "ro", "Turkish": "tr"
        }

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, cache_dir=self.CACHE_DIR)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            cache_dir=self.CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # ------------------------------
    # Input Validation
    # ------------------------------
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return (
            isinstance(input_data.get("texts"), list)
            and isinstance(input_data.get("src_lang"), str)
            and isinstance(input_data.get("tgt_lang"), str)
            and all(isinstance(t, str) for t in input_data["texts"])
        )

    # ------------------------------
    # Output Validation
    # ------------------------------
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        return (
            "translations" in output_data
            and "unfiltered_translations" in output_data
            and isinstance(output_data["translations"], list)
            and isinstance(output_data["unfiltered_translations"], list)
        )

    # ------------------------------
    # Core Translation Logic
    # ------------------------------
    def create_batches(self, texts: List[str], batch_size: int):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    def _generate(self, texts: List[str], src_lang: str, tgt_lang: str) -> Dict[str, List[str]]:
        text_batches = self.create_batches(texts, batch_size=4)
        hyps = []

        for batch in text_batches:
            prompts = [
                f"Translate the sentence from {src_lang} to {tgt_lang}: {text} <{self.lang_map[tgt_lang]}> "
                for text in batch
            ]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=5,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            hyps.extend(decoded)

        unfiltered_hyps = hyps.copy()
        filtered_hyps = []

        for text, hyp in zip(texts, hyps):
            if "-" not in text and not "–" in text:
                hyp = hyp.replace("-", " ")

            if (hyp and hyp.endswith('.') and text and not text.rstrip().endswith('.')):
                hyp = hyp.rstrip('.')

            filtered_hyps.append(hyp)

        return {
            "translations": filtered_hyps,
            "unfiltered_translations": unfiltered_hyps,
        }

    # ------------------------------
    # Unified Processing API
    # ------------------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            raise ValueError("Invalid input format")

        result = self._generate(
            input_data["texts"],
            input_data["src_lang"],
            input_data["tgt_lang"],
        )

        if not self.validate_output(result):
            raise ValueError("Invalid output format")

        return result


# ===================================================
# 🚀 FastAPI Integration
# ===================================================
app = FastAPI(title="Translation Worker with BaseWorker")
worker = TranslationWorker()

@app.post("/translate")
async def translate_endpoint(
    texts_json: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...)
):
    try:
        texts = json.loads(texts_json)
        input_data = {"texts": texts, "src_lang": src_lang, "tgt_lang": tgt_lang}

        result = worker.process(input_data)

        return JSONResponse(content=result)

    except Exception as e:
        print("❌ Translation Worker Error:")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8004"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
