from fastapi import FastAPI, Form, File, UploadFile
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
import base64
from omni_fusion_model import OmniFusionModel
from PIL import Image
import io

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
        self.MODEL_NAME = "skoneru/OmniFusion_v2"

        self.system = OmniFusionModel(checkpoint_path=self.MODEL_NAME, cache_dir=self.CACHE_DIR)
        # Load model and tokenizer


    # ------------------------------
    # Input Validation
    # ------------------------------
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return (
            isinstance(input_data.get("texts"), list)
            and isinstance(input_data.get("src_lang"), str)
            and isinstance(input_data.get("tgt_lang"), str)
            and isinstance(input_data.get("images"), list)
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

    def _generate(self, texts: List[str], src_lang: str, tgt_lang: str, images: list) -> Dict[str, List[str]]:
        batch_size = 1
        text_batches = self.create_batches(texts, batch_size=1)
        image_batches = self.create_batches(images, batch_size=1
                                            )
        hyps = []
        for batch, image_batch in zip(text_batches, image_batches):
            batch = [x.lstrip().rstrip() for x in batch]
            translations = self.system.translate(
                    audio_paths=[],
                    image_paths=image_batch,
                    source_texts=batch,
                    target_lang=tgt_lang,
                    num_beams=1)
            print("Source Texts:", batch)
            print("Translations:", translations)
            hyps.extend(translations)


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
            input_data["images"]
        )

        if not self.validate_output(result):
            raise ValueError("Invalid output format")

        return result


# ===================================================
# 🚀 FastAPI Integration
# ===================================================
app = FastAPI(title="Translation Worker with BaseWorker")
worker = TranslationWorker()

@app.post("/multimodal_translate")
async def translate_endpoint(
    texts_json: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        texts = json.loads(texts_json)
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images = [img] * len(texts)
        input_data = {"texts": texts, "src_lang": src_lang, "tgt_lang": tgt_lang, "images": images}

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
