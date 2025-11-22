"""
SaT Segment Worker with BaseWorker Interface
Groups OCR text boxes into translation-ready segments.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import json, io, traceback, os, uvicorn
from abc import ABC, abstractmethod
from wtpsplit import SaT


# ======================================================
# 🧩 Base Worker Interface
# ======================================================
class BaseWorker(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        pass


# ======================================================
# 🧠 SaT Segment Worker Implementation
# ======================================================
class SegmentWorker(BaseWorker):
    def __init__(self):
        print("Loading SaT Segmenter...")
        try:
            self.sat = SaT("sat-12l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.backend = "ONNX CUDA"
            print("✅ SaT Segmenter loaded with ONNX CUDA acceleration")
        except Exception as e:
            print(f"⚠️ ONNX CUDA failed ({e}), falling back to PyTorch")
            import torch
            self.sat = SaT("sat-3l-sm")
            if torch.cuda.is_available():
                self.sat.half().to("cuda")
                self.backend = "PyTorch CUDA"
            else:
                self.backend = "PyTorch CPU"
            print(f"✅ SaT Segmenter loaded with {self.backend}")

    # --------------------------
    # Validation
    # --------------------------
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "ocr_results" in input_data and isinstance(input_data["ocr_results"], list)

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        return "groups" in output_data and "merged_results" in output_data

    # --------------------------
    # Core Logic
    # --------------------------
    def merge_boxes_from_groups(self, ocr_results, groups):
        merged_results = []
        for group in groups:
            if not group:
                continue
            group_boxes = [ocr_results[i]["box"] for i in group]
            group_texts = [ocr_results[i]["text"] for i in group]

            all_points = np.array([pt for box in group_boxes for pt in box])
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            poly = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.float32)

            merged_results.append({
                "merged_box": poly.tolist(),
                "merged_text": " ".join(group_texts),
                "group_indices": group
            })
        return merged_results

    def detect_titles_and_keywords(self, ocr_results):
        isolated_indices = set()
        normalize_map = {}
        for i, item in enumerate(ocr_results):
            text = item.get("text", "").strip()
            if not text:
                continue
            if text.isupper() and len(text) > 1:
                normalize_map[i] = text.capitalize()
            elif all(w and w[0].isupper() for w in text.split() if w[0].isalpha()):
                isolated_indices.add(i)
        return isolated_indices, normalize_map

    def segment_with_sat(self, ocr_results, lang_code="en"):
        if not ocr_results:
            return []

        valid_indices = [i for i, o in enumerate(ocr_results) if o.get("text", "").strip()]
        if not valid_indices:
            return []

        isolated_indices, normalize_map = self.detect_titles_and_keywords(ocr_results)

        text_parts = []
        char_to_index = []
        current_char = 0

        for idx in valid_indices:
            text = ocr_results[idx]["text"].strip()
            if idx in normalize_map:
                text = normalize_map[idx]
            if text_parts:
                char_to_index.append((current_char, current_char + 1, None))
                current_char += 1
            start_char, end_char = current_char, current_char + len(text)
            text_parts.append(text)
            char_to_index.append((start_char, end_char, idx))
            current_char = end_char

        full_text = " ".join(text_parts)
        try:
            sentences = list(self.sat.split(full_text))
        except Exception:
            import re
            sentences = re.split(r'[.!?。！？]+\s+', full_text)

        groups = []
        processed = set()

        for sentence in sentences:
            start = full_text.find(sentence)
            if start == -1:
                continue
            end = start + len(sentence)
            group = [idx for s, e, idx in char_to_index if idx and s < end and e > start and idx not in processed]
            if not group:
                continue

            isolated = [i for i in group if i in isolated_indices]
            regular = [i for i in group if i not in isolated_indices]
            for iso in isolated:
                groups.append([iso])
            if regular:
                groups.append(sorted(regular))
            processed.update(group)

        # Add any unprocessed
        for idx in valid_indices:
            if idx not in processed:
                groups.append([idx])

        groups.sort(key=lambda g: min(g))
        return groups

    # --------------------------
    # Unified process()
    # --------------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            raise ValueError("Invalid input for SegmentWorker")

        ocr_results = input_data["ocr_results"]
        lang_code = input_data.get("lang_code", "en")

        groups = self.segment_with_sat(ocr_results, lang_code)
        merged_results = self.merge_boxes_from_groups(ocr_results, groups)

        print(merged_results)
        output = {"groups": groups, "merged_results": merged_results, "num_groups": len(groups)}
        if not self.validate_output(output):
            raise ValueError("Invalid output from SegmentWorker")

        return output


# ======================================================
# 🚀 FastAPI Wrapper
# ======================================================
app = FastAPI(title="SaT Segment Worker (BaseWorker)")
worker = SegmentWorker()


@app.post("/segment")
async def segment_endpoint(file: UploadFile = File(...), ocr_results_json: str = Form(...), lang_code: str = Form(default="en")):
    try:
        image_bytes = await file.read()
        ocr_results = json.loads(ocr_results_json)
        input_data = {"ocr_results": ocr_results, "lang_code": lang_code, "image_bytes": image_bytes}
        result = worker.process(input_data)
        return JSONResponse(content=result)
    except Exception as e:
        print("❌ Segment Worker Error:")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.get("/")
async def root():
    return {"status": "SaT Segment Worker running", "model": "sat-12l-sm", "backend": worker.backend}


if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8002"))
    uvicorn.run(app, host=host, port=port, log_level="debug", reload=False)
