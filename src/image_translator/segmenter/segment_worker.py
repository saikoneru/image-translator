from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import json
import io
import traceback
from wtpsplit import SaT
import os
import uvicorn

# --- Model setup ---
print("Loading SaT Segmenter...")
# Use ONNX with CUDA for maximum speed (~50% faster than PyTorch)
try:
    sat = SaT("sat-3l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("✅ SaT Segmenter loaded with ONNX CUDA acceleration")
except Exception as e:
    print(f"⚠️ ONNX CUDA failed ({e}), falling back to PyTorch")
    import torch
    sat = SaT("sat-3l-sm")
    if torch.cuda.is_available():
        sat.half().to("cuda")
        print("✅ SaT Segmenter loaded with PyTorch CUDA")
    else:
        print("✅ SaT Segmenter loaded with PyTorch CPU")

app = FastAPI(title="SaT Segment Worker")


def merge_boxes_from_groups(ocr_results, groups):
    """
    Merge OCR results based on grouped indices
    """
    merged_results = []
    for group in groups:
        if not group:
            continue

        group_boxes = [ocr_results[i]["box"] for i in group]
        group_texts = [ocr_results[i]["text"] for i in group]

        # Convert boxes to numpy array for processing
        all_points = []
        for box in group_boxes:
            all_points.extend(box)

        all_points = np.array(all_points)
        x_min = float(np.min(all_points[:, 0]))
        y_min = float(np.min(all_points[:, 1]))
        x_max = float(np.max(all_points[:, 0]))
        y_max = float(np.max(all_points[:, 1]))

        # Create merged polygon
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


def detect_titles_and_keywords(ocr_results):
    """
    Detect titles and keywords to isolate or normalize

    Returns:
        isolated_indices: set of indices that should be kept isolated (Title Case only)
        normalize_map: dict mapping index -> normalized text (for ALL CAPS normalization)
    """
    isolated_indices = set()
    normalize_map = {}

    for i, item in enumerate(ocr_results):
        text = item.get("text", "").strip()

        if not text:
            continue

        # Check if text has letters
        has_letters = any(c.isalpha() for c in text)
        if not has_letters:
            continue

        # Check if text is ALL UPPERCASE (all letters are uppercase)
        if text.isupper() and len(text) > 1:
            # Normalize all caps to Title case (first letter capital, rest lowercase)
            normalize_map[i] = text.capitalize()

        # Check if first letter of each word is capitalized (Title Case)
        # But NOT all caps
        elif not text.isupper():
            words = text.split()
            is_title_case = all(
                word[0].isupper() if word and word[0].isalpha() else True
                for word in words
            )

            # If Title Case, isolate it
            if is_title_case and len(words) >= 1:
                isolated_indices.add(i)

    return isolated_indices, normalize_map


def segment_with_sat(ocr_results, lang_code="en"):
    """
    Use SaT to segment OCR results into meaningful groups for translation

    Rules:
    - ALL CAPS text: normalize to capitalize (first letter capital) for SaT processing
    - Title Case (First Letter Capital): isolate as separate group
    - Let SaT infer sentence boundaries

    Args:
        ocr_results: List of dicts with 'text' and 'box' keys
        lang_code: Language code for segmentation (e.g., 'en', 'de', 'es')

    Returns:
        groups: List of lists, where each sublist contains indices to merge
    """
    if not ocr_results:
        return []

    # Filter empty texts
    valid_indices = [i for i, item in enumerate(ocr_results) if item.get("text", "").strip()]
    if not valid_indices:
        return []

    # Detect items that should be isolated and get normalization map
    isolated_indices, normalize_map = detect_titles_and_keywords(ocr_results)

    # Build full text with character position tracking
    text_parts = []
    char_to_index = []  # Maps (start_char, end_char, ocr_index)

    current_char = 0
    for idx in valid_indices:
        text = ocr_results[idx]["text"].strip()

        # Apply normalization if needed (ALL CAPS -> capitalize)
        if idx in normalize_map:
            text = normalize_map[idx]

        if not text:
            continue

        # Add space between words
        if text_parts:
            char_to_index.append((current_char, current_char + 1, None))  # space marker
            current_char += 1

        # Track this word's position
        start_char = current_char
        end_char = current_char + len(text)
        text_parts.append(text)
        char_to_index.append((start_char, end_char, idx))
        current_char = end_char

    if not text_parts:
        return [[i] for i in valid_indices]

    # Create full text for SaT
    full_text = " ".join(text_parts)

    print(f"DEBUG: Full text for SaT:\n{repr(full_text)}")

    # Use SaT to find sentence boundaries
    try:
        sentences = list(sat.split(full_text))
        print(f"DEBUG: SaT sentences: {sentences}")
    except Exception as e:
        print(f"SaT split error: {e}, falling back to simple split")
        import re
        sentences = re.split(r'[.!?。！？]+\s+', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]

    # Map sentences back to OCR indices
    groups = []
    processed_indices = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find where this sentence is in the full text
        sentence_start = full_text.find(sentence)
        if sentence_start == -1:
            continue
        sentence_end = sentence_start + len(sentence)

        # Find which OCR indices overlap with this sentence
        group = []
        for start_char, end_char, idx in char_to_index:
            if idx is None:  # Skip space markers
                continue
            # Check overlap
            if start_char < sentence_end and end_char > sentence_start:
                if idx not in processed_indices:
                    group.append(idx)
                    processed_indices.add(idx)

        if not group:
            continue

        # Separate isolated items (Title Case) from regular text
        isolated_in_group = [idx for idx in group if idx in isolated_indices]
        regular_in_group = [idx for idx in group if idx not in isolated_indices]

        # Add isolated items as individual groups
        for iso_idx in sorted(isolated_in_group):
            groups.append([iso_idx])

        # Add regular items as a merged group
        if regular_in_group:
            groups.append(sorted(regular_in_group))

    # Handle any unprocessed OCR items
    ungrouped = [idx for idx in valid_indices if idx not in processed_indices]
    for idx in ungrouped:
        groups.append([idx])

    # Sort groups by first index to maintain reading order
    groups.sort(key=lambda g: min(g) if g else float('inf'))

    return groups


def segment_boxes(ocr_results, image_bytes=None, lang_code="en"):
    """
    Segment OCR results using SaT
    image_bytes is kept for API compatibility but not used
    """
    groups = segment_with_sat(ocr_results, lang_code=lang_code)
    return groups


@app.post("/segment")
async def segment_endpoint(
    file: UploadFile = File(...),
    ocr_results_json: str = Form(...),
    lang_code: str = Form(default="en")
):
    """
    Segment OCR results into translation-ready groups

    Args:
        file: Image file (for API compatibility, not used)
        ocr_results_json: JSON string of OCR results
        lang_code: Language code (en, de, es, ja, etc.)
    """
    try:
        image_bytes = await file.read()
        ocr_results = json.loads(ocr_results_json)

        groups = segment_boxes(ocr_results, image_bytes, lang_code=lang_code)
        merged_results = merge_boxes_from_groups(ocr_results, groups)

        return JSONResponse(content={
            "groups": groups,
            "merged_results": merged_results,
            "num_groups": len(groups)
        })

    except Exception as e:
        print("❌ Segment Worker Error:")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "SaT Segment Worker running",
        "model": "sat-3l-sm",
        "backend": "ONNX CUDA" if hasattr(sat, 'ort_providers') else "PyTorch"
    }


if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8002"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
