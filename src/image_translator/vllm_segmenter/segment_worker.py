"""
Qwen-VL Segment Worker
Groups OCR text boxes into translation-ready segments using vision-language model.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import json
import io
import traceback
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import ast

app = FastAPI(title="Segment Worker")

# ======================================================
# 🧠 Model Setup
# ======================================================
print("Loading Qwen2-VL-7B-Instruct...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/app/models/"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir="/app/models/")
print(f"✅ Model loaded on device: {model.device}")

system_prompt = """
You are given OCR results extracted from an image.
Each OCR element has:
- an index (its order in the list)
- recognized text (string)
Your task:
- Group the OCR elements that belong to the same semantic phrase according to the image.
- Each group should contain the indices of words that form a meaningful unit.
- Do not group based on text content alone. Group texts that are spatially and contextually related in the image.
Constraints:
- Keep the order natural (left-to-right, top-to-bottom).
- Output only valid JSON as a list of groups (each group is a list of indices).
- Do not hallucinate and start generating all numbers, the total elements should be the same number of elements in OCR results
Example:
{'groups': [[1], [2,3,4]]}
"""

# ======================================================
# 🔧 Core Logic
# ======================================================
def merge_boxes_from_groups(ocr_results, groups):
    merged_results = []
    print(len(ocr_results), len(groups))
    for group in groups:
        print(group)
        group_boxes = [ocr_results[i-1]["box"] for i in group]
        group_texts = [ocr_results[i-1]["text"] for i in group]
        group_boxes = np.array(group_boxes)
        x_min = float(np.min(group_boxes[:, 0]))
        y_min = float(np.min(group_boxes[:, 1]))
        x_max = float(np.max(group_boxes[:, 2]))
        y_max = float(np.max(group_boxes[:, 3]))
        box = [x_min, y_min, x_max, y_max]
        poly = np.array([
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]]
        ], dtype=np.float32)
        group = [x - 1 for x in group]
        merged_results.append({
            "merged_box": poly.tolist(),  # Convert to list for JSON
            "merged_text": " ".join(group_texts),
            "group_indices": group
        })
    return merged_results


def validate_and_fix_groups(ocr_results, groups):
    """
    Validate that all OCR indices appear in the groups.
    If any are missing, add them as individual groups.
    Returns validated and fixed groups.
    """
    # Get all indices that should be present (1-indexed to match the format)
    expected_indices = set(range(1, len(ocr_results) + 1))

    # Collect all indices present in groups
    present_indices = set()
    for group in groups:
        present_indices.update(group)

    # Find missing indices
    missing_indices = expected_indices - present_indices

    if missing_indices:
        print(f"⚠️ Warning: Missing indices {sorted(missing_indices)} in groups. Adding them individually.")
        # Add missing indices as individual groups
        for idx in sorted(missing_indices):
            groups.append([idx])

    # Also check for invalid indices (out of range)
    invalid_indices = present_indices - expected_indices
    if invalid_indices:
        print(f"⚠️ Warning: Invalid indices {sorted(invalid_indices)} found. Removing them.")
        # Filter out invalid indices
        groups = [[idx for idx in group if idx in expected_indices] for group in groups]
        # Remove empty groups
        groups = [group for group in groups if group]

    # Sort groups by their first element to maintain order
    groups.sort(key=lambda g: min(g) if g else float('inf'))

    return groups


def segment_boxes(ocr_results, image_bytes):
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare messages for Qwen-VL
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": 'Group the following OCR results and return json with groups key only:\n' + str(ocr_results)
                }
            ]
        }
    ]

    # Process with Qwen-VL
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0,
            do_sample=False
        )

    # Trim input tokens from generation
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Parse model output
    content = output_text.strip()
    print("Segmenter Output")
    print(content)
    # Remove any non-JSON prefix/suffix if model adds markdown syntax
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        groups = ast.literal_eval(content)['groups']
    except:
        print(f"⚠️ Failed to parse JSON: {content}")
        print(f"Error: {e}")
        # Fallback: create individual groups for all OCR results
        groups = [[i+1] for i in range(len(ocr_results))]
        return groups

    # Validate and fix groups
    groups = validate_and_fix_groups(ocr_results, groups)

    return groups


# ======================================================
# 🚀 FastAPI Endpoint
# ======================================================
@app.post("/segment")
async def segment_endpoint(
    file: UploadFile = File(...),
    ocr_results_json: str = Form(...)
):
    try:
        image_bytes = await file.read()
        ocr_results = json.loads(ocr_results_json)
        groups = segment_boxes(ocr_results, image_bytes)
        merged_results = merge_boxes_from_groups(ocr_results, groups)
        return JSONResponse(content={
            "groups": groups,
            "merged_results": merged_results
        })
    except Exception as e:
        print("❌ Segment Worker Error:")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8007"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
