from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
from typing import List, Dict, Any
from PIL import Image
import base64
import numpy as np
import json
import io
import traceback
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import ast
import re

# --- Model setup ---
#client = InferenceClient(base_url="http://i13hpc66:8054/v1/")
CACHE_DIR = "/export/data1/skoneru/hf_cache"

print("Loading Model")
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=CACHE_DIR,
    )

vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=CACHE_DIR,)

system_prompt = """
You are given OCR results extracted from an image.
Each OCR element has:
- an index (its order in the list)
- recognized text (string)

Your task:
- Identify which parts of the image should be translated together.
- Group the OCR elements that belong to the same reading sentence and should be translated as a phrase.
- KIT name at the top of the slide should be alone as group.
- Do not group just looking at the text alone, use visual information.
- Make sure always that all Names, Iitle of slides and website names or links in the picture/slide should not be merged and kept alone.
- Title of the slide, Section names and Keywords should not be grouped
- Each group should contain the indices of words that form a meaningful unit (e.g. "No" + "Parking" → [0,1]).
- Keep the order of text natural (left-to-right, top-to-bottom).
- Output only valid JSON as a list of groups (each group is a list of indices).
Example:

Input:
[
  {"id": 0, "text": "Notice"},
  {"id": 1, "text": "No"},
  {"id": 2, "text": "Parking"},
  {"id": 3, "text": "Monday"}
  {"id": 4, "text": "to"}
  {"id": 5, "text": "Friday"}
]

Output:
[[0], [1, 2], [3,4,5]]

If all items are already semantically separate, output something like:
[[0], [1], [2], [3], [4]]

Return only JSON, no explanation.
"""
app = FastAPI(title="Segment Worker")



# --- Core logic ---
def merge_boxes_from_groups(ocr_results, groups):
    merged_results = []
    print(ocr_results)
    for group in groups:
        print(group)
        group_boxes = [ocr_results[i]["box"] for i in group]
        group_texts = [ocr_results[i]["text"] for i in group]

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

        merged_results.append({
            "merged_box": poly.tolist(),  # Convert to list for JSON
            "merged_text": " ".join(group_texts),
            "group_indices": group
        })
    return merged_results


def segment_boxes(ocr_results, image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{image_b64}"
    image = Image.open(io.BytesIO(image_bytes))

    ocr_lines_dict = [{"id": str(i), "text": line["text"]} for i, line in enumerate(ocr_results)]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_data_uri,
                },
                {"type": "text", "text": f'{system_prompt}\nGroup the following OCR results:\n' + str(ocr_lines_dict)},
            ],
        }
    ]

    # Preparation for inference
    text = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)


    #image_inputs, video_inputs = process_vision_info(messages)
    inputs = vl_processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

# Inference: Generation of the output
    generated_ids = vl_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    content = vl_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(content)
    # Remove any non-JSON prefix/suffix if model adds markdown syntax
    content = content.replace("```json", "").replace("```", "")
    groups = json.loads(content)

    groups = [ [int(y) for y in x] for x in groups ]

    return groups


# --- FastAPI Endpoint ---
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
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=False)
