from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
from typing import List, Dict, Any
import base64
import numpy as np
import json
import io
import traceback

app = FastAPI(title="Segment Worker")

# --- Model setup ---
client = InferenceClient(base_url="http://i13hpc66:8054/v1/")

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
Example:
[[0, 1], [2,3,4]]
"""


# --- Core logic ---
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

        merged_results.append({
            "merged_box": poly.tolist(),  # Convert to list for JSON
            "merged_text": " ".join(group_texts),
            "group_indices": group
        })
    return merged_results


def segment_boxes(ocr_results, image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{image_b64}"

    chat = client.chat_completion(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": 'Group the following OCR results:\n' + str(ocr_results)},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]},
        ],
        seed=42,
        max_tokens=512,
        temperature=0,
    )

    # Parse model output
    content = chat.choices[0].message.content.strip()
    # Remove any non-JSON prefix/suffix if model adds markdown syntax
    content = content.replace("```json", "").replace("```", "")
    groups = json.loads(content)
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
    uvicorn.run("segment_worker:app", host="127.0.0.1", port=8002, reload=False)
