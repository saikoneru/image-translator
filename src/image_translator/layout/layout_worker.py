"""
Hi-SAM Layout Detection Worker with FastAPI
Receives images, detects text layout, sends full image to OCR, assigns words to layout structure
"""

import json
import numpy as np
import torch
import cv2
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import base64
from typing import Optional, List
import uvicorn
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
from tqdm import tqdm
import random
from shapely.geometry import Polygon, Point

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from utils import utilities

# Initialize FastAPI app
app = FastAPI(title="Hi-SAM Layout Worker", description="Text layout detection with OCR word assignment")

# Global model variables
hisam_model = None
amg = None


class LayoutConfig(BaseModel):
    """Configuration for layout detection"""
    model_type: str = "vit_l"
    checkpoint: str = "pretrained_checkpoint/hi_sam_l.pth"
    device: str = "cuda"
    total_points: int = 1500
    batch_points: int = 100
    layout_thresh: float = 0.5


# Global config
config = LayoutConfig()


def unclip(p, unclip_ratio=2.0):
    """Expand polygon for better text coverage"""
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def sort_and_reindex_hierarchy(hierarchy):
    """
    Sort paragraphs and lines in reading order (top-to-bottom, left-to-right)
    and reindex them. Also filters out empty lines/words.
    """

    sorted_paragraphs = []

    for p_idx, para in enumerate(sorted(hierarchy, key=lambda p: np.mean([w['box'][0][1] for w in p['lines'][0]['words']]) if p['lines'] else float('inf'))):
        # Sort lines in paragraph
        sorted_lines = []
        for l_idx, line in enumerate(sorted(para['lines'], key=lambda l: np.mean([pt[1] for w in l['words'] for pt in w['box']]) if l['words'] else float('inf'))):
            # Filter out empty or whitespace-only words
            filtered_words = [
                w for w in line['words']
                if w.get('text') and w['text'].strip()
            ]

            if not filtered_words:
                continue  # Skip empty line

            # Sort words left-to-right
            filtered_words = sorted(filtered_words, key=lambda w: np.mean([pt[0] for pt in w['box']]))
            sorted_lines.append({
                "line_index": len(sorted_lines),
                "words": filtered_words
            })

        if not sorted_lines:
            continue  # Skip paragraph if it has no valid lines

        sorted_paragraphs.append({
            "paragraph_index": len(sorted_paragraphs),
            "lines": sorted_lines
        })

    return sorted_paragraphs

def assign_words_to_masks(ocr_results, layout_result):
    """
    Assign OCR words to layout paragraphs/lines based on geometry.
    Unassigned words are grouped into new single-line paragraphs.

    Args:
        ocr_results: list of dicts with keys {'text', 'box': [[x,y], ...]}
        layout_result: dict with {'paragraphs': [{'lines': [{'words': [{'vertices': [[x,y],...]}, ...]}]}]}

    Returns:
        paragraphs: list of dicts
            [
                {
                    "paragraph_index": int,
                    "lines": [
                        {"line_index": int, "words": [ocr_word_dicts]}
                    ]
                }
            ]
    """
    paragraphs_out = []
    assigned_ids = set()

    layout_paragraphs = layout_result.get("paragraphs", [])
    if not ocr_results:
        return []

    # Filter out empty OCR results
    ocr_results = [w for w in ocr_results if str(w.get("text", "")).strip()]
    if not ocr_results:
        return []

    # --- Pass 1: assign OCR words to layout lines ---
    for p_idx, para in enumerate(layout_paragraphs):
        para_out = {"paragraph_index": p_idx, "lines": []}

        for l_idx, line in enumerate(para.get("lines", [])):
            line_out = {"line_index": l_idx, "words": []}

            # Collect all vertices for this line polygon
            line_pts = []
            for word in line.get("words", []):
                verts = np.array(word.get("vertices", []), dtype=np.float32)
                if len(verts) >= 3:
                    line_pts.extend(verts.tolist())

            if len(line_pts) < 3:
                continue

            line_poly = Polygon(line_pts)

            for i, word in enumerate(ocr_results):
                if i in assigned_ids:
                    continue
                box = np.array(word["box"], dtype=np.float32)
                cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
                pt = Point(cx, cy)

                # Check if centroid is inside or very close to polygon
                if line_poly.contains(pt) or line_poly.distance(pt) < 5:
                    line_out["words"].append(word)
                    assigned_ids.add(i)

            # Only keep line if it has words
            if line_out["words"]:
                # Sort words left → right
                line_out["words"].sort(key=lambda w: np.mean([pt[0] for pt in w["box"]]))
                para_out["lines"].append(line_out)

        # Only keep paragraph if it has lines
        if para_out["lines"]:
            paragraphs_out.append(para_out)

    # --- Pass 2: assign leftover OCR words to their own paragraphs ---
    unassigned = [ocr_results[i] for i in range(len(ocr_results)) if i not in assigned_ids]

    for u_idx, word in enumerate(unassigned):
        new_para = {
            "paragraph_index": len(paragraphs_out),
            "lines": [
                {"line_index": 0, "words": [word]}
            ],
        }
        paragraphs_out.append(new_para)

    return paragraphs_out

def run_full_image_ocr(image_bytes, ocr_url):
    """
    Sends the full image to OCR worker and returns OCR results.
    """
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(ocr_url, files=files, timeout=60)
    return response.json()


def initialize_model(cfg: LayoutConfig):
    """Initialize Hi-SAM model"""
    global hisam_model, amg

    # Create args-like object
    class Args:
        def __init__(self, cfg):
            self.model_type = cfg.model_type
            self.checkpoint = cfg.checkpoint
            self.device = cfg.device
            self.total_points = cfg.total_points
            self.batch_points = cfg.batch_points
            self.hier_det = True
            self.attn_layers = 1
            self.prompt_len = 12
            self.input_size = [1024, 1024]

    args = Args(cfg)

    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    hisam_model = model_registry[args.model_type](args)
    hisam_model.eval()
    hisam_model.to(args.device)

    efficient_hisam = args.model_type in ['vit_s', 'vit_t']
    amg = AutoMaskGenerator(hisam_model, efficient_hisam=efficient_hisam)

    print(f"✅ Hi-SAM model loaded: {args.model_type}")


def process_image_layout(image: np.ndarray, cfg: LayoutConfig):
    """
    Process image to detect text layout and extract structure
    Returns layout information with paragraphs and lines
    """
    global amg

    img_h, img_w = image.shape[:2]

    # Set image in AutoMaskGenerator
    amg.set_image(image)

    # Predict masks
    masks, scores, affinity = amg.predict(
        from_low_res=False,
        fg_points_num=cfg.total_points,
        batch_points_num=cfg.batch_points,
        score_thresh=0.5,
        nms_thresh=0.5,
    )

    if masks is None:
        # Return empty structure
        lines = [{'words': [{'text': '', 'vertices': [[0,0],[1,0],[1,1],[0,1]]}], 'text': ''}]
        paragraphs = [{'lines': lines}]
        return {"paragraphs": paragraphs}

    masks = (masks[:, 0, :, :]).astype(np.uint8)  # word masks, (n, h, w)
    lines = []
    line_indices = []

    # Extract word polygons from masks
    for index, mask in enumerate(masks):
        line = {'words': [], 'text': ''}
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            epsilon = 0.002 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4:
                continue

            pts = unclip(points)
            if len(pts) != 1:
                continue

            pts = pts[0].astype(np.int32)
            if Polygon(pts).area < 32:
                continue

            pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
            cnt_list = pts.tolist()
            line['words'].append({'text': '', 'vertices': cnt_list})

        if line['words']:
            lines.append(line)
            line_indices.append(index)

    # Group lines into paragraphs using affinity
    line_grouping = utilities.DisjointSet(len(line_indices))
    affinity = affinity[line_indices][:, line_indices]

    for i1, i2 in zip(*np.where(affinity > cfg.layout_thresh)):
        line_grouping.union(i1, i2)

    line_groups = line_grouping.to_group()
    paragraphs = []

    for line_group in line_groups:
        paragraph = {'lines': []}
        for id_ in line_group:
            paragraph['lines'].append(lines[id_])
        if paragraph:
            paragraphs.append(paragraph)

    return {"paragraphs": paragraphs}

def convert_hierarchy_to_paragraphs(hierarchy_sorted):
    """
    Converts sorted hierarchy into list of paragraphs,
    where each paragraph is a list of dicts for lines:
        {"text": line_text, "box": line_polygon}
    """
    paragraphs_list = []

    for para in hierarchy_sorted:
        para_lines = []
        for line in para["lines"]:
            # Merge word texts for the line
            line_text = " ".join([word.get("text", "") for word in line["words"]])

            # Merge all word vertices into a bounding polygon for the line
            # One simple approach: take min/max of all word vertices
            all_pts = np.vstack([np.array(word["box"]) for word in line["words"]])
            x_min, y_min = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
            box = [[int(x_min), int(y_min)], [int(x_max), int(y_min)],
                   [int(x_max), int(y_max)], [int(x_min), int(y_max)]]

            para_lines.append({"text": line_text, "box": box})

        paragraphs_list.append(para_lines)

    return paragraphs_list


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model(config)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "Hi-SAM Layout Worker is running",
        "model": config.model_type,
        "checkpoint": config.checkpoint,
    }


@app.post("/detect_layout")
async def process_image(
    file: UploadFile = File(...),
    ocr_results_json: str = Form(...)
    ):
    """
    Process image: detect layout, run OCR, assign words to layout structure

    Args:
        file: Image file

    Returns:
        JSON with sorted hierarchical structure (paragraphs -> lines -> words with OCR text)
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Convert BGR to RGB for Hi-SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Detect layout using Hi-SAM
        layout_result = process_image_layout(image_rgb, config)

        ocr_results = json.loads(ocr_results_json)

        # 3. Assign OCR words to layout structure
        hierarchy = assign_words_to_masks(ocr_results, layout_result)

        # 4. Sort hierarchy in reading order
        hierarchy_sorted = sort_and_reindex_hierarchy(hierarchy)
        paragraphs = convert_hierarchy_to_paragraphs(hierarchy_sorted)

        return JSONResponse(content={
            "success": True,
            "paragraphs": paragraphs
        })

    except HTTPException:
        raise
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/detect_layout_only")
async def detect_layout_only(file: UploadFile = File(...)):
    """
    Only detect text layout without OCR

    Args:
        file: Image file

    Returns:
        JSON with layout structure (paragraphs and lines with vertices only)
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    layout_result = process_image_layout(image, config)

    return JSONResponse(content={
        "success": True,
        "num_paragraphs": len(layout_result["paragraphs"]),
        "layout": layout_result
    })



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hi-SAM Layout Detection Worker")
    parser.add_argument("--model-type", type=str, default="vit_l", help="Model type")
    parser.add_argument("--checkpoint", type=str, default="pretrained_checkpoint/hi_sam_l.pth",
                       help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8005, help="Port")

    args = parser.parse_args()

    # Update config
    config.model_type = args.model_type
    config.checkpoint = args.checkpoint
    config.device = args.device

    # Run server
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", reload=False)
