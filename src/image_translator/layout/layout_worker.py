"""
Hi-SAM Layout Detection Worker with FastAPI + BaseWorker
Receives images, detects text layout, assigns OCR words to layout structure.
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
from typing import Optional, List, Dict, Any
import requests
import base64
import uvicorn
from PIL import Image
import io
from shapely.geometry import Polygon, Point
import pyclipper
import random
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from utils import utilities
from abc import ABC, abstractmethod
import traceback


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
# 🧠 Hi-SAM Layout Worker Implementation
# ======================================================
class LayoutConfig(BaseModel):
    model_type: str = "vit_l"
    checkpoint: str = "/app/pretrained_checkpoint/hi_sam_l.pth"
    device: str = "cuda"
    total_points: int = 1500
    batch_points: int = 100
    layout_thresh: float = 0.5

class HiSamLayoutWorker(BaseWorker):
    def __init__(self, config: LayoutConfig):
        self.config = config
        self.hisam_model = None
        self.amg = None
        self.initialize_model(config)

    # --------------------------
    # Validation
    # --------------------------
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return (
            "image" in input_data
            and isinstance(input_data["image"], np.ndarray)
            and "ocr_results" in input_data
            and isinstance(input_data["ocr_results"], list)
        )

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        return "paragraphs" in output_data and isinstance(output_data["paragraphs"], list)

    # --------------------------
    # Model Initialization
    # --------------------------
    def initialize_model(self, cfg: LayoutConfig):
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
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.hisam_model = model_registry[args.model_type](args)
        self.hisam_model.eval()
        self.hisam_model.to(args.device)

        efficient_hisam = args.model_type in ["vit_s", "vit_t"]
        self.amg = AutoMaskGenerator(self.hisam_model, efficient_hisam=efficient_hisam)

        print(f"✅ Hi-SAM model loaded: {args.model_type}")

    # --------------------------
    # Helper Functions
    # --------------------------
    def unclip(self, p, unclip_ratio=2.0):
        poly = Polygon(p)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    # --------------------------
    # Layout with Resize Support
    # --------------------------
    def process_image_layout(self, image: np.ndarray) -> Dict[str, Any]:
        cfg = self.config
        amg = self.amg

        # --- store original size ---
        orig_h, orig_w = image.shape[:2]

        # --- target Hi-SAM size ---
        target_w, target_h = 1024, 1024
        img_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # scale factors
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        amg.set_image(img_resized)

        masks, scores, affinity = amg.predict(
            from_low_res=False,
            fg_points_num=cfg.total_points,
            batch_points_num=cfg.batch_points,
            score_thresh=0.5,
            nms_thresh=0.5,
        )

        if masks is None:
            return {"paragraphs": [{"lines": [{"words": []}]}]}

        masks = (masks[:, 0, :, :]).astype(np.uint8)
        lines, line_indices = [], []

        for index, mask in enumerate(masks):
            line = {"words": []}

            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cont in contours:
                epsilon = 0.002 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
                points = approx.reshape((-1, 2))

                if points.shape[0] < 4:
                    continue

                pts = self.unclip(points)
                if len(pts) != 1:
                    continue

                pts = pts[0].astype(np.float32)

                # --- SCALE BACK TO ORIGINAL COORDINATES ---
                pts[:, 0] = pts[:, 0] * scale_x
                pts[:, 1] = pts[:, 1] * scale_y

                # clip
                pts[:, 0] = np.clip(pts[:, 0], 0, orig_w)
                pts[:, 1] = np.clip(pts[:, 1], 0, orig_h)

                if Polygon(pts).area < 32:
                    continue

                line["words"].append({"vertices": pts.astype(np.int32).tolist()})

            if line["words"]:
                lines.append(line)
                line_indices.append(index)

        # nothing else changes — affinity remains based on groups
        line_grouping = utilities.DisjointSet(len(line_indices))
        affinity = affinity[line_indices][:, line_indices]
        for i1, i2 in zip(*np.where(affinity > cfg.layout_thresh)):
            line_grouping.union(i1, i2)

        line_groups = line_grouping.to_group()
        paragraphs = [{"lines": [lines[i] for i in group]} for group in line_groups]

        return {"paragraphs": paragraphs}

    # --------------------------
    # Assignment & Final Formatting
    # --------------------------
    def assign_words_to_masks(self, ocr_results, layout_result):
        paragraphs_out = []
        assigned_ids = set()

        layout_paragraphs = layout_result.get("paragraphs", [])
        if not ocr_results:
            return []

        ocr_results = [w for w in ocr_results if str(w.get("text", "")).strip()]

        for p_idx, para in enumerate(layout_paragraphs):
            para_out = {"paragraph_index": p_idx, "lines": []}

            for l_idx, line in enumerate(para.get("lines", [])):
                line_out = {"line_index": l_idx, "words": []}

                # build polygon union for checking inside/near
                line_pts = []
                for word in line.get("words", []):
                    verts = np.array(word.get("vertices", []), dtype=np.float32)
                    if len(verts) >= 3:
                        line_pts.extend(verts.tolist())

                if len(line_pts) < 3:
                    continue

                line_poly = Polygon(line_pts)

                # assign OCR words
                for i, word in enumerate(ocr_results):
                    if i in assigned_ids:
                        continue
                    box = np.array(word["box"], dtype=np.float32)
                    cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
                    if line_poly.contains(Point(cx, cy)) or line_poly.distance(Point(cx, cy)) < 5:
                        line_out["words"].append(word)
                        assigned_ids.add(i)

                if line_out["words"]:
                    line_out["words"].sort(key=lambda w: np.mean([p[0] for p in w["box"]]))
                    para_out["lines"].append(line_out)

            if para_out["lines"]:
                paragraphs_out.append(para_out)

        # leftover OCR words become new paragraphs
        unassigned = [ocr_results[i] for i in range(len(ocr_results)) if i not in assigned_ids]
        for word in unassigned:
            paragraphs_out.append({
                "paragraph_index": len(paragraphs_out),
                "lines": [{"line_index": 0, "words": [word]}]
            })

        return paragraphs_out

    def sort_and_reindex_hierarchy(self, hierarchy):
        sorted_paragraphs = []

        for para in sorted(hierarchy,
                           key=lambda p: np.mean([w["box"][0][1] for w in p["lines"][0]["words"]])
                           if p["lines"] else float("inf")):

            sorted_lines = []
            for line in sorted(para["lines"],
                               key=lambda l: np.mean([pt[1] for w in l["words"] for pt in w["box"]])
                               if l["words"] else float("inf")):

                filtered_words = [w for w in line["words"] if w.get("text") and w["text"].strip()]
                if not filtered_words:
                    continue
                filtered_words.sort(key=lambda w: np.mean([p[0] for p in w["box"]]))
                sorted_lines.append({"line_index": len(sorted_lines), "words": filtered_words})

            if sorted_lines:
                sorted_paragraphs.append({"paragraph_index": len(sorted_paragraphs), "lines": sorted_lines})

        return sorted_paragraphs

    def convert_hierarchy_to_paragraphs(self, hierarchy_sorted):
        paragraphs_list = []

        for para in hierarchy_sorted:
            para_lines = []
            for line in para["lines"]:
                line_text = " ".join([w.get("text", "") for w in line["words"]])
                pts = np.vstack([np.array(w["box"]) for w in line["words"]])
                x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
                x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])

                box = [[int(x_min), int(y_min)],
                       [int(x_max), int(y_min)],
                       [int(x_max), int(y_max)],
                       [int(x_min), int(y_max)]]

                para_lines.append({"text": line_text, "box": box})

            paragraphs_list.append(para_lines)

        return paragraphs_list

    # --------------------------
    # Unified Process Interface
    # --------------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            raise ValueError("Invalid input format for Hi-SAM worker")

        image_rgb = cv2.cvtColor(input_data["image"], cv2.COLOR_BGR2RGB)
        layout_result = self.process_image_layout(image_rgb)

        hierarchy = self.assign_words_to_masks(input_data["ocr_results"], layout_result)
        hierarchy_sorted = self.sort_and_reindex_hierarchy(hierarchy)
        paragraphs = self.convert_hierarchy_to_paragraphs(hierarchy_sorted)

        output = {"paragraphs": paragraphs}
        if not self.validate_output(output):
            raise ValueError("Invalid output format")

        return output

# ======================================================
# 🚀 FastAPI Integration
# ======================================================
app = FastAPI(title="Hi-SAM Layout Worker with BaseWorker")
worker = HiSamLayoutWorker(LayoutConfig())


@app.post("/detect_layout")
async def detect_layout(file: UploadFile = File(...), ocr_results_json: str = Form(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_rgb = np.array(pil_image)
        image = image_rgb[:, :, ::-1]
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        ocr_results = json.loads(ocr_results_json)
        input_data = {"image": image, "ocr_results": ocr_results}
        result = worker.process(input_data)
        return JSONResponse(content={"success": True, **result})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "running", "model": worker.config.model_type}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="debug")