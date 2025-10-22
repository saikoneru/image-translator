"""
Inpaint Worker with BaseWorker Interface
Uses SimpleLama for object removal via bounding boxes or polygons.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw
import numpy as np
import io, json, base64, traceback, os, uvicorn
from abc import ABC, abstractmethod
from typing import Dict, Any, List


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
# 🧠 Inpaint Worker Implementation
# ======================================================
class InpaintWorker(BaseWorker):
    def __init__(self):
        self.model = SimpleLama()
        print("✅ SimpleLama model initialized")

    # --------------------------
    # Validation
    # --------------------------
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return (
            "image" in input_data
            and isinstance(input_data["image"], Image.Image)
            and "boxes" in input_data
            and isinstance(input_data["boxes"], list)
        )

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        return "inpainted_image_base64" in output_data

    # --------------------------
    # Core Processing Logic
    # --------------------------
    def inpaint_with_boxes(self, image: Image.Image, boxes: List[Any]) -> Image.Image:
        """Inpaint an image using bounding boxes or polygons."""
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        for box in boxes:
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) > 0:
                # Polygon mask
                poly_points = [(float(x), float(y)) for x, y in box]
                draw.polygon(poly_points, fill=255)
            else:
                # Bounding box mask (x_min, y_min, x_max, y_max)
                try:
                    x_min, y_min, x_max, y_max = map(float, box)
                    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
                except Exception:
                    continue

        return self.model(image, mask)

    # --------------------------
    # Unified Process Interface
    # --------------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            raise ValueError("Invalid input for InpaintWorker")

        image = input_data["image"]
        boxes = input_data["boxes"]

        result_img = self.inpaint_with_boxes(image, boxes)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        output = {"inpainted_image_base64": img_b64}

        if not self.validate_output(output):
            raise ValueError("Invalid output from InpaintWorker")

        return output


# ======================================================
# 🚀 FastAPI Integration
# ======================================================
app = FastAPI(title="Inpaint Worker with BaseWorker")
worker = InpaintWorker()


@app.post("/inpaint")
async def inpaint_endpoint(file: UploadFile = File(...), boxes_json: str = Form(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        boxes = json.loads(boxes_json)

        input_data = {"image": image, "boxes": boxes}
        result = worker.process(input_data)

        return JSONResponse(content=result)

    except Exception as e:
        print("❌ Inpaint Worker Error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "Inpaint Worker running", "model": "SimpleLama"}


if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8003"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, log_level=log_level, reload=False)
