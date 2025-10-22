from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw
import numpy as np
import io, json, base64, traceback
import os
import uvicorn

app = FastAPI(title="Inpaint Worker")

# Initialize the model once
simple_lama = SimpleLama()


def inpaint_with_boxes(image: Image.Image, boxes):
    """Inpaint an image using given bounding boxes."""
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        if isinstance(box, (list, tuple, np.ndarray)):
            poly_points = [(float(x), float(y)) for x, y in box]
            draw.polygon(poly_points, fill=255)
        else:
            x_min, y_min, x_max, y_max = map(float, box)
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

    # Run SimpleLama inpainting
    result = simple_lama(image, mask)
    return result


@app.post("/inpaint")
async def inpaint_endpoint(file: UploadFile = File(...), boxes_json: str = Form(...)):
    try:
        # Read image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Parse boxes JSON
        boxes = json.loads(boxes_json)

        # Run inpainting
        result = inpaint_with_boxes(image, boxes)

        # Convert result to base64
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse(content={"inpainted_image_base64": img_b64})

    except Exception as e:
        print("❌ Inpaint Worker Error:")
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8003"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
