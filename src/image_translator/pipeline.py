# src/image_translator/pipeline.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
from PIL import Image
import io, base64
import json
from image_translator.utils import merge_translations, draw_paragraphs_polys

app = FastAPI(title="Image Translator Gateway")

WORKER_URLS = {
    "ocr": "http://i13hpc69:8001/ocr",
    "segment": "http://i13hpc69:8002/segment",
    "inpaint": "http://i13hpc69:8003/inpaint",
    "translate": "http://i13hpc69:8004/translate",
    "layout": "http://i13hpc69:8005/detect_layout",
}

@app.post("/process")
async def process_image(file: UploadFile, src_lang: str = Form(...), tgt_lang: str = Form(...)):
    """
    Example gateway endpoint that would orchestrate the OCR → Segment → Inpaint → Translate → Draw pipeline.
    For now, it just reads and returns the uploaded image.
    """
    img_bytes = await file.read()

    async with httpx.AsyncClient() as client:
        ocr_resp = await client.post(WORKER_URLS["ocr"], files={"file": ("image.png", img_bytes)})
        ocr_results = ocr_resp.json()["results"]

        multipart_data = {
            "file": ("image.png", img_bytes, "image/png"),  # file
            "ocr_results_json": (None, json.dumps(ocr_results))  # form field
        }

        layout_resp = await client.post(WORKER_URLS["layout"], files=multipart_data)
        layout_data = layout_resp.json()

        print(layout_data)

        ocr_para_trans_results = []

        for ocr_paragraph in layout_data.get("paragraphs", []):
            seg_resp = await client.post(
                WORKER_URLS["segment"],
                files={"file": ("image.png", img_bytes)},
                data={"ocr_results_json": json.dumps(ocr_paragraph)}
            )
            seg_data = seg_resp.json()
            merged_results = seg_data["merged_results"]

            trans_resp = await client.post(
                WORKER_URLS["translate"],
                data={
                    "texts_json": json.dumps([res["merged_text"] for res in merged_results]),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                },
            )
            translations = trans_resp.json()["translations"]

            for entry in merged_results:
                entry["merged_text"] = translations.pop(0) if translations else ""

            ocr_trans_results = merge_translations(merged_results, ocr_paragraph)
            ocr_para_trans_results.append(ocr_trans_results)

        inpaint_resp = await client.post(
            WORKER_URLS["inpaint"],
            files={"file": ("image.png", img_bytes)},
            data={"boxes_json": json.dumps([r["box"] for r in ocr_results])}
        )

        inpaint_data = inpaint_resp.json()
        inpainted_image_b64 = inpaint_data["inpainted_image_base64"]
        inpainted_image = Image.open(io.BytesIO(base64.b64decode(inpainted_image_b64)))

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        trans_image = draw_paragraphs_polys(inpainted_image.copy(), ocr_para_trans_results, image)

    # You cannot return a PIL image directly — it’s not JSON serializable
    buf = io.BytesIO()
    trans_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": img_b64}


if __name__ == "__main__":
    import uvicorn
    # Important: specify the module path, not the package
    # Here, `image_translator.pipeline:app` matches the file location inside src/
    uvicorn.run("image_translator.pipeline:app", host="0.0.0.0", port=8080, reload=False)
