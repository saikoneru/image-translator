# src/image_translator/pipeline.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
from PIL import Image
import io, base64

app = FastAPI(title="Image Translator Gateway")

WORKER_URLS = {
    "ocr": "http://127.0.0.1:8001/ocr",
    "segment": "http://127.0.0.1:8002/segment",
    "inpaint": "http://127.0.0.1:8003/inpaint",
    "translate": "http://127.0.0.1:8004/translate",
}

@app.post("/process")
async def process_image(file: UploadFile, src_lang: str = Form(...), tgt_lang: str = Form(...)):
    """
    Example gateway endpoint that would orchestrate the OCR → Segment → Inpaint → Translate → Draw pipeline.
    For now, it just reads and returns the uploaded image.
    """
    img_bytes = await file.read()

    async with httpx.AsyncClient() as client:
        print("Dummy processing pipeline triggered...")
        # -- In a real pipeline, you'd call the worker services here --
        # ocr_resp = await client.post(WORKER_URLS["ocr"], files={"file": ("image.png", img_bytes)})
        # ocr_results = ocr_resp.json()["results"]

        # seg_resp = await client.post(WORKER_URLS["segment"], json={"ocr_results": ocr_results})
        # merged = seg_resp.json()["merged"]

        # inpaint_resp = await client.post(
        #     WORKER_URLS["inpaint"],
        #     files={"file": ("image.png", img_bytes)},
        #     data={"boxes": str([r["box"] for r in ocr_results])},
        # )
        # masked_image = inpaint_resp.content

        # trans_resp = await client.post(
        #     WORKER_URLS["translate"],
        #     json={"texts": [m["merged_text"] for m in merged], "src": src_lang, "tgt": tgt_lang},
        # )
        # translations = trans_resp.json()["translations"]

        # draw_resp = await client.post(
        #     WORKER_URLS["draw"],
        #     files={"file": ("image.png", masked_image)},
        #     data={"ocr_results": str(merged)},
        # )
        # final_img = draw_resp.content

        # Dummy behavior — just convert uploaded bytes into a PIL image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # You cannot return a PIL image directly — it’s not JSON serializable
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": img_b64}
    

if __name__ == "__main__":
    import uvicorn
    # Important: specify the module path, not the package
    # Here, `image_translator.pipeline:app` matches the file location inside src/
    uvicorn.run("image_translator.pipeline:app", host="127.0.0.1", port=8080, reload=True)
