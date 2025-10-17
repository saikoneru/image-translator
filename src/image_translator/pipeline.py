# src/image_translator/pipeline.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
from PIL import Image
import io, base64
import json
from image_translator.utils import merge_translations, draw_paragraphs_polys
import time

app = FastAPI(title="Image Translator Gateway")
REQUEST_TIMEOUT = 300

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
    Optimized pipeline: batch all translations into a single request
    OCR → Layout → Segment (per paragraph) → Translate (all at once) → Inpaint → Draw
    """
    img_bytes = await file.read()

    async with httpx.AsyncClient() as client:
        # Step 1: OCR
        start_time = time.time()
        ocr_resp = await client.post(WORKER_URLS["ocr"], files={"file": ("image.png", img_bytes)}, timeout=REQUEST_TIMEOUT)
        ocr_results = ocr_resp.json()["results"]

        print(f"OCR: {time.time() - start_time}")

        # Step 2: Layout Detection
        start_time = time.time()
        multipart_data = {
            "file": ("image.png", img_bytes, "image/png"),
            "ocr_results_json": (None, json.dumps(ocr_results))
        }
        layout_resp = await client.post(WORKER_URLS["layout"], files=multipart_data, timeout=REQUEST_TIMEOUT)
        layout_data = layout_resp.json()
        print("Layout detected:", len(layout_data.get("paragraphs", [])), "paragraphs")
        print(f"Layout: {time.time() - start_time}")

        # Step 3: Segment all paragraphs and collect texts for batch translation
        start_time = time.time()
        all_merged_results = []  # Store merged results for each paragraph
        all_texts_to_translate = []  # Collect all texts that need translation
        paragraph_segment_mapping = []  # Track which texts belong to which paragraph

        for para_idx, ocr_paragraph in enumerate(layout_data.get("paragraphs", [])):
            seg_resp = await client.post(
                WORKER_URLS["segment"],
                files={"file": ("image.png", img_bytes)},
                data={"ocr_results_json": json.dumps(ocr_paragraph)},
                timeout=REQUEST_TIMEOUT,
            )
            seg_data = seg_resp.json()
            merged_results = seg_data["merged_results"]

            # Store merged results
            all_merged_results.append(merged_results)

            # Collect texts for translation
            start_idx = len(all_texts_to_translate)
            texts = [res["merged_text"] for res in merged_results]
            all_texts_to_translate.extend(texts)
            end_idx = len(all_texts_to_translate)

            # Track which indices in the translation batch belong to this paragraph
            paragraph_segment_mapping.append({
                "para_idx": para_idx,
                "ocr_paragraph": ocr_paragraph,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "merged_results": merged_results
            })

        print(f"Collected {len(all_texts_to_translate)} texts for batch translation")
        print(f"Segment: {time.time() - start_time}")

        # Step 4: Translate ALL texts in a single batch request
        translations = []
        start_time = time.time()
        if all_texts_to_translate:
            trans_resp = await client.post(
                WORKER_URLS["translate"],
                data={
                    "texts_json": json.dumps(all_texts_to_translate),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                },
                timeout=REQUEST_TIMEOUT,
            )
            translations = trans_resp.json()["translations"]
            print(f"Received {len(translations)} translations")

        print(f"Translation: {time.time() - start_time}")
        # Step 5: Distribute translations back to paragraphs
        ocr_para_trans_results = []
        start_time = time.time()
        for mapping in paragraph_segment_mapping:
            # Extract translations for this paragraph
            para_translations = translations[mapping["start_idx"]:mapping["end_idx"]]

            # Assign translations to merged results
            for entry, translation in zip(mapping["merged_results"], para_translations):
                entry["merged_text"] = translation

            # Merge translations back to OCR results
            ocr_trans_results = merge_translations(
                mapping["merged_results"],
                mapping["ocr_paragraph"]
            )
            ocr_para_trans_results.append(ocr_trans_results)

        print(f"Merging: {time.time() - start_time}")
        # Step 6: Inpaint
        start_time = time.time()
        inpaint_resp = await client.post(
            WORKER_URLS["inpaint"],
            files={"file": ("image.png", img_bytes)},
            data={"boxes_json": json.dumps([r["box"] for r in ocr_results])},
            timeout=REQUEST_TIMEOUT,
        )
        inpaint_data = inpaint_resp.json()
        inpainted_image_b64 = inpaint_data["inpainted_image_base64"]
        inpainted_image = Image.open(io.BytesIO(base64.b64decode(inpainted_image_b64)))
        print(f"Inpainting: {time.time() - start_time}")

        # Step 7: Draw translations on inpainted image
        start_time = time.time()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        trans_image = draw_paragraphs_polys(inpainted_image.copy(), ocr_para_trans_results, image)
        print(f"Drawing: {time.time() - start_time}")

    # Return result as base64
    buf = io.BytesIO()
    trans_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": img_b64}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_translator.pipeline:app", host="0.0.0.0", port=8080, reload=False)
