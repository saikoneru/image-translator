from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
import httpx
from PIL import Image
import io, base64, json, time, os, tempfile, subprocess, re, asyncio
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from typing import Optional
from image_translator.utils import merge_translations_smart, draw_paragraphs_polys
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Unified Document Translator")
REQUEST_TIMEOUT = 300

WORKER_URLS = {
    "ocr": os.getenv("OCR_WORKER_URL", "http://i13hpc69:8001/ocr"),
    "segment": os.getenv("SEGMENT_WORKER_URL", "http://i13hpc69:8002/segment"),
    "inpaint": os.getenv("INPAINT_WORKER_URL", "http://i13hpc69:8003/inpaint"),
    "translate": os.getenv("TRANSLATE_WORKER_URL", "http://i13hpc69:8004/translate"),
    "layout": os.getenv("LAYOUT_WORKER_URL", "http://i13hpc69:8005/detect_layout"),
}

# ---------------------- IMAGE TRANSLATION ---------------------- #

async def translate_image_bytes(img_bytes: bytes, src_lang: str, tgt_lang: str, client: httpx.AsyncClient) -> bytes:
    original = Image.open(io.BytesIO(img_bytes))
    has_transparency = original.mode in ("RGBA", "LA") or (
        original.mode == "P" and "transparency" in original.info
    )

    # OCR
    ocr = (await client.post(
        WORKER_URLS["ocr"],
        files={"file": ("image.png", img_bytes)},
        timeout=REQUEST_TIMEOUT
    )).json()["results"]

    if not ocr or all(not (r.get("text") or "").strip() for r in ocr):
        return img_bytes

    # Layout
    layout = (await client.post(
        WORKER_URLS["layout"],
        files={"file": ("image.png", img_bytes),
               "ocr_results_json": (None, json.dumps(ocr))}
    )).json()

    # Segment and collect all text
    all_texts = []
    mappings = []

    for para_idx, ocr_para in enumerate(layout.get("paragraphs", [])):
        seg = (await client.post(
            WORKER_URLS["segment"],
            files={"file": ("image.png", img_bytes)},
            data={"ocr_results_json": json.dumps(ocr_para)}
        )).json()

        merged = seg["merged_results"]
        start = len(all_texts)
        all_texts.extend([x["merged_text"] for x in merged])
        end = len(all_texts)

        mappings.append({
            "ocr_para": ocr_para,
            "merged": merged,
            "start": start,
            "end": end
        })

    # Batch translation
    translated = []
    if all_texts:
        translated = (await client.post(
            WORKER_URLS["translate"],
            data={
                "texts_json": json.dumps(all_texts),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
        )).json()["translations"]

    # merge translations
    merged_paragraphs = []
    for m in mappings:
        block = translated[m["start"]:m["end"]]
        for entry, t in zip(m["merged"], block):
            entry["merged_text"] = t
        merged_paragraphs.append(
            merge_translations_smart(m["merged"], m["ocr_para"])
        )

    # Inpaint
    inpaint = (await client.post(
        WORKER_URLS["inpaint"],
        files={"file": ("image.png", img_bytes)},
        data={"boxes_json": json.dumps([r["box"] for r in ocr])}
    )).json()

    inpainted = Image.open(io.BytesIO(base64.b64decode(inpaint["inpainted_image_base64"])))
    if has_transparency:
        inpainted = inpainted.convert("RGBA")

    # Draw translations
    base = original.convert("RGBA" if has_transparency else "RGB")
    result = draw_paragraphs_polys(inpainted.copy(), merged_paragraphs, base)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------- TRANSLATION HELPERS ---------------------- #

async def call_translate_api(texts, src, tgt, client):
    if not texts:
        return []
    resp = await client.post(
        WORKER_URLS["translate"],
        data={"texts_json": json.dumps(texts), "src_lang": src, "tgt_lang": tgt}
    )
    data = resp.json()
    return data.get("translations") or data.get("results") or data.get("translated_texts") or []

def merge_translated_paragraph(paragraph, translated):
    runs = paragraph.runs
    if not runs:
        paragraph.text = "\n".join(translated)
        return

    original = paragraph.text.split("\n")
    if len(translated) != len(original):
        translated = [" ".join(translated)] * len(original)

    # simple replacement: preserve number of lines and runs
    parts = iter(" ".join(translated))
    for r in runs:
        text = r.text
        r.text = "".join(next(parts, "") for _ in text)

async def process_text_frame(text_frame, src, tgt, client):
    for p in text_frame.paragraphs:
        lines = p.text.split("\n")
        if any(lines):
            t = await call_translate_api(lines, src, tgt, client)
            merge_translated_paragraph(p, t)

async def process_table(table, src, tgt, client):
    for row in table.rows:
        for cell in row.cells:
            if hasattr(cell, "text_frame"):
                await process_text_frame(cell.text_frame, src, tgt, client)

async def translate_image_shape(shape, src, tgt, client):
    try:
        orig = shape.image.blob
        new = await translate_image_bytes(orig, src, tgt, client)
        stream = io.BytesIO(new)
        image_part, rId = shape.part.get_or_add_image_part(stream)
        shape._element.blipFill.blip.rEmbed = rId
    except:
        pass

async def process_shape(shape, src, tgt, client, translate_images=True):
    if getattr(shape, "has_text_frame", False):
        await process_text_frame(shape.text_frame, src, tgt, client)
    elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
        await process_table(shape.table, src, tgt, client)
    elif getattr(shape, "image", None) and translate_images:
        await translate_image_shape(shape, src, tgt, client)
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for shp in shape.shapes:
            await process_shape(shp, src, tgt, client, translate_images)

async def convert_pptx_to_pdf(pptx_bytes: bytes) -> bytes:
    """Convert PPTX→PDF via libreoffice."""
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
        tmp.write(pptx_bytes)
        path = tmp.name

    outdir = tempfile.mkdtemp()
    subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", outdir, path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
    )

    pdf_path = path.replace(".pptx", ".pdf").replace(
        os.path.dirname(path), outdir
    )

    with open(pdf_path, "rb") as f:
        content = f.read()

    os.unlink(path)
    os.unlink(pdf_path)
    os.rmdir(outdir)
    return content

async def translate_pptx_bytes(
    pptx_bytes: bytes,
    src: str,
    tgt: str,
    client: httpx.AsyncClient,
    translate_master=True,
    translate_images=True
) -> bytes:

    prs = Presentation(io.BytesIO(pptx_bytes))

    # Masters
    if translate_master:
        masters = {id(slide.slide_layout.slide_master): slide.slide_layout.slide_master
                   for slide in prs.slides}.values()
        for m in masters:
            for shape in m.shapes:
                await process_shape(shape, src, tgt, client, translate_images)
            for layout in m.slide_layouts:
                for shape in layout.shapes:
                    await process_shape(shape, src, tgt, client, translate_images)

    # Slides
    for slide in prs.slides:
        for shape in slide.shapes:
            await process_shape(shape, src, tgt, client, translate_images)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()

# ---------------------- ENDPOINTS ---------------------- #

@app.post("/translate/image")
async def translate_image(file: UploadFile, src_lang: str = Form(...), tgt_lang: str = Form(...)):
    img = await file.read()
    async with httpx.AsyncClient() as c:
        out = await translate_image_bytes(img, src_lang, tgt_lang, c)
    return Response(content=out, media_type="image/png")

@app.post("/translate/pptx")
async def translate_pptx(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    output_format: str = Form("pdf"),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    pptx = await file.read()
    async with httpx.AsyncClient() as client:
        translated = await translate_pptx_bytes(
            pptx, src_lang, tgt_lang, client, translate_master, translate_images
        )

    if output_format.lower() == "pdf":
        pdf = await convert_pptx_to_pdf(translated)
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={file.filename.replace('.pptx','_translated.pdf')}"}
        )

    return Response(
        content=translated,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

@app.post("/translate/auto")
async def translate_auto(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    output_format: str = Form("auto"),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    name = file.filename.lower()

    if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
        return await translate_image(file, src_lang, tgt_lang)

    if name.endswith(".pptx"):
        fmt = "pdf" if output_format == "auto" else output_format
        return await translate_pptx(file, src_lang, tgt_lang, fmt, translate_master, translate_images)

    raise HTTPException(400, "Unsupported file type")

@app.post("/translate/pptx/sse")
async def translate_pptx_sse(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    pptx_bytes = await file.read()

    async def event_stream():
        try:
            prs = Presentation(io.BytesIO(pptx_bytes))
            total = len(prs.slides)

            # Initial event
            yield {
                "event": "status",
                "data": json.dumps({"status": "started", "total_slides": total})
            }
            await asyncio.sleep(0)

            async with httpx.AsyncClient() as client:

                # --- Translate master/layout once ---
                if translate_master:
                    yield {"event": "status", "data": json.dumps({"status": "processing_masters"})}
                    await asyncio.sleep(0)

                    masters = {id(slide.slide_layout.slide_master): slide.slide_layout.slide_master
                               for slide in prs.slides}.values()

                    for master in masters:
                        for shape in master.shapes:
                            await process_shape(shape, src_lang, tgt_lang, client, translate_images)
                        for layout in master.slide_layouts:
                            for shape in layout.shapes:
                                await process_shape(shape, src_lang, tgt_lang, client, translate_images)

                # --- Slide-by-slide translation + streaming ---
                for idx, slide in enumerate(prs.slides):
                    slide_no = idx + 1

                    # Progress event
                    yield {
                        "event": "progress",
                        "data": json.dumps({"slide": slide_no, "total": total})
                    }
                    await asyncio.sleep(0)

                    # Translate slide content
                    for shape in slide.shapes:
                        await process_shape(shape, src_lang, tgt_lang, client, translate_images)

                    # Convert current PPTX state → PDF
                    buf = io.BytesIO()
                    prs.save(buf)
                    pdf_bytes = await convert_pptx_to_pdf(buf.getvalue())
                    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

                    # Slide event
                    yield {
                        "event": "slide",
                        "data": json.dumps({
                            "slide": slide_no,
                            "total": total,
                            "file_base64": pdf_b64,
                            "format": "pdf"
                        }),
                    }
                    await asyncio.sleep(0)

                # Completion event
                yield {
                    "event": "complete",
                    "data": json.dumps({"status": "completed", "total_slides": total})
                }
                await asyncio.sleep(0)

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    # SSE response with no buffering
    return EventSourceResponse(
        event_stream(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        },
        ping=10
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}
