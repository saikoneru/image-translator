from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
import httpx
from PIL import Image
import io, base64
import json
from image_translator.utils import merge_translations_smart, draw_paragraphs_polys
import time
import os
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from typing import Optional
import subprocess
import tempfile
import re
from typing import AsyncGenerator
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


async def translate_image_bytes(img_bytes: bytes, src_lang: str, tgt_lang: str, client: httpx.AsyncClient) -> bytes:
    """
    Core image translation logic - returns translated image bytes
    """
    # Detect if original image has transparency
    original_image = Image.open(io.BytesIO(img_bytes))
    has_transparency = original_image.mode in ('RGBA', 'LA') or (original_image.mode == 'P' and 'transparency' in original_image.info)
    print(f"Original image mode: {original_image.mode}, has_transparency: {has_transparency}")

    # Step 1: OCR
    start_time = time.time()
    ocr_resp = await client.post(WORKER_URLS["ocr"], files={"file": ("image.png", img_bytes)}, timeout=REQUEST_TIMEOUT)
    ocr_results = ocr_resp.json()["results"]
    print(f"OCR: {time.time() - start_time:.2f}s")

    # If OCR produced nothing, return original image
    if not ocr_results or all(not (r.get("text") and r.get("text").strip()) for r in ocr_results):
        print("No OCR text detected - returning original image")
        return img_bytes

    # Step 2: Layout Detection
    start_time = time.time()
    multipart_data = {
        "file": ("image.png", img_bytes, "image/png"),
        "ocr_results_json": (None, json.dumps(ocr_results))
    }
    layout_resp = await client.post(WORKER_URLS["layout"], files=multipart_data, timeout=REQUEST_TIMEOUT)
    layout_data = layout_resp.json()
    print(f"Layout: {time.time() - start_time:.2f}s - {len(layout_data.get('paragraphs', []))} paragraphs")

    # Step 3: Segment all paragraphs and collect texts for batch translation
    start_time = time.time()
    all_merged_results = []
    all_texts_to_translate = []
    paragraph_segment_mapping = []

    for para_idx, ocr_paragraph in enumerate(layout_data.get("paragraphs", [])):
        seg_resp = await client.post(
            WORKER_URLS["segment"],
            files={"file": ("image.png", img_bytes)},
            data={"ocr_results_json": json.dumps(ocr_paragraph)},
            timeout=REQUEST_TIMEOUT,
        )
        seg_data = seg_resp.json()
        merged_results = seg_data["merged_results"]
        all_merged_results.append(merged_results)

        start_idx = len(all_texts_to_translate)
        texts = [res["merged_text"] for res in merged_results]
        all_texts_to_translate.extend(texts)
        end_idx = len(all_texts_to_translate)

        paragraph_segment_mapping.append({
            "para_idx": para_idx,
            "ocr_paragraph": ocr_paragraph,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "merged_results": merged_results
        })

    print(f"Segment: {time.time() - start_time:.2f}s - {len(all_texts_to_translate)} texts collected")

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
        print(f"Translation: {time.time() - start_time:.2f}s - {len(translations)} translations")

    # Step 5: Distribute translations back to paragraphs
    start_time = time.time()
    ocr_para_trans_results = []
    for mapping in paragraph_segment_mapping:
        para_translations = translations[mapping["start_idx"]:mapping["end_idx"]]
        for entry, translation in zip(mapping["merged_results"], para_translations):
            entry["merged_text"] = translation
        ocr_trans_results = merge_translations_smart(
            mapping["merged_results"],
            mapping["ocr_paragraph"]
        )
        ocr_para_trans_results.append(ocr_trans_results)
    print(f"Merging: {time.time() - start_time:.2f}s")

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

    if has_transparency and inpainted_image.mode != 'RGBA':
        inpainted_image = inpainted_image.convert('RGBA')
    print(f"Inpainting: {time.time() - start_time:.2f}s")

    # Step 7: Draw translations on inpainted image
    start_time = time.time()
    if has_transparency:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    else:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    trans_image = draw_paragraphs_polys(inpainted_image.copy(), ocr_para_trans_results, image)
    print(f"Drawing: {time.time() - start_time:.2f}s")

    # Return result as bytes
    buf = io.BytesIO()
    if has_transparency:
        trans_image.save(buf, format="PNG")
    else:
        trans_image.save(buf, format="PNG")

    return buf.getvalue()


# ============ PPTX Translation Functions ============

async def call_translate_api(texts, src_lang, tgt_lang, client: httpx.AsyncClient):
    """Call translation API with batch of texts"""
    if not texts:
        return []
    payload = {"texts_json": json.dumps(texts), "src_lang": src_lang, "tgt_lang": tgt_lang}
    resp = await client.post(WORKER_URLS["translate"], data=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    translations = data.get("translations") or data.get("results") or data.get("translated_texts")
    if translations is None:
        if isinstance(data, dict) and len(data) == 1:
            val = list(data.values())[0]
            if isinstance(val, str):
                return [val]
        raise RuntimeError(f"Unexpected translate API response: {data}")
    return translations


def merge_translated_paragraph_preserve_runs(paragraph, translated_lines, force_single_line=False):
    """Merge translations back into paragraph runs while preserving formatting"""
    original_lines = paragraph.text.split("\n")
    num_lines = len(original_lines)

    # Adjust translated lines to match original line count
    if len(translated_lines) < num_lines:
        merged_text = " ".join(translated_lines)
        translated_lines = []
        start = 0
        for line in original_lines:
            line_len = len(line)
            translated_lines.append(merged_text[start:start+line_len])
            start += line_len
        if start < len(merged_text):
            translated_lines[-1] += merged_text[start:]
    elif len(translated_lines) > num_lines:
        merged_text = " ".join(translated_lines)
        avg_len = len(merged_text) // num_lines
        translated_lines = [merged_text[i*avg_len:(i+1)*avg_len] for i in range(num_lines)]
        if len(merged_text) % num_lines:
            translated_lines[-1] += merged_text[num_lines*avg_len:]

    # Distribute translated lines to runs
    run_texts = [r.text or "" for r in paragraph.runs]
    run_assignments = {i: [] for i in range(len(paragraph.runs))}
    run_idx = 0
    run_pos = 0
    for orig_line, trans_line in zip(original_lines, translated_lines):
        remaining = len(orig_line)
        segments = []
        while remaining > 0 and run_idx < len(run_texts):
            cur_text = run_texts[run_idx]
            avail = len(cur_text) - run_pos
            if avail <= 0:
                run_idx += 1
                run_pos = 0
                continue
            take = min(avail, remaining)
            segments.append((run_idx, cur_text[run_pos:run_pos+take]))
            remaining -= take
            run_pos += take
            if run_pos >= len(cur_text):
                run_idx += 1
                run_pos = 0
        # Assign translated line proportionally
        total_len = sum(len(s[1]) for s in segments) or 1
        pos = 0
        for i, (r_idx, s_text) in enumerate(segments):
            if i == len(segments) - 1:
                seg = trans_line[pos:]
            else:
                seg_len = int(round(len(s_text) / total_len * len(trans_line)))
                seg = trans_line[pos:pos+seg_len]
            run_assignments[r_idx].append(seg)
            pos += len(seg)

    for r_idx, parts in run_assignments.items():
        paragraph.runs[r_idx].text = "\n".join(parts) if parts else ""
        paragraph.runs[r_idx].text = re.sub(r'_x[0-9A-Fa-f]{4}_', ' ', paragraph.runs[r_idx].text)


async def process_text_frame(text_frame, src_lang, tgt_lang, client: httpx.AsyncClient):
    """Process text frame with translation"""
    for paragraph in text_frame.paragraphs:
        lines = paragraph.text.split("\n")
        if not any(lines):
            continue
        is_single_line = len(lines) == 1
        try:
            translations = await call_translate_api(lines, src_lang, tgt_lang, client)
        except Exception as e:
            print("Translation API failed:", e)
            continue
        try:
            merge_translated_paragraph_preserve_runs(paragraph, translations, force_single_line=is_single_line)
        except Exception as e:
            paragraph.text = " ".join(translations) if is_single_line else "\n".join(translations)
            paragraph.text = re.sub(r'_x[0-9A-Fa-f]{4}_', ' ', paragraph.text)


async def process_table(table, src_lang, tgt_lang, client: httpx.AsyncClient):
    """Process table cells with translation"""
    for r in range(len(table.rows)):
        for c in range(len(table.columns)):
            cell = table.cell(r, c)
            if getattr(cell, "text_frame", None):
                await process_text_frame(cell.text_frame, src_lang, tgt_lang, client)


def replace_image_blob(shape, new_image_bytes):
    """Replace image in shape with new bytes"""
    try:
        image_stream = io.BytesIO(new_image_bytes) if isinstance(new_image_bytes, bytes) else new_image_bytes
        image_part, rId = shape.part.get_or_add_image_part(image_stream)
        blip = shape._element.blipFill.blip
        blip.rEmbed = rId
    except Exception as e:
        print("replace_image_blob failed:", e)


async def translate_image_shape(shape, src_lang, tgt_lang, client: httpx.AsyncClient):
    """Translate image within PPTX shape"""
    try:
        if not getattr(shape, "image", None):
            return
        orig_bytes = shape.image.blob

        # Translate the image
        translated_bytes = await translate_image_bytes(orig_bytes, src_lang, tgt_lang, client)

        # Replace the image
        replace_image_blob(shape, translated_bytes)
    except Exception as e:
        print("translate_image_shape failed:", e)


async def process_shape(shape, src_lang, tgt_lang, client: httpx.AsyncClient, translate_images=True):
    """Process individual shape (text, table, image, or group)"""
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        print("Processing text frame...")
        await process_text_frame(shape.text_frame, src_lang, tgt_lang, client)
    elif getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.TABLE and getattr(shape, "table", None):
        print("Processing table...")
        await process_table(shape.table, src_lang, tgt_lang, client)
    elif getattr(shape, "image", None) is not None and translate_images:
        print("Processing image...")
        await translate_image_shape(shape, src_lang, tgt_lang, client)
    elif getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
        print("Processing group shape...")
        for shp in shape.shapes:
            await process_shape(shp, src_lang, tgt_lang, client, translate_images)


async def process_slide_master(slide_master, src_lang, tgt_lang, client: httpx.AsyncClient, translate_images=True):
    """Process slide master shapes"""
    for shape in list(slide_master.shapes):
        await process_shape(shape, src_lang, tgt_lang, client, translate_images)


async def process_slide_layout(slide_layout, src_lang, tgt_lang, client: httpx.AsyncClient, translate_images=True):
    """Process slide layout shapes"""
    for shape in list(slide_layout.shapes):
        await process_shape(shape, src_lang, tgt_lang, client, translate_images)

async def translate_single_slide_to_pdf(
    slide,
    src_lang: str,
    tgt_lang: str,
    client: httpx.AsyncClient,
    translate_images: bool = True
) -> bytes:
    """
    Translate a single slide and return it as a 1-page PPTX bytes
    """
    # Create a new presentation with just this slide
    temp_prs = Presentation()
    temp_prs.slide_width = slide.part.presentation.slide_width
    temp_prs.slide_height = slide.part.presentation.slide_height

    # Copy slide layout
    slide_layout = temp_prs.slide_layouts[0]  # Use blank layout
    new_slide = temp_prs.slides.add_slide(slide_layout)

    # Copy all shapes from original slide
    for shape in slide.shapes:
        # Create a copy of the shape in the new slide
        # This is simplified - you may need more sophisticated shape copying
        await process_shape(shape, src_lang, tgt_lang, client, translate_images)

    # Actually, better approach: translate in place then extract
    for shape in list(slide.shapes):
        await process_shape(shape, src_lang, tgt_lang, client, translate_images)

    # Return the slide's presentation as bytes
    out = io.BytesIO()
    slide.part.presentation.save(out)
    out.seek(0)
    return out.read()


async def convert_pptx_to_pdf(pptx_bytes: bytes) -> bytes:
    """Convert PPTX bytes to PDF bytes using LibreOffice"""
    with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp_pptx:
        tmp_pptx.write(pptx_bytes)
        tmp_pptx_path = tmp_pptx.name

    tmp_dir = tempfile.mkdtemp()

    try:
        result = subprocess.run(
            ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir',
             tmp_dir, tmp_pptx_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise Exception(f"LibreOffice conversion failed: {result.stderr}")

        pdf_filename = os.path.basename(tmp_pptx_path).replace('.pptx', '.pdf')
        pdf_path = os.path.join(tmp_dir, pdf_filename)

        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()

        pdf_b64 = base64.b64encode(pdf_content).decode('utf-8')
        return pdf_b64
    finally:
        os.unlink(tmp_pptx_path)
        if os.path.exists(os.path.join(tmp_dir, pdf_filename)):
            os.unlink(os.path.join(tmp_dir, pdf_filename))
        os.rmdir(tmp_dir)


async def translate_pptx_incremental(
    input_bytes: bytes,
    src_lang: str,
    tgt_lang: str,
    client: httpx.AsyncClient,
    translate_master: bool = True,
    translate_images: bool = True
) -> AsyncGenerator[bytes, None]:
    """
    Translate PPTX slide by slide, yielding incremental PDFs
    Yields: PDF with 1 slide, then PDF with 2 slides, then PDF with 3 slides, etc.
    """
    prs = Presentation(io.BytesIO(input_bytes))
    total_slides = len(prs.slides)

    # Translate masters and layouts first if requested
    if translate_master:
        used_masters = []
        seen_master_ids = set()
        for slide in prs.slides:
            if hasattr(slide, 'slide_layout') and hasattr(slide.slide_layout, 'slide_master'):
                master = slide.slide_layout.slide_master
                master_id = id(master)
                if master_id not in seen_master_ids:
                    seen_master_ids.add(master_id)
                    used_masters.append(master)

        for master in used_masters:
            await process_slide_master(master, src_lang, tgt_lang, client, translate_images)
            for layout in master.slide_layouts:
                await process_slide_layout(layout, src_lang, tgt_lang, client, translate_images)

    # Create a temporary presentation that we'll build incrementally
    translated_prs = Presentation(io.BytesIO(input_bytes))

    # Translate slides one by one and yield incremental PDFs
    for slide_idx, slide in enumerate(translated_prs.slides):
        print(f"Processing slide {slide_idx + 1}/{total_slides}")

        # Translate this slide
        for shape in list(slide.shapes):
            await process_shape(shape, src_lang, tgt_lang, client, translate_images)

        # Save current progress (all slides up to this point)
        pptx_output = io.BytesIO()
        translated_prs.save(pptx_output)
        pptx_output.seek(0)
        pptx_bytes = pptx_output.read()

        # Convert to PDF
        pdf_bytes = await convert_pptx_to_pdf(pptx_bytes)

        # Create a boundary marker and yield the PDF
        boundary = f"\n--SLIDE-{slide_idx + 1}-of-{total_slides}--\n".encode()
        yield boundary + pdf_bytes


@app.post("/translate/pptx/stream")
async def translate_pptx_streaming_endpoint(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    """
    Translate PPTX and stream incremental PDFs
    Each chunk contains a complete PDF with all slides translated so far
    Format: --SLIDE-N-of-TOTAL--\\n<PDF_BYTES>
    """
    pptx_bytes = await file.read()

    async def generate():
        async with httpx.AsyncClient() as client:
            async for pdf_chunk in translate_pptx_incremental(
                pptx_bytes,
                src_lang,
                tgt_lang,
                client,
                translate_master,
                translate_images
            ):
                yield pdf_chunk

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=translated_{file.filename.replace('.pptx', '.pdf')}",
            "X-Content-Type-Options": "nosniff"
        }
    )


@app.post("/translate/pptx/stream-json")
async def translate_pptx_streaming_json(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    """
    Translate PPTX and stream incremental PDFs as JSON
    Each message: {"slide": N, "total": M, "pdf_base64": "..."}
    """
    pptx_bytes = await file.read()

    async def generate():
        async with httpx.AsyncClient() as client:
            prs = Presentation(io.BytesIO(pptx_bytes))
            total_slides = len(prs.slides)

            # Translate masters
            if translate_master:
                used_masters = []
                seen_master_ids = set()
                for slide in prs.slides:
                    if hasattr(slide, 'slide_layout') and hasattr(slide.slide_layout, 'slide_master'):
                        master = slide.slide_layout.slide_master
                        master_id = id(master)
                        if master_id not in seen_master_ids:
                            seen_master_ids.add(master_id)
                            used_masters.append(master)

                for master in used_masters:
                    await process_slide_master(master, src_lang, tgt_lang, client, translate_images)
                    for layout in master.slide_layouts:
                        await process_slide_layout(layout, src_lang, tgt_lang, client, translate_images)

            # Translate slides incrementally
            for slide_idx, slide in enumerate(prs.slides):
                print(f"Processing slide {slide_idx + 1}/{total_slides}")

                for shape in list(slide.shapes):
                    await process_shape(shape, src_lang, tgt_lang, client, translate_images)

                # Save and convert to PDF
                pptx_output = io.BytesIO()
                prs.save(pptx_output)
                pptx_output.seek(0)

                pdf_bytes = await convert_pptx_to_pdf(pptx_output.read())
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

                # Yield JSON message
                message = json.dumps({
                    "slide": slide_idx + 1,
                    "total": total_slides,
                    "pdf_base64": pdf_base64
                }) + "\n"

                yield message.encode()

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",  # Newline-delimited JSON
        headers={
            "Content-Disposition": f"attachment; filename=stream.ndjson",
            "X-Content-Type-Options": "nosniff"
        }
    )

@app.post("/translate/pptx/sse")
async def translate_pptx_sse(
    file: UploadFile,
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    translate_master: bool = Form(True),
    translate_images: bool = Form(True)
):
    """
    Translate PPTX and stream incremental PDFs via Server-Sent Events
    Each event contains: slide number, total slides, and base64 PDF
    """
    pptx_bytes = await file.read()

    async def event_generator():
        async with httpx.AsyncClient() as client:
            try:
                # Load presentation to get total slides
                prs = Presentation(io.BytesIO(pptx_bytes))
                total_slides = len(prs.slides)

                print("Total Slides: {}".format(total_slides))

                # Send initial status
                yield {
                    "event": "status",
                    "data": json.dumps({
                        "status": "started",
                        "total_slides": total_slides
                    })
                }

                # Translate masters if requested
                if translate_master:
                    yield {
                        "event": "status",
                        "data": json.dumps({
                            "status": "processing_masters"
                        })
                    }

                    used_masters = []
                    seen_master_ids = set()
                    for slide in prs.slides:
                        if hasattr(slide, 'slide_layout') and hasattr(slide.slide_layout, 'slide_master'):
                            master = slide.slide_layout.slide_master
                            master_id = id(master)
                            if master_id not in seen_master_ids:
                                seen_master_ids.add(master_id)
                                used_masters.append(master)

                    for master in used_masters:
                        await process_slide_master(master, src_lang, tgt_lang, client, translate_images)
                        for layout in master.slide_layouts:
                            await process_slide_layout(layout, src_lang, tgt_lang, client, translate_images)

                # Process slides incrementally
                for slide_idx, slide in enumerate(prs.slides):
                    current_slide = slide_idx + 1

                    # Send progress update
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "slide": current_slide,
                            "total": total_slides,
                            "status": "translating"
                        })
                    }

                    # Translate this slide
                    for shape in list(slide.shapes):
                        await process_shape(shape, src_lang, tgt_lang, client, translate_images)

                    # Save current state and convert to PDF
                    pptx_output = io.BytesIO()
                    prs.save(pptx_output)
                    pptx_output.seek(0)

                    pdf_base64 = await convert_pptx_to_pdf(pptx_output.read())

                    # Send slide update with PDF
                    yield {
                        "event": "slide",
                        "data": json.dumps({
                            "slide": current_slide,
                            "total": total_slides,
                            "pdf_base64": pdf_base64
                        })
                    }

                # Send completion event
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "status": "completed",
                        "total_slides": total_slides
                    })
                }

                print("Current Slide {}".format(current_slide))

            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": str(e)
                    })
                }

    return EventSourceResponse(event_generator())

@app.post("/convert/pptx_to_pdf")
async def convert_pptx_to_pdf_endpoint(file: UploadFile):
    """
    Simple endpoint to convert PPTX to PDF (for initial upload)
    """
    try:
        pptx_bytes = await file.read()
        pdf_bytes = await convert_pptx_to_pdf(pptx_bytes)
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        return {
            "status": "ok",
            "pdf_base64": pdf_base64
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }, 500

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "incremental-pdf-translator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_translator.streaming_pipeline:app", host="0.0.0.0", port=8080, reload=False)
