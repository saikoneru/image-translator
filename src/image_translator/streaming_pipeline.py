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
    "vllm_segment": os.getenv("VLLM_SEGMENT_WORKER_URL", "http://i13hpc69:8007/segment"),
    "inpaint": os.getenv("INPAINT_WORKER_URL", "http://i13hpc69:8003/inpaint"),
    "translate": os.getenv("TRANSLATE_WORKER_URL", "http://i13hpc69:8004/translate"),
    "multimodal_translate": os.getenv("MULTIMODAL_TRANSLATE_WORKER_URL", "http://i13hpc69:8006/multimodal_translate"),
    "layout": os.getenv("LAYOUT_WORKER_URL", "http://i13hpc69:8005/detect_layout"),
}

# ---------------------- IMAGE TRANSLATION ---------------------- #

from datetime import datetime
from pathlib import Path
from PIL import ImageDraw, ImageFont
import numpy as np

# Create visualization directory
VIZ_DIR = Path("/app/uploads/")
VIZ_DIR.mkdir(exist_ok=True)

def save_viz_stage(image: Image.Image, stage_name: str, timestamp: str) -> str:
    """Save a visualization stage and return the path"""
    filename = f"{timestamp}_{stage_name}.png"
    filepath = VIZ_DIR / filename
    image.save(filepath)
    print(f"Saved visualization: {filepath}")
    return str(filepath)

def draw_ocr_boxes(image: Image.Image, ocr_results: list, color=(0, 255, 0)) -> Image.Image:
    """Draw OCR bounding boxes on image (boxes only, no text)"""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, result in enumerate(ocr_results):
        box = result.get("box", [])

        if isinstance(box, list):
            box = np.array(box)

        points = box.reshape((-1, 2)).astype(np.int32)

        # Draw polygon only
        draw.polygon([tuple(p) for p in points], outline=color, width=3)

    return Image.alpha_composite(img, overlay).convert("RGB")

def draw_ocr_with_text(image: Image.Image, ocr_results: list, color=(0, 255, 0)) -> Image.Image:
    """Draw OCR boxes with text inside on inpainted background"""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for idx, result in enumerate(ocr_results):
        box = result.get("box", [])
        text = result.get("text", "")

        if isinstance(box, list):
            box = np.array(box)

        points = box.reshape((-1, 2)).astype(np.int32)

        # Draw polygon
        draw.polygon([tuple(p) for p in points], outline=color, width=2)

        # Calculate box center for text placement
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        # Draw text inside the box
        text_display = text if len(text) <= 30 else text[:27] + "..."
        bbox = draw.textbbox((center_x, center_y), text_display, font=font_small, anchor="mm")

        # Draw semi-transparent background for readability
        draw.rectangle(bbox, fill=(255, 255, 255, 200))
        draw.text((center_x, center_y), text_display, fill=(0, 0, 0), font=font_small, anchor="mm")

    return Image.alpha_composite(img, overlay).convert("RGB")

def draw_layout_paragraphs(image: Image.Image, layout: dict, ocr_results: list) -> Image.Image:
    """Draw layout paragraph groupings with different colors"""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Different colors for different paragraphs
    colors = [
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 192, 203) # Pink
    ]

    paragraphs = layout.get("paragraphs", [])
    for para_idx, paragraph in enumerate(paragraphs):
        color = colors[para_idx % len(colors)]

        for line in paragraph:
            box = line.get("box", [])
            if isinstance(box, list):
                box = np.array(box)

            points = box.reshape((-1, 2)).astype(np.int32)

            # Draw polygon with paragraph-specific color
            draw.polygon([tuple(p) for p in points], outline=color, width=4)

        # Draw paragraph label
        if paragraph:
            first_box = np.array(paragraph[0]["box"]).reshape((-1, 2))
            label_pos = (int(first_box[0][0]), int(first_box[0][1]) - 35)
            label = f"Para {para_idx}"
            bbox = draw.textbbox(label_pos, label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text(label_pos, label, fill=(255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

def draw_segmented_groups(image: Image.Image, merged_results: list) -> Image.Image:
    """Draw segmented text groups"""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 0)]

    for group_idx, group in enumerate(merged_results):
        color = colors[group_idx % len(colors)]
        merged_text = group.get("merged_text", "")

        # Get all boxes in this group
        for line in group.get("lines", []):
            box = line.get("box", [])
            if isinstance(box, list):
                box = np.array(box)

            points = box.reshape((-1, 2)).astype(np.int32)
            draw.polygon([tuple(p) for p in points], outline=color, width=3)

        # Draw group label
        if group.get("lines"):
            first_box = np.array(group["lines"][0]["box"]).reshape((-1, 2))
            label_pos = (int(first_box[0][0]), int(first_box[0][1]) - 25)
            label = f"G{group_idx}: {merged_text[:30]}..."
            bbox = draw.textbbox(label_pos, label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text(label_pos, label, fill=(0, 0, 0), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

def draw_inpaint_mask(image: Image.Image, boxes: list) -> Image.Image:
    """Draw inpainting mask regions"""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        if isinstance(box, list):
            box = np.array(box)

        points = box.reshape((-1, 2)).astype(np.int32)

        # Fill with semi-transparent red
        draw.polygon([tuple(p) for p in points], fill=(255, 0, 0, 100), outline=(255, 0, 0), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")

async def translate_image_bytes(
    img_bytes: bytes,
    src_lang: str,
    tgt_lang: str,
    client: httpx.AsyncClient,
    enable_viz: bool = False
) -> bytes:
    """
    Translate image with detailed debugging and optional visualization
    """
    import json
    import io
    import base64
    from PIL import Image
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_paths = {}

    original = Image.open(io.BytesIO(img_bytes))
    has_transparency = original.mode in ("RGBA", "LA") or (
        original.mode == "P" and "transparency" in original.info
    )

    # Save original
    if enable_viz:
        viz_paths["original"] = save_viz_stage(original, "01_original", timestamp)

    # OCR
    ocr = (await client.post(
        WORKER_URLS["ocr"],
        files={"file": ("image.png", img_bytes)},
        timeout=REQUEST_TIMEOUT
    )).json()["results"]

    print(f"\n{'='*60}")
    print(f"OCR Results: {len(ocr)} words detected")
    print(f"{'='*60}")

    if enable_viz and ocr:
        ocr_viz = draw_ocr_boxes(original, ocr)
        viz_paths["ocr"] = save_viz_stage(ocr_viz, "02_ocr_boxes", timestamp)

    if not ocr or all(not (r.get("text") or "").strip() for r in ocr):
        return img_bytes

    # Layout
    layout = (await client.post(
        WORKER_URLS["layout"],
        files={"file": ("image.png", img_bytes),
               "ocr_results_json": (None, json.dumps(ocr))},
        timeout=REQUEST_TIMEOUT
    )).json()

    print(f"\n{'='*60}")
    print(f"Layout Results: {len(layout.get('paragraphs', []))} paragraphs")
    print(f"{'='*60}")
    for p_idx, para in enumerate(layout.get('paragraphs', [])):
        print(f"  Paragraph {p_idx}: {len(para)} lines")

    if enable_viz:
        layout_viz = draw_layout_paragraphs(original, layout, ocr)
        viz_paths["layout"] = save_viz_stage(layout_viz, "03_layout_paragraphs", timestamp)

    # Segment and collect all text
    all_texts = []
    mappings = []

    print(f"\n{'='*60}")
    print("SEGMENTATION PHASE")
    print(f"{'='*60}")

    for para_idx, ocr_para in enumerate(layout.get("paragraphs", [])):
        print(f"\n--- Paragraph {para_idx} ---")
        print(f"Lines in paragraph: {len(ocr_para)}")

        # Print structure
        for line_idx, line in enumerate(ocr_para):
            text = line.get("text", "")
            print(f"  Line {line_idx}: '{text[:50]}...' (box: {line.get('box', [])[0] if line.get('box') else 'NO BOX'})")

        seg = (await client.post(
            WORKER_URLS["segment"],
            files={"file": ("image.png", img_bytes)},
            data={"ocr_results_json": json.dumps(ocr_para), "join_all": True}
        )).json()

        merged = seg["merged_results"]
        print(f"  Merged into {len(merged)} groups:")
        for m_idx, m in enumerate(merged):
            print(f"    Group {m_idx}: indices {m.get('group_indices', [])} -> '{m.get('merged_text', '')[:50]}...'")

        if enable_viz:
            seg_viz = draw_segmented_groups(original, merged)
            viz_paths[f"segment_para_{para_idx}"] = save_viz_stage(
                seg_viz, f"04_segmented_para_{para_idx}", timestamp
            )

        start = len(all_texts)
        all_texts.extend([x["merged_text"] for x in merged])
        end = len(all_texts)

        mappings.append({
            "ocr_para": ocr_para,
            "merged": merged,
            "start": start,
            "end": end
        })

    print(f"\n{'='*60}")
    print(f"TRANSLATION PHASE")
    print(f"{'='*60}")
    print(f"Total texts to translate: {len(all_texts)}")
    for idx, txt in enumerate(all_texts):
        print(f"  [{idx}] '{txt[:60]}...'")

    # Batch translation
    translated = []
    if all_texts:
        resp = await client.post(
            WORKER_URLS["multimodal_translate"],
            files={"file": ("image.png", img_bytes)},
            data={
                "texts_json": json.dumps(all_texts),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            },
            timeout=60.0
        )
        resp.raise_for_status()
        translated = resp.json()["translations"]

        print(f"\nTranslations received: {len(translated)}")
        for idx, txt in enumerate(translated):
            print(f"  [{idx}] '{txt[:60]}...'")

    # Merge translations back
    print(f"\n{'='*60}")
    print("MERGE TRANSLATIONS BACK TO LINES")
    print(f"{'='*60}")

    merged_paragraphs = []
    for m_idx, m in enumerate(mappings):
        print(f"\n--- Paragraph {m_idx} ---")
        print(f"  Original lines: {len(m['ocr_para'])}")
        print(f"  Translation range: [{m['start']}:{m['end']}]")

        # Get translations for this paragraph
        block = translated[m["start"]:m["end"]]
        print(f"  Translation block size: {len(block)}")
        print(f"  Merged groups: {len(m['merged'])}")

        # Verify structure
        if len(block) != len(m['merged']):
            print(f"  ⚠️  WARNING: Mismatch! {len(block)} translations vs {len(m['merged'])} groups")

        # Create a deep copy of merged results and update with translations
        merged_with_translations = []
        for entry, translation in zip(m["merged"], block):
            updated_entry = entry.copy()
            updated_entry["merged_text"] = translation
            merged_with_translations.append(updated_entry)
            print(f"  Group {entry.get('group_indices', [])}: '{translation[:40]}...'")

        # Now merge back into OCR line structure
        print(f"\n  Calling merge_translations_smart...")
        print(f"  Input: {len(merged_with_translations)} merged groups")
        print(f"  Output: {len(m['ocr_para'])} lines")

        # Make a deep copy to avoid modifying original
        ocr_para_copy = []
        for line in m["ocr_para"]:
            line_copy = line.copy()
            # Ensure box is preserved
            if "box" in line:
                line_copy["box"] = line["box"]
            ocr_para_copy.append(line_copy)

        merged_result = merge_translations_smart(merged_with_translations, ocr_para_copy)

        # Debug: print final assignments
        print(f"\n  Final line assignments:")
        for idx, line in enumerate(merged_result):
            merged_text = line.get('merged_text', '')
            original_text = line.get('text', '')
            has_box = 'box' in line and line['box']
            print(f"    Line {idx}: '{merged_text[:40]}...' (orig: '{original_text[:20]}...') [has_box: {has_box}]")

        merged_paragraphs.append(merged_result)

    print(f"\n{'='*60}")
    print("FINAL MERGED PARAGRAPHS STRUCTURE")
    print(f"{'='*60}")
    for p_idx, para in enumerate(merged_paragraphs):
        print(f"Paragraph {p_idx}: {len(para)} lines")
        for l_idx, line in enumerate(para):
            text = line.get('merged_text', line.get('text', ''))
            box = line.get('box', [])
            print(f"  Line {l_idx}: '{text[:50]}...' [box: {len(box)} points]")

    # Inpaint
    print(f"\n{'='*60}")
    print("INPAINTING")
    print(f"{'='*60}")

    inpaint = (await client.post(
        WORKER_URLS["inpaint"],
        files={"file": ("image.png", img_bytes)},
        data={"boxes_json": json.dumps([r["box"] for r in ocr])}
    )).json()

    inpainted = Image.open(io.BytesIO(base64.b64decode(inpaint["inpainted_image_base64"])))
    if has_transparency:
        inpainted = inpainted.convert("RGBA")

    if enable_viz:
        mask_viz = draw_inpaint_mask(original, [r["box"] for r in ocr])
        viz_paths["inpaint_mask"] = save_viz_stage(mask_viz, "05_inpaint_mask", timestamp)
        viz_paths["inpainted"] = save_viz_stage(inpainted, "06_inpainted", timestamp)
        ocr_text_viz = draw_ocr_with_text(inpainted, ocr)
        viz_paths["ocr_text_inpainted"] = save_viz_stage(ocr_text_viz, "06b_ocr_text_on_inpainted", timestamp)

    # Draw translations
    print(f"\n{'='*60}")
    print("DRAWING PHASE")
    print(f"{'='*60}")
    print(f"Drawing {len(merged_paragraphs)} paragraphs")

    base = original.convert("RGBA" if has_transparency else "RGB")
    result = draw_paragraphs_polys(inpainted.copy(), merged_paragraphs, base)

    if enable_viz:
        viz_paths["final"] = save_viz_stage(result, "07_final_result", timestamp)
        print(f"\n{'='*60}")
        print("VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"All visualizations saved to: {VIZ_DIR}")
        for stage, path in viz_paths.items():
            print(f"  {stage}: {path}")

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
    """
    Replace paragraph text with translation.
    """
    # Join translated lines
    full_translation = "\n".join(translated) if isinstance(translated, list) else translated

    # Simply set the text - this clears runs and creates a new one
    paragraph.text = full_translation

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
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input PPTX file
        input_path = os.path.join(tmpdir, "input.pptx")
        with open(input_path, "wb") as f:
            f.write(pptx_bytes)

        # Create output directory for PDF
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Run LibreOffice conversion
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", output_dir, input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False
        )

        if result.returncode != 0:
            raise Exception(f"LibreOffice conversion failed: {result.stderr.decode()}")

        # Read the generated PDF
        pdf_path = os.path.join(output_dir, "input.pdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not generated at {pdf_path}")

        with open(pdf_path, "rb") as f:
            content = f.read()

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
