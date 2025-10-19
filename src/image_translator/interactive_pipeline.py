import gradio as gr
import httpx
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import numpy as np
import pandas as pd
import asyncio
from typing import List, Dict, Any
import os
import re

# Import your existing utilities
from image_translator.utils import (
    merge_translations,
    draw_paragraphs_polys,
    extract_text_color_from_diff
)

WORKER_URLS = {
    "ocr": "http://i13hpc69:8001/ocr",
    "segment": "http://i13hpc69:8002/segment",
    "inpaint": "http://i13hpc69:8003/inpaint",
    "translate": "http://i13hpc69:8004/translate",
    "layout": "http://i13hpc69:8005/detect_layout",
}

REQUEST_TIMEOUT = 300

lang_map = {
    "English": "en", "Chinese": "zh", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl",
    "Romanian": "ro", "Turkish": "tr"
}

# Font path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "dejavu-sans.oblique.ttf")


class InteractivePipelineState:
    """Store all pipeline state for human-in-the-loop corrections"""
    def __init__(self):
        self.original_image = None
        self.original_bytes = None
        self.ocr_results = []
        self.layout_data = {}
        self.all_merged_results = []
        self.paragraph_segment_mapping = []
        self.translations = []
        self.inpainted_image = None
        self.final_image = None


def get_points(box):
    """Convert box coordinates to list of (x, y) tuples"""
    if not box:
        return []
    if isinstance(box[0], (int, float)):
        return [(box[j], box[j+1]) for j in range(0, len(box), 2) if len(box) >= j+2]
    else:
        return [(pt[0], pt[1]) for pt in box if len(pt) == 2]


def visualize_ocr_boxes(image, ocr_results):
    """Draw OCR bounding boxes on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    try:
        font = ImageFont.truetype(FONT_PATH, 14)
    except:
        font = ImageFont.load_default()

    for i, result in enumerate(ocr_results):
        points = get_points(result.get("box", []))

        if len(points) >= 4:
            draw.polygon(points, outline=(0, 255, 0), width=2, fill=(0, 255, 0, 30))
            draw.text((points[0][0]+3, points[0][1]+3), f"W{i}", fill=(0, 0, 0), font=font)

    return img


def visualize_layout(image, layout_data, highlight_para=None):
    """Draw layout detection boxes: paragraphs, lines, and words"""
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    try:
        font = ImageFont.truetype(FONT_PATH, 14)
    except:
        font = ImageFont.load_default()

    paragraphs = layout_data.get("paragraphs", [])

    for i, paragraph in enumerate(paragraphs):
        if not paragraph:
            continue

        # Get all word points for paragraph box
        all_pts = []
        for word in paragraph:
            pts = get_points(word.get("box", []))
            all_pts.extend(pts)

        if not all_pts:
            continue

        all_pts_np = np.array(all_pts)
        x_min, y_min = all_pts_np[:, 0].min(), all_pts_np[:, 1].min()
        x_max, y_max = all_pts_np[:, 0].max(), all_pts_np[:, 1].max()

        # Paragraph color and style
        para_color = (0, 0, 255) if i != highlight_para else (255, 0, 0)
        para_width = 3 if i != highlight_para else 4
        para_fill = (0, 0, 255, 10) if i != highlight_para else (255, 0, 0, 20)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=para_color, width=para_width, fill=para_fill)
        draw.text((x_min, y_min - 20), f"P{i}", fill=para_color, font=font)

        # Group words into lines based on y-coordinates
        words_with_pts = [word for word in paragraph if get_points(word.get("box", []))]
        if not words_with_pts:
            continue

        # Sort by min y (top)
        words_with_pts.sort(key=lambda w: min(p[1] for p in get_points(w.get("box", []))))

        # Cluster into lines
        lines = []
        current_line = [words_with_pts[0]]
        for word in words_with_pts[1:]:
            prev_bottom = max(max(p[1] for p in get_points(w.get("box", []))) for w in current_line)
            curr_top = min(p[1] for p in get_points(word.get("box", [])))
            curr_height = max(p[1] for p in get_points(word.get("box", []))) - curr_top
            if curr_top <= prev_bottom + curr_height * 0.3:  # Allow small gap/overlap
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        if current_line:
            lines.append(current_line)

        # Draw lines
        for j, line in enumerate(lines):
            line_pts = []
            for word in line:
                line_pts.extend(get_points(word.get("box", [])))
            if not line_pts:
                continue
            line_pts_np = np.array(line_pts)
            lx_min, ly_min = line_pts_np[:, 0].min(), line_pts_np[:, 1].min()
            lx_max, ly_max = line_pts_np[:, 0].max(), line_pts_np[:, 1].max()

            line_color = (255, 165, 0)
            line_width = 2
            line_fill = (255, 165, 0, 15)
            draw.rectangle([lx_min, ly_min, lx_max, ly_max], outline=line_color, width=line_width, fill=line_fill)
            draw.text((lx_min, ly_min - 15), f"L{j}", fill=line_color, font=font)

        # Draw individual words
        for k, word in enumerate(paragraph):
            pts = get_points(word.get("box", []))
            if pts:
                word_color = (0, 255, 0)
                word_width = 1
                word_fill = (0, 255, 0, 10)
                draw.polygon(pts, outline=word_color, width=word_width, fill=word_fill)
                draw.text((pts[0][0] + 3, pts[0][1] + 3), f"W{k}", fill=(0, 0, 0), font=font)

    return img


def ocr_results_to_dataframe(ocr_results):
    """Convert OCR results to editable dataframe"""
    if not ocr_results:
        return pd.DataFrame(columns=["ID", "Text", "Confidence", "Box (JSON)"])

    data = []
    for i, result in enumerate(ocr_results):
        data.append({
            "ID": i,
            "Text": result.get("text", ""),
            "Confidence": round(result.get("confidence", 0), 3),
            "Box (JSON)": json.dumps(result.get("box", []))
        })

    return pd.DataFrame(data)


def dataframe_to_ocr_results(df):
    """Convert dataframe back to OCR results format"""
    results = []
    for _, row in df.iterrows():
        try:
            box = json.loads(row["Box (JSON)"])
        except:
            box = []

        results.append({
            "text": str(row["Text"]),
            "confidence": float(row["Confidence"]),
            "box": box
        })

    return results


def segments_to_dataframe(all_merged_results):
    """Convert segmentation results to editable dataframe"""
    if not all_merged_results:
        return pd.DataFrame(columns=["Para", "Seg", "Merged Text", "Group Indices"])

    data = []
    for para_idx, merged_results in enumerate(all_merged_results):
        for seg_idx, segment in enumerate(merged_results):
            data.append({
                "Para": para_idx,
                "Seg": seg_idx,
                "Merged Text": segment.get("merged_text", ""),
                "Group Indices": json.dumps(segment.get("group_indices", []))
            })

    return pd.DataFrame(data)


def dataframe_to_segments(df):
    """Convert dataframe back to segmentation format"""
    all_merged_results = []

    if df.empty:
        return all_merged_results

    grouped = df.groupby("Para")
    for para_idx, group in grouped:
        merged_results = []
        for _, row in group.iterrows():
            try:
                group_indices = json.loads(row["Group Indices"])
            except:
                group_indices = []

            merged_results.append({
                "merged_text": str(row["Merged Text"]),
                "group_indices": group_indices
            })
        all_merged_results.append(merged_results)

    return all_merged_results


def translations_to_dataframe(translations, all_merged_results=None):
    """Convert translations to editable dataframe with context"""
    if not translations:
        return pd.DataFrame(columns=["ID", "Original", "Translation"])

    data = []
    if all_merged_results:
        all_originals = []
        for merged_results in all_merged_results:
            for segment in merged_results:
                all_originals.append(segment.get("merged_text", ""))

        for i, (orig, trans) in enumerate(zip(all_originals, translations)):
            data.append({
                "ID": i,
                "Original": orig,
                "Translation": trans
            })
    else:
        for i, trans in enumerate(translations):
            data.append({
                "ID": i,
                "Original": "",
                "Translation": trans
            })

    return pd.DataFrame(data)


def dataframe_to_translations(df):
    """Convert dataframe back to translations list"""
    return df["Translation"].tolist()


async def run_ocr_async(image_bytes):
    """Run OCR on image"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            WORKER_URLS["ocr"],
            files={"file": ("image.png", image_bytes)},
            timeout=REQUEST_TIMEOUT
        )
        return response.json()["results"]


async def run_layout_async(image_bytes, ocr_results):
    """Run layout detection"""
    async with httpx.AsyncClient() as client:
        multipart_data = {
            "file": ("image.png", image_bytes, "image/png"),
            "ocr_results_json": (None, json.dumps(ocr_results))
        }
        response = await client.post(
            WORKER_URLS["layout"],
            files=multipart_data,
            timeout=REQUEST_TIMEOUT
        )
        return response.json()


async def run_segmentation_async(image_bytes, paragraph):
    """Run segmentation for a single paragraph"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            WORKER_URLS["segment"],
            files={"file": ("image.png", image_bytes)},
            data={"ocr_results_json": json.dumps(paragraph)},
            timeout=REQUEST_TIMEOUT,
        )
        return response.json()["merged_results"]


async def run_translation_async(texts, tgt_lang):
    """Run batch translation (no src_lang needed)"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            WORKER_URLS["translate"],
            data={
                "texts_json": json.dumps(texts),
                "src_lang": "English",
                "tgt_lang": tgt_lang,
            },
            timeout=REQUEST_TIMEOUT,
        )
        return response.json()["translations"]


async def run_inpaint_async(image_bytes, boxes):
    """Run inpainting"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            WORKER_URLS["inpaint"],
            files={"file": ("image.png", image_bytes)},
            data={"boxes_json": json.dumps(boxes)},
            timeout=REQUEST_TIMEOUT,
        )
        inpaint_data = response.json()
        inpainted_image_b64 = inpaint_data["inpainted_image_base64"]
        return Image.open(io.BytesIO(base64.b64decode(inpainted_image_b64)))


def create_gradio_app():
    """Create the interactive Gradio interface"""

    state = InteractivePipelineState()

    css = """
    .dataframe-container {font-size: 12px;}
    .stage-header {background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .highlight {background-color: #ffeb3b;}
    """

    with gr.Blocks(css=css, title="Interactive Image Translation") as demo:
        gr.Markdown("# 🌐 Interactive Image Translation Pipeline")
        gr.Markdown("Edit OCR text, segmentation, and translations in interactive tables")

        # Add help section at the top
        with gr.Accordion("❓ Quick Help & Tips", open=False):
            gr.Markdown("""
            ### 🎯 Workflow Overview
            1. **Upload image** → Run OCR → Review & edit detections
            2. **Detect layout** → Review paragraph groupings
            3. **Segment text** → Edit merged text if needed
            4. **Translate** → Correct any mistranslations
            5. **Inpaint & Draw** → Adjust parameters → Download

            ### ⌨️ Table Editing Tips
            - **Click cell** to edit
            - **Tab** to move between cells
            - **Enter** to confirm edit
            - **Escape** to cancel

            ### 🎨 Visual Indicators
            - 🟢 **Green boxes** = Words
            - 🟠 **Orange boxes** = Lines
            - 🔵 **Blue boxes** = Paragraphs
            - 🔴 **Red boxes** = Selected/highlighted
            - **Thicker border** = Currently active

            ### 💡 Pro Tips
            - Use **Highlight** to verify which box you're editing
            - Apply **Batch Operations** for consistent fixes
            - **Export data** before major changes (backup)
            - **Redraw** multiple times to perfect the result

            ### 🔧 Troubleshooting
            - If table not updating → Click the "Apply" button
            - If preview frozen → Enter an ID and click "Highlight"
            - If translation wrong → Check original text in context column
            """)

        # State variables
        state_image_bytes = gr.State(None)
        state_ocr = gr.State([])
        state_layout = gr.State({})
        state_segments = gr.State([])
        state_mapping = gr.State([])
        state_translations = gr.State([])
        state_inpainted = gr.State(None)

        # Add import functionality at the top
        with gr.Accordion("📥 Import Previous Session", open=False):
            gr.Markdown("**Resume work from exported data**")
            with gr.Row():
                with gr.Column():
                    import_ocr_file = gr.File(label="Import OCR JSON", file_types=[".json"])
                    import_ocr_btn = gr.Button("📥 Load OCR Data")
                with gr.Column():
                    import_trans_file = gr.File(label="Import Translations JSON", file_types=[".json"])
                    import_trans_btn = gr.Button("📥 Load Translations")
            import_status = gr.Textbox(label="Import Status", interactive=False)

        # ==================== TAB 1: OCR ====================
        with gr.Tab("1️⃣ OCR Detection"):
            gr.Markdown("### Step 1: Extract text from image")

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Image", type="pil")
                    tgt_lang = gr.Dropdown(
                        choices=list(lang_map.keys()),
                        value="English",
                        label="Target Language"
                    )
                    run_ocr_btn = gr.Button("🔍 Run OCR", variant="primary", size="lg")

                    # Add statistics panel
                    with gr.Group():
                        gr.Markdown("**📊 Statistics**")
                        ocr_stats = gr.Textbox(
                            label="OCR Stats",
                            value="No data yet",
                            interactive=False,
                            lines=4
                        )

                with gr.Column(scale=2):
                    ocr_preview = gr.Image(label="OCR Detection Preview (Green boxes for words)")

            gr.Markdown("### ✏️ Edit OCR Results")
            gr.Markdown("**Table editing** - Click cells to edit text or confidence")

            ocr_dataframe = gr.Dataframe(
                headers=["ID", "Text", "Confidence", "Box (JSON)"],
                datatype=["number", "str", "number", "str"],
                row_count="dynamic",
                col_count=(4, "fixed"),
                interactive=True,
                wrap=True
            )

            with gr.Row():
                update_ocr_btn = gr.Button("✅ Apply OCR Edits", variant="primary")
                ocr_status = gr.Textbox(label="Status", interactive=False, scale=2)

        # ==================== TAB 2: LAYOUT DETECTION ====================
        with gr.Tab("2️⃣ Layout Detection"):
            gr.Markdown("### Step 2: Detect paragraphs from OCR results")

            with gr.Row():
                with gr.Column(scale=1):
                    run_layout_btn = gr.Button("📋 Detect Layout", variant="primary", size="lg")

                    # Add statistics panel for layout
                    with gr.Group():
                        gr.Markdown("**📊 Layout Statistics**")
                        layout_stats = gr.Textbox(
                            label="Layout Stats",
                            value="No data yet",
                            interactive=False,
                            lines=4
                        )

                with gr.Column(scale=2):
                    layout_preview = gr.Image(label="Layout Detection Preview (Blue: paragraphs, Orange: lines, Green: words)")

            with gr.Accordion("🎯 Advanced: Paragraph Selection", open=False):
                gr.Markdown("**Visual paragraph navigation**")
                with gr.Row():
                    selected_para_idx = gr.Number(label="Enter Paragraph ID", value=-1, precision=0)
                    highlight_para_btn = gr.Button("🔍 Highlight Paragraph")

        # ==================== TAB 3: SEGMENTATION ====================
        with gr.Tab("3️⃣ Segmentation"):
            gr.Markdown("### Step 3: Segment text for translation")

            with gr.Row():
                with gr.Column(scale=1):
                    run_segment_btn = gr.Button("✂️ Segment Text", variant="primary", size="lg")
                    segment_info = gr.Textbox(
                        label="Segmentation Summary",
                        lines=5,
                        interactive=False
                    )

            gr.Markdown("### ✏️ Edit Segmentation")
            gr.Markdown("Edit 'Merged Text' to change what gets translated together. Group Indices show which OCR results are merged.")

            segments_dataframe = gr.Dataframe(
                headers=["Para", "Seg", "Merged Text", "Group Indices"],
                datatype=["number", "number", "str", "str"],
                row_count="dynamic",
                col_count=(4, "fixed"),
                interactive=True,
                wrap=True
            )

            with gr.Row():
                update_segments_btn = gr.Button("✅ Apply Segmentation Edits", variant="primary")
                segments_status = gr.Textbox(label="Status", interactive=False, scale=2)

            with gr.Accordion("🎯 Advanced: Segment Editing", open=False):
                with gr.Row():
                    selected_seg_row = gr.Number(label="Edit Segment (Row number in table)", value=-1, precision=0)
                    new_seg_text = gr.Textbox(label="New merged text", placeholder="Edit segment text...")
                    update_seg_text_btn = gr.Button("✏️ Update Segment Text")
                with gr.Row():
                    gr.Markdown("**Segment Operations:**")
                    split_seg_btn = gr.Button("✂️ Split Segment at Space (creates 2 segments)")
                    delete_seg_btn = gr.Button("🗑️ Delete Segment", variant="stop")

        # ==================== TAB 4: TRANSLATION ====================
        with gr.Tab("4️⃣ Translation"):
            gr.Markdown("### Step 4: Translate segmented text")

            with gr.Row():
                with gr.Column(scale=1):
                    run_translate_btn = gr.Button("🌐 Translate All", variant="primary", size="lg")
                    translate_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    pass
            gr.Markdown("### ✏️ Edit Translations")
            gr.Markdown("Edit the 'Translation' column to correct any mistranslations. Original text is shown for reference.")

            translations_dataframe = gr.Dataframe(
                headers=["ID", "Original", "Translation"],
                datatype=["number", "str", "str"],
                row_count="dynamic",
                col_count=(3, "fixed"),
                interactive=True,
                wrap=True
            )

            with gr.Row():
                update_translations_btn = gr.Button("✅ Apply Translation Edits", variant="primary")
                translation_status = gr.Textbox(label="Status", interactive=False, scale=2)

            with gr.Accordion("🔧 Batch Operations", open=False):
                gr.Markdown("**Apply corrections to multiple translations at once**")
                with gr.Row():
                    with gr.Column():
                        find_text = gr.Textbox(label="Find text in translations", placeholder="e.g., 'color'")
                        replace_text = gr.Textbox(label="Replace with", placeholder="e.g., 'colour'")
                        find_replace_btn = gr.Button("🔄 Find & Replace All")
                    with gr.Column():
                        gr.Markdown("**Quick fixes:**")
                        fix_spaces_btn = gr.Button("📏 Fix Extra Spaces")
                        fix_punctuation_btn = gr.Button("✒️ Fix Punctuation Spacing")
                        capitalize_first_btn = gr.Button("🔠 Capitalize First Letter")

        # ==================== TAB 5: INPAINT & DRAW ====================
        with gr.Tab("5️⃣ Draw Result"):
            gr.Markdown("### Step 5: Inpaint original text and draw translations")

            with gr.Row():
                with gr.Column():
                    run_inpaint_btn = gr.Button("🎨 Inpaint Original Text", variant="primary", size="lg")
                    inpaint_preview = gr.Image(label="Inpainted Image")

                with gr.Column():
                    run_draw_btn = gr.Button("✍️ Draw Translations", variant="primary", size="lg")
                    final_preview = gr.Image(label="Final Result")

            gr.Markdown("### Drawing Parameters")
            with gr.Row():
                font_size_min = gr.Slider(5, 20, value=5, step=1, label="Minimum Font Size")
                padding = gr.Slider(0, 10, value=2, step=1, label="Text Padding")
                redraw_btn = gr.Button("🔄 Redraw with New Parameters")

            with gr.Row():
                download_btn = gr.Button("💾 Download Result", variant="primary", size="lg")
                download_output = gr.File(label="Download")

            with gr.Accordion("📤 Export Data", open=False):
                gr.Markdown("**Export intermediate results for external editing or backup**")
                with gr.Row():
                    export_ocr_btn = gr.Button("📄 Export OCR Results (JSON)")
                    export_ocr_file = gr.File(label="OCR JSON")
                with gr.Row():
                    export_translations_btn = gr.Button("📄 Export Translations (JSON)")
                    export_trans_file = gr.File(label="Translations JSON")
                with gr.Row():
                    export_all_btn = gr.Button("📦 Export All Data (ZIP)")
                    export_all_file = gr.File(label="Complete Data ZIP")

        # ==================== CALLBACKS ====================

        def process_ocr(image, tgt):
            """Run OCR detection"""
            if image is None:
                return None, pd.DataFrame(), None, [], "❌ Please upload an image", "No data"

            # Save image
            state.original_image = image
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            image_bytes = buf.getvalue()
            state.original_bytes = image_bytes

            # Run OCR
            ocr_results = asyncio.run(run_ocr_async(image_bytes))
            state.ocr_results = ocr_results

            # Visualize
            preview = visualize_ocr_boxes(image, ocr_results)

            # Convert to dataframe
            df = ocr_results_to_dataframe(ocr_results)

            # Calculate statistics
            total_boxes = len(ocr_results)
            total_chars = sum(len(r.get("text", "")) for r in ocr_results)
            avg_confidence = np.mean([r.get("confidence", 0) for r in ocr_results]) if ocr_results else 0
            low_conf_count = sum(1 for r in ocr_results if r.get("confidence", 0) < 0.8)

            stats = f"""Total detections: {total_boxes}
Total characters: {total_chars}
Avg confidence: {avg_confidence:.2%}
Low confidence (<80%): {low_conf_count}"""

            status = f"✅ Detected {len(ocr_results)} text regions"

            return preview, df, image_bytes, ocr_results, status, stats


        def update_ocr_from_dataframe(df, image_bytes):
            """Update OCR results from edited dataframe"""
            try:
                ocr_results = dataframe_to_ocr_results(df)
                state.ocr_results = ocr_results

                if state.original_image:
                    preview = visualize_ocr_boxes(state.original_image, ocr_results)
                    return preview, ocr_results, f"✅ Updated {len(ocr_results)} OCR results"

                return None, ocr_results, "✅ OCR results updated"
            except Exception as e:
                return None, [], f"❌ Error: {str(e)}"


        def process_layout(image_bytes, ocr_results):
            """Run layout detection"""
            if not ocr_results or not image_bytes:
                return None, {}, "No data", "❌ Run OCR first"

            layout_data = asyncio.run(run_layout_async(image_bytes, ocr_results))
            state.layout_data = layout_data

            # Visualize
            preview = visualize_layout(state.original_image, layout_data)

            # Calculate statistics
            paragraphs = layout_data.get("paragraphs", [])
            num_paras = len(paragraphs)
            total_words = sum(len(p) for p in paragraphs)
            avg_words_per_para = total_words / num_paras if num_paras > 0 else 0

            stats = f"""Total paragraphs: {num_paras}
Total words: {total_words}
Avg words per paragraph: {avg_words_per_para:.1f}"""

            status = f"✅ Detected {num_paras} paragraphs"

            return preview, layout_data, stats, status


        def highlight_paragraph(idx, layout_data):
            """Highlight a specific paragraph"""
            try:
                idx = int(idx)
                if idx < 0 or state.original_image is None:
                    return visualize_layout(state.original_image, layout_data)

                preview = visualize_layout(state.original_image, layout_data, highlight_para=idx)
                return preview
            except:
                return visualize_layout(state.original_image, layout_data)


        def process_segmentation(image_bytes, layout_data):
            """Run segmentation on all paragraphs"""
            if not layout_data or not image_bytes:
                return "", pd.DataFrame(), [], [], "❌ Run layout detection first"

            paragraphs = layout_data.get("paragraphs", [])

            # Run segmentation for each paragraph
            all_merged_results = []
            paragraph_segment_mapping = []
            all_texts = []

            for para_idx, paragraph in enumerate(paragraphs):
                merged_results = asyncio.run(run_segmentation_async(image_bytes, paragraph))
                all_merged_results.append(merged_results)

                # Collect texts for translation
                start_idx = len(all_texts)
                texts = [res["merged_text"] for res in merged_results]
                all_texts.extend(texts)
                end_idx = len(all_texts)

                paragraph_segment_mapping.append({
                    "para_idx": para_idx,
                    "ocr_paragraph": paragraph,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "merged_results": merged_results
                })

            state.all_merged_results = all_merged_results
            state.paragraph_segment_mapping = paragraph_segment_mapping

            # Create summary
            summary = f"Total segments: {len(all_texts)}\n\n"
            for i, para_map in enumerate(paragraph_segment_mapping):
                num_segs = para_map["end_idx"] - para_map["start_idx"]
                summary += f"Paragraph {i}: {num_segs} segments\n"

            # Convert to dataframe
            df = segments_to_dataframe(all_merged_results)

            status = f"✅ Created {len(all_texts)} segments across {len(paragraphs)} paragraphs"

            return summary, df, all_merged_results, paragraph_segment_mapping, status


        def update_segments_from_dataframe(df):
            """Update segmentation from edited dataframe"""
            try:
                all_merged_results = dataframe_to_segments(df)
                state.all_merged_results = all_merged_results

                total_segments = sum(len(mr) for mr in all_merged_results)
                return all_merged_results, f"✅ Updated {total_segments} segments"
            except Exception as e:
                return [], f"❌ Error: {str(e)}"


        def update_segment_text(row_idx, new_text, df):
            """Update text for a specific segment"""
            try:
                row_idx = int(row_idx)
                if row_idx < 0 or row_idx >= len(df):
                    return df, "❌ Invalid row number"

                if not new_text or not new_text.strip():
                    return df, "❌ Text cannot be empty"

                df.loc[row_idx, "Merged Text"] = new_text.strip()

                # Update state
                all_merged_results = dataframe_to_segments(df)
                state.all_merged_results = all_merged_results

                return df, f"✅ Updated segment at row {row_idx}"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def split_segment(row_idx, df):
            """Split a segment at the middle space"""
            try:
                row_idx = int(row_idx)
                if row_idx < 0 or row_idx >= len(df):
                    return df, "❌ Invalid row number"

                text = df.loc[row_idx, "Merged Text"]
                words = text.split()

                if len(words) <= 1:
                    return df, "❌ Cannot split: segment has only one word"

                # Split at middle
                mid = len(words) // 2
                first_half = " ".join(words[:mid])
                second_half = " ".join(words[mid:])

                para = df.loc[row_idx, "Para"]
                seg = df.loc[row_idx, "Seg"]

                # Update current row
                df.loc[row_idx, "Merged Text"] = first_half

                # Insert new row after current
                new_row = pd.DataFrame([{
                    "Para": para,
                    "Seg": seg + 0.5,  # Temporary seg number
                    "Merged Text": second_half,
                    "Group Indices": "[]"  # Will need to be manually set
                }])

                df = pd.concat([df.iloc[:row_idx+1], new_row, df.iloc[row_idx+1:]], ignore_index=True)

                # Renumber segments within paragraph
                para_mask = df["Para"] == para
                df.loc[para_mask, "Seg"] = range(para_mask.sum())

                # Update state
                all_merged_results = dataframe_to_segments(df)
                state.all_merged_results = all_merged_results

                return df, f"✅ Split segment at row {row_idx} into 2 parts"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def delete_segment(row_idx, df):
            """Delete a segment"""
            try:
                row_idx = int(row_idx)
                if row_idx < 0 or row_idx >= len(df):
                    return df, "❌ Invalid row number"

                para = df.loc[row_idx, "Para"]

                # Remove row
                df = df.drop(df.index[row_idx]).reset_index(drop=True)

                # Renumber segments within affected paragraph
                para_mask = df["Para"] == para
                df.loc[para_mask, "Seg"] = range(para_mask.sum())

                # Update state
                all_merged_results = dataframe_to_segments(df)
                state.all_merged_results = all_merged_results

                return df, f"✅ Deleted segment at row {row_idx}"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def process_translation(all_merged_results, mapping, tgt):
            """Run batch translation"""
            if not all_merged_results:
                return pd.DataFrame(), [], "❌ Run segmentation first"

            # Collect all texts
            all_texts = []
            for merged_results in all_merged_results:
                texts = [res["merged_text"] for res in merged_results]
                all_texts.extend(texts)

            if not all_texts:
                return pd.DataFrame(), [], "❌ No text to translate"

            # Get language code
            tgt_code = tgt

            # Translate
            translations = asyncio.run(run_translation_async(all_texts, tgt_code))
            state.translations = translations

            # Convert to dataframe
            df = translations_to_dataframe(translations, all_merged_results)

            status = f"✅ Translated {len(translations)} segments to {tgt}"

            return df, translations, status


        def update_translations_from_dataframe(df):
            """Update translations from edited dataframe"""
            try:
                translations = dataframe_to_translations(df)
                state.translations = translations

                return translations, f"✅ Updated {len(translations)} translations"
            except Exception as e:
                return [], f"❌ Error: {str(e)}"


        def find_replace_translations(find_text, replace_text, df):
            """Find and replace text in all translations"""
            try:
                if not find_text:
                    return df, "❌ Please enter text to find"

                count = 0
                for idx in df.index:
                    old_trans = df.loc[idx, "Translation"]
                    new_trans = old_trans.replace(find_text, replace_text)
                    if new_trans != old_trans:
                        df.loc[idx, "Translation"] = new_trans
                        count += 1

                # Update state
                translations = dataframe_to_translations(df)
                state.translations = translations

                return df, f"✅ Replaced in {count} translations"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def fix_extra_spaces(df):
            """Remove extra spaces in translations"""
            try:
                import re
                count = 0
                for idx in df.index:
                    old_trans = df.loc[idx, "Translation"]
                    # Replace multiple spaces with single space
                    new_trans = re.sub(r'\s+', ' ', old_trans).strip()
                    if new_trans != old_trans:
                        df.loc[idx, "Translation"] = new_trans
                        count += 1

                # Update state
                translations = dataframe_to_translations(df)
                state.translations = translations

                return df, f"✅ Fixed spaces in {count} translations"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def fix_punctuation_spacing(df):
            """Fix spacing around punctuation marks"""
            try:
                import re
                count = 0
                for idx in df.index:
                    old_trans = df.loc[idx, "Translation"]
                    new_trans = old_trans

                    # Remove space before punctuation
                    new_trans = re.sub(r'\s+([.,!?;:])', r'\1', new_trans)
                    # Ensure space after punctuation (except at end)
                    new_trans = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', new_trans)

                    if new_trans != old_trans:
                        df.loc[idx, "Translation"] = new_trans
                        count += 1

                # Update state
                translations = dataframe_to_translations(df)
                state.translations = translations

                return df, f"✅ Fixed punctuation in {count} translations"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def capitalize_first_letter(df):
            """Capitalize first letter of each translation"""
            try:
                count = 0
                for idx in df.index:
                    old_trans = df.loc[idx, "Translation"]
                    if old_trans and not old_trans[0].isupper():
                        new_trans = old_trans[0].upper() + old_trans[1:]
                        df.loc[idx, "Translation"] = new_trans
                        count += 1

                # Update state
                translations = dataframe_to_translations(df)
                state.translations = translations

                return df, f"✅ Capitalized {count} translations"
            except Exception as e:
                return df, f"❌ Error: {str(e)}"


        def process_inpaint(image_bytes, ocr_results):
            """Run inpainting"""
            if not ocr_results or not image_bytes:
                return None, None

            boxes = [r["box"] for r in ocr_results]
            inpainted = asyncio.run(run_inpaint_async(image_bytes, boxes))
            state.inpainted_image = inpainted

            return inpainted, inpainted


        def process_drawing(inpainted_img, all_merged_results, mapping, translations,
                           font_min, pad):
            """Draw translations on inpainted image"""
            if inpainted_img is None:
                return None, "❌ Run inpainting first"

            if not translations:
                return None, "❌ Run translation first"

            # Distribute translations back to paragraphs
            ocr_para_trans_results = []

            for map_entry in mapping:
                start_idx = map_entry["start_idx"]
                end_idx = map_entry["end_idx"]
                para_translations = translations[start_idx:end_idx]

                # Assign translations to merged results
                merged_results = map_entry["merged_results"]
                for entry, translation in zip(merged_results, para_translations):
                    entry["merged_text"] = translation

                # Merge translations back to OCR results
                ocr_paragraph = map_entry["ocr_paragraph"]
                ocr_trans_results = merge_translations(merged_results, ocr_paragraph)
                ocr_para_trans_results.append(ocr_trans_results)

            # Draw
            final_image = draw_paragraphs_polys(
                inpainted_img.copy(),
                ocr_para_trans_results,
                state.original_image,
                padding=pad,
                font_min=font_min
            )

            state.final_image = final_image

            return final_image, "✅ Drawing complete"


        def download_result():
            """Prepare download"""
            if state.final_image is None:
                return None

            buf = io.BytesIO()
            state.final_image.save(buf, format='PNG')
            buf.seek(0)
            return buf


        def export_ocr_results():
            """Export OCR results as JSON"""
            if not state.ocr_results:
                return None

            buf = io.BytesIO()
            json_str = json.dumps(state.ocr_results, indent=2, ensure_ascii=False)
            buf.write(json_str.encode('utf-8'))
            buf.seek(0)
            return buf


        def export_translations():
            """Export translations as JSON"""
            if not state.translations:
                return None

            buf = io.BytesIO()
            json_str = json.dumps(state.translations, indent=2, ensure_ascii=False)
            buf.write(json_str.encode('utf-8'))
            buf.seek(0)
            return buf


        def export_all_data():
            """Export all pipeline data as ZIP"""
            import zipfile

            if not state.ocr_results:
                return None

            buf = io.BytesIO()

            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add OCR results
                if state.ocr_results:
                    ocr_json = json.dumps(state.ocr_results, indent=2, ensure_ascii=False)
                    zipf.writestr('ocr_results.json', ocr_json)

                # Add layout data
                if state.layout_data:
                    layout_json = json.dumps(state.layout_data, indent=2, ensure_ascii=False)
                    zipf.writestr('layout_data.json', layout_json)

                # Add segmentation
                if state.all_merged_results:
                    seg_json = json.dumps(state.all_merged_results, indent=2, ensure_ascii=False)
                    zipf.writestr('segmentation.json', seg_json)

                # Add translations
                if state.translations:
                    trans_json = json.dumps(state.translations, indent=2, ensure_ascii=False)
                    zipf.writestr('translations.json', trans_json)

                # Add original image
                if state.original_image:
                    img_buf = io.BytesIO()
                    state.original_image.save(img_buf, format='PNG')
                    zipf.writestr('original_image.png', img_buf.getvalue())

                # Add final image
                if state.final_image:
                    final_buf = io.BytesIO()
                    state.final_image.save(final_buf, format='PNG')
                    zipf.writestr('final_result.png', final_buf.getvalue())

            buf.seek(0)
            return buf


        def import_ocr_data(file):
            """Import OCR results from JSON file"""
            try:
                if file is None:
                    return pd.DataFrame(), None, [], "❌ No file selected"

                # Read file
                content = file.read() if hasattr(file, 'read') else open(file.name, 'rb').read()
                ocr_results = json.loads(content)

                state.ocr_results = ocr_results

                # Convert to dataframe
                df = ocr_results_to_dataframe(ocr_results)

                # Update preview if image exists
                preview = None
                if state.original_image:
                    preview = visualize_ocr_boxes(state.original_image, ocr_results)

                return df, preview, ocr_results, f"✅ Imported {len(ocr_results)} OCR results"
            except Exception as e:
                return pd.DataFrame(), None, [], f"❌ Import error: {str(e)}"


        def import_translations_data(file):
            """Import translations from JSON file"""
            try:
                if file is None:
                    return pd.DataFrame(), [], "❌ No file selected"

                # Read file
                content = file.read() if hasattr(file, 'read') else open(file.name, 'rb').read()
                translations = json.loads(content)

                state.translations = translations

                # Convert to dataframe
                df = translations_to_dataframe(translations, state.all_merged_results)

                return df, translations, f"✅ Imported {len(translations)} translations"
            except Exception as e:
                return pd.DataFrame(), [], f"❌ Import error: {str(e)}"


        # ==================== WIRE UP CALLBACKS ====================

        run_ocr_btn.click(
            process_ocr,
            inputs=[input_image, tgt_lang],
            outputs=[ocr_preview, ocr_dataframe, state_image_bytes, state_ocr, ocr_status, ocr_stats]
        )

        update_ocr_btn.click(
            update_ocr_from_dataframe,
            inputs=[ocr_dataframe, state_image_bytes],
            outputs=[ocr_preview, state_ocr, ocr_status]
        )

        run_layout_btn.click(
            process_layout,
            inputs=[state_image_bytes, state_ocr],
            outputs=[layout_preview, state_layout, layout_stats, ocr_status]
        )

        highlight_para_btn.click(
            highlight_paragraph,
            inputs=[selected_para_idx, state_layout],
            outputs=[layout_preview]
        )

        run_segment_btn.click(
            process_segmentation,
            inputs=[state_image_bytes, state_layout],
            outputs=[segment_info, segments_dataframe, state_segments, state_mapping, segments_status]
        )

        update_segments_btn.click(
            update_segments_from_dataframe,
            inputs=[segments_dataframe],
            outputs=[state_segments, segments_status]
        )

        update_seg_text_btn.click(
            update_segment_text,
            inputs=[selected_seg_row, new_seg_text, segments_dataframe],
            outputs=[segments_dataframe, segments_status]
        )

        split_seg_btn.click(
            split_segment,
            inputs=[selected_seg_row, segments_dataframe],
            outputs=[segments_dataframe, segments_status]
        )

        delete_seg_btn.click(
            delete_segment,
            inputs=[selected_seg_row, segments_dataframe],
            outputs=[segments_dataframe, segments_status]
        )

        run_translate_btn.click(
            process_translation,
            inputs=[state_segments, state_mapping, tgt_lang],
            outputs=[translations_dataframe, state_translations, translate_status]
        )

        update_translations_btn.click(
            update_translations_from_dataframe,
            inputs=[translations_dataframe],
            outputs=[state_translations, translation_status]
        )

        find_replace_btn.click(
            find_replace_translations,
            inputs=[find_text, replace_text, translations_dataframe],
            outputs=[translations_dataframe, translation_status]
        )

        fix_spaces_btn.click(
            fix_extra_spaces,
            inputs=[translations_dataframe],
            outputs=[translations_dataframe, translation_status]
        )

        fix_punctuation_btn.click(
            fix_punctuation_spacing,
            inputs=[translations_dataframe],
            outputs=[translations_dataframe, translation_status]
        )

        capitalize_first_btn.click(
            capitalize_first_letter,
            inputs=[translations_dataframe],
            outputs=[translations_dataframe, translation_status]
        )

        run_inpaint_btn.click(
            process_inpaint,
            inputs=[state_image_bytes, state_ocr],
            outputs=[inpaint_preview, state_inpainted]
        )

        run_draw_btn.click(
            process_drawing,
            inputs=[
                state_inpainted, state_segments, state_mapping,
                state_translations, font_size_min, padding
            ],
            outputs=[final_preview, translate_status]
        )

        redraw_btn.click(
            process_drawing,
            inputs=[
                state_inpainted, state_segments, state_mapping,
                state_translations, font_size_min, padding
            ],
            outputs=[final_preview, translate_status]
        )

        download_btn.click(
            download_result,
            outputs=[download_output]
        )

        export_ocr_btn.click(
            export_ocr_results,
            outputs=[export_ocr_file]
        )

        export_translations_btn.click(
            export_translations,
            outputs=[export_trans_file]
        )

        export_all_btn.click(
            export_all_data,
            outputs=[export_all_file]
        )

        import_ocr_btn.click(
            import_ocr_data,
            inputs=[import_ocr_file],
            outputs=[ocr_dataframe, ocr_preview, state_ocr, import_status]
        )

        import_trans_btn.click(
            import_translations_data,
            inputs=[import_trans_file],
            outputs=[translations_dataframe, state_translations, import_status]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7868)
