import os
import json
from pathlib import Path
from typing import List, Union, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import io

# -----------------------------
# Config (change as needed)
# -----------------------------
FONT_PATH = "fonts/dejavu-sans.ttf"   # choose appropriate TTF; fall back to default if missing
GLYPH_FONT_SIZE = 48                  # default glyph render size (will be scaled if needed)
GLYPH_PADDING = 6                      # pixels padding when rendering glyph (around text)
MASK_PADDING = 4                       # pixels padding around detected rectangle in mask
OUTPUT_DIR = "fluxtext_inputs"
# -----------------------------


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def entry_to_bbox(entry: Union[dict, list]) -> np.ndarray:
    """
    Convert an entry (dict or list-of-dicts) to a single bounding rect polygon (4x2 int numpy array).
    Returns rectangle polygon: [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
    """
    if isinstance(entry, dict):
        pts = np.array(entry.get("box", entry.get("box", [])))
    elif isinstance(entry, list):
        pts_list = []
        for w in entry:
            if "box" in w:
                pts_list.append(np.array(w["box"]))
        if not pts_list:
            raise ValueError("List entry contains no 'box' fields")
        pts = np.vstack(pts_list)
    else:
        raise ValueError("Entry must be dict or list")

    if pts.size == 0:
        raise ValueError("Empty box coordinates")

    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))

    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
    return rect


def get_merged_text(entry: Union[dict, list]) -> str:
    """
    Extract merged_text (or text) from entry. If entry is list, join with spaces.
    """
    if isinstance(entry, dict):
        return str(entry.get("merged_text") or entry.get("text") or "").strip()
    else:
        parts = []
        for w in entry:
            t = w.get("merged_text") or w.get("text") or ""
            if t:
                parts.append(str(t).strip())
        return " ".join(parts).strip()


def create_mask_image_for_box(base_image: Image.Image, rect: np.ndarray, padding: int = 4) -> Image.Image:
    """
    Make a single-channel (L) mask image same size as base_image where rect area is white (255).
    rect is a 4x2 polygon (rectangle). padding expands the rect by pixels.
    """
    w, h = base_image.size
    mask = np.zeros((h, w), dtype=np.uint8)

    x_min = max(0, rect[:, 0].min() - padding)
    x_max = min(w - 1, rect[:, 0].max() + padding)
    y_min = max(0, rect[:, 1].min() - padding)
    y_max = min(h - 1, rect[:, 1].max() + padding)

    rect_poly = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
    cv2.fillPoly(mask, [rect_poly], 255)
    return Image.fromarray(mask)


def render_glyph_on_full_image(base_image: Image.Image, rect: np.ndarray, text: str,
                               font_path: str, font_size: int = 48, padding: int = 4) -> Image.Image:
    """
    Render the target text onto a full-size transparent image, same size as base_image,
    with text positioned inside rect.

    Args:
        base_image: original image (RGB)
        rect: 4x2 polygon rectangle [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
        text: target text to render
        font_path: path to TTF font
        font_size: starting font size (will shrink to fit)
        padding: inner padding inside rect

    Returns:
        RGBA image (same size as base_image) with text rendered in rect
    """
    w, h = base_image.size
    glyph_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glyph_img)

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Compute rect size
    x_min, y_min = rect[:, 0].min() + padding, rect[:, 1].min() + padding
    x_max, y_max = rect[:, 0].max() - padding, rect[:, 1].max() - padding
    box_w, box_h = x_max - x_min, y_max - y_min

    # Split text into lines if needed
    lines = text.split("\n")
    num_lines = len(lines)

    # Shrink font until all lines fit inside box
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    text_h = line_h * num_lines
    max_line_w = max(draw.textlength(line, font=font) for line in lines)

    while (text_h > box_h or max_line_w > box_w) and font_size > 4:
        font_size -= 1
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        text_h = line_h * num_lines
        max_line_w = max(draw.textlength(line, font=font) for line in lines)

    # Compute vertical start to center block inside rect
    y_start = y_min + (box_h - text_h) / 2

    # Draw lines
    for i, line in enumerate(lines):
        line_w = draw.textlength(line, font=font)
        x_text = x_min + (box_w - line_w) / 2  # horizontal center
        y_text = y_start + i * line_h
        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))  # white glyph

    return glyph_img

def render_glyph_from_crop(base_image: Image.Image, rect: np.ndarray, padding: int = 2) -> Image.Image:
    """
    Create a full-size RGBA glyph image where the text region is cropped from the original,
    and the rest is transparent.
    
    Args:
        base_image: PIL.Image RGB
        rect: 4x2 polygon rectangle [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
        padding: extra pixels around rect
    
    Returns:
        Full-size RGBA image with cropped text in place
    """
    # Full-size transparent image
    w, h = base_image.size
    glyph_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    
    # Crop the text region from original
    x_min = max(0, int(rect[:,0].min()) - padding)
    y_min = max(0, int(rect[:,1].min()) - padding)
    x_max = min(w, int(rect[:,0].max()) + padding)
    y_max = min(h, int(rect[:,1].max()) + padding)
    
    cropped_text = base_image.crop((x_min, y_min, x_max, y_max))
    
    # Paste cropped region into the same coordinates in the transparent image
    glyph_img.paste(cropped_text, (x_min, y_min))
    
    return glyph_img


def save_flux_inputs(
    base_image_bytes: bytes,
    ocr_para_trans_results: List[Union[dict, list]],
    output_dir: str = OUTPUT_DIR,
    mask_padding: int = MASK_PADDING
) -> Dict[str, Any]:
    """
    Produce and save glyph (cropped from original), mask, and prompt files per entry for FLUX-Text.
    The glyphs are full-size images with text region from the original, transparent elsewhere.

    Args:
        base_image_bytes: original image content as bytes
        ocr_para_trans_results: list of dicts or lists with OCR results
        output_dir: directory to save masks, glyphs, prompts
        mask_padding: extra pixels around bounding box for mask

    Returns:
        Dict with metadata and paths
    """
    ensure_dir(output_dir)
    base_image = Image.open(io.BytesIO(base_image_bytes)).convert("RGB")
    w, h = base_image.size

    metadata = []
    for idx, entry in enumerate(ocr_para_trans_results):
        try:
            rect = entry_to_bbox(entry)
        except Exception as e:
            print(f"Skipping entry {idx}: {e}")
            continue

        merged_text = get_merged_text(entry)  # may be empty

        # Filenames
        prefix = f"entry_{idx:03d}"
        mask_fname = os.path.join(output_dir, f"{prefix}_mask.png")
        glyph_fname = os.path.join(output_dir, f"{prefix}_glyph.png")
        prompt_fname = os.path.join(output_dir, f"{prefix}_prompt.txt")

        # 1) Mask image
        mask_img = create_mask_image_for_box(base_image, rect, padding=mask_padding)
        mask_img.save(mask_fname)

        # 2) Glyph image (cropped text from original)
        glyph_img = render_glyph_from_crop(base_image, rect, padding=mask_padding)
        glyph_img.save(glyph_fname)

        # 3) Prompt
        if merged_text:
            prompt = (
                f"Replace the text in the highlighted area with: \"{merged_text}\". "
                f"Preserve the original style, color and background texture as much as possible."
            )
        else:
            prompt = "Remove the text in the highlighted area and reconstruct background (no visible text)."

        with open(prompt_fname, "w", encoding="utf-8") as pf:
            pf.write(prompt)

        metadata.append({
            "index": idx,
            "bbox_rect": rect.tolist(),
            "mask_path": mask_fname,
            "glyph_path": glyph_fname,
            "prompt_path": prompt_fname,
            "merged_text": merged_text
        })

    # Save metadata JSON
    meta_path = os.path.join(output_dir, "flux_inputs_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)

    return {"metadata": metadata, "meta_path": meta_path}
