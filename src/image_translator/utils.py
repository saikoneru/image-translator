import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont

def merge_translations(merged_ocr_results, ocr_line_results):
    """
    Heuristically distribute merged translations back into OCR results.
    Geometry-aware: uses bounding box sizes to allocate translated text.
    Ensures no text loss through careful word distribution.
    """
    for entry in merged_ocr_results:
        group_ids = entry["group_indices"]
        translated_text = entry["merged_text"].strip()
        if not translated_text or not group_ids:
            continue

        # 1️⃣ Single OCR element → direct assignment
        if len(group_ids) == 1:
            ocr_line_results[group_ids[0] - 1]["merged_text"] = translated_text
            continue

        # 2️⃣ Gather box geometry
        boxes = []
        for i in group_ids:
            poly = np.array(ocr_line_results[i - 1].get("box", []))
            if len(poly) == 0:
                boxes.append((1, 1))  # Default size if no box
                continue
            x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
            x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
            boxes.append((x_max - x_min, y_max - y_min))  # (width, height)

        widths = np.array([w for w, _ in boxes])
        heights = np.array([h for _, h in boxes])

        # Determine orientation — horizontal vs vertical text line
        total_width = widths.sum()
        total_height = heights.sum()
        orientation = "vertical" if total_height > total_width * 1.5 else "horizontal"

        # 3️⃣ Word-level distribution
        trans_words = translated_text.split()
        total_words = len(trans_words)

        if total_words == 0:
            continue

        if total_words <= 1:
            # 🧠 One-word translation: assign to dominant (widest/tallest) box
            dominant_idx = (
                np.argmax(heights) if orientation == "vertical" else np.argmax(widths)
            )
            for j, gid in enumerate(group_ids):
                ocr_line_results[gid - 1]["merged_text"] = (
                    translated_text if j == dominant_idx else ""
                )
            continue

        # 4️⃣ Multi-word translation: proportional allocation by geometry
        geom_sizes = heights if orientation == "vertical" else widths
        geom_sizes = np.maximum(geom_sizes, 1e-3)
        proportions = geom_sizes / geom_sizes.sum()

        # Calculate word counts per element
        num_elements = len(group_ids)
        word_counts = np.zeros(num_elements, dtype=int)
        
        # Distribute words proportionally
        for i in range(num_elements):
            word_counts[i] = max(1, int(proportions[i] * total_words))
        
        # Ensure all words are allocated (fix rounding errors)
        allocated = word_counts.sum()
        if allocated < total_words:
            # Distribute remaining words to elements with highest proportions
            remaining = total_words - allocated
            top_indices = np.argsort(proportions)[-remaining:]
            for idx in top_indices:
                word_counts[idx] += 1
        elif allocated > total_words:
            # Remove excess words from elements with lowest proportions
            excess = allocated - total_words
            for _ in range(excess):
                # Remove 1 word from the element with lowest proportion that has > 1 word
                for idx in np.argsort(proportions):
                    if word_counts[idx] > 1:
                        word_counts[idx] -= 1
                        break

        # 5️⃣ Assign words to OCR elements in order
        word_idx = 0
        for gid, count in zip(group_ids, word_counts):
            chunk = " ".join(trans_words[word_idx:word_idx + count])
            word_idx += count
            ocr_line_results[gid - 1]["merged_text"] = chunk

    return ocr_line_results

def extract_text_color_from_diff(poly, orig_cv, inpaint_cv):
    """
    Estimate text color by comparing original and inpainted images safely.
    Args:
        poly: polygon (Nx2 numpy array)
        orig_cv: original image (BGR or RGB)
        inpaint_cv: inpainted image (BGR or RGB)
    Returns:
        (r, g, b) tuple of estimated text color
    """

    # --- Ensure 3-channel uint8 ---
    def ensure_3ch_uint8(img):
        if img is None:
            raise ValueError("Image is None")
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img

    orig_cv = ensure_3ch_uint8(orig_cv)
    inpaint_cv = ensure_3ch_uint8(inpaint_cv)

    # --- Ensure same size ---
    if orig_cv.shape[:2] != inpaint_cv.shape[:2]:
        inpaint_cv = cv2.resize(inpaint_cv, (orig_cv.shape[1], orig_cv.shape[0]))

    # --- Safe mask creation ---
    mask = np.zeros(orig_cv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

    # --- Apply safely ---
    orig_region = cv2.bitwise_and(orig_cv, orig_cv, mask=mask)
    inpaint_region = cv2.bitwise_and(inpaint_cv, inpaint_cv, mask=mask)

    # --- Difference detection ---
    diff = cv2.absdiff(orig_region, inpaint_region)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate changed pixels (likely text)
    _, text_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # --- Extract text pixels ---
    text_pixels = orig_region[text_mask == 255].reshape(-1, 3)
    if len(text_pixels) < 10:
        mean_bg = cv2.mean(inpaint_region, mask=mask)[:3]
        mean_bg_rgb = np.array(mean_bg[::-1])
        text_color_rgb = 255 - mean_bg_rgb
        return tuple(int(c) for c in text_color_rgb)

    # --- Use median color for stability ---
    median_bgr = np.median(text_pixels, axis=0)
    color_rgb = np.array(median_bgr[::-1])  # convert to RGB

    # --- Snap near-black or near-white ---
    brightness = np.mean(color_rgb)
    if brightness < 80:
        # Snap to full black if close
        color_rgb[:] = 0
    elif brightness > 200:
        # Snap to full white if close
        color_rgb[:] = 255

    return tuple(int(c) for c in np.clip(color_rgb, 0, 255))

def draw_ocr_polys(image, ocr_results, orig_image, padding=2, font_min=5):
    """
    Draw OCR results on an image using polygon boxes and auto color/font sizing.
    Compatible with Pillow ≥10 (uses textlength instead of deprecated textsize).

    Args:
        image: PIL.Image (RGB)
        ocr_results: list of dicts with keys:
            - "text": string
            - "poly": list of (x, y) coordinates
            - "font": path to font file (optional)
        padding: inner margin inside polygon box
        font_min: minimum font size allowed
    Returns:
        PIL.Image with text drawn
    """
    draw = ImageDraw.Draw(image)
    img_cv = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    orig_img_cv = np.array(orig_image.convert("RGB"))[:, :, ::-1].copy()

    for result in ocr_results:
        text = str(result.get("merged_text", "")).strip()
        if not text:
            continue

        poly = np.array(result.get("box"), dtype=np.float32)
        font_path = result.get("font", None)

        # Compute bounding box of polygon
        x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
        x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
        box_w, box_h = x_max - x_min, y_max - y_min

        color = extract_text_color_from_diff(poly, orig_img_cv, img_cv)

        # Split text into lines if needed
        lines = text.split("\n")
        rows = len(lines)

        # Initial font size estimation
        #font_size = max(font_min, int(box_h / max(rows, 1) * 0.8))
        font_path = "fonts/dejavu-sans.oblique.ttf"
        font_size = int(min(box_h * 0.9, box_w / max(len(text), 1) * 1.8))
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # Measure text using new Pillow methods

        text_w = max(draw.textlength(line, font=font) for line in lines)
        text_h = font_size * rows

        # Adjust font size to fit width
        while (text_w > box_w - 2 * padding or text_h > box_h - 2 * padding) and font_size > font_min:
            font_size -= 1
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
            text_w = max(draw.textlength(line, font=font) for line in lines)
            text_h = font_size * rows

        # Compute centered position
        x_text = x_min + (box_w - text_w) / 2
        y_text = y_min + (box_h - text_h) / 2

        # Draw text (multi-line supported)
        y_offset = 0
        for line in lines:
            line_w = draw.textlength(line, font=font)
            draw.text((x_text + (text_w - line_w) / 2, y_text + y_offset),
                      line, font=font, fill=color)
            y_offset += font_size
            
    return image