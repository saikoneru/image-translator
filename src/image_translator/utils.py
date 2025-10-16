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
            ocr_line_results[group_ids[0]]["merged_text"] = translated_text
            continue

        # 2️⃣ Gather box geometry
        boxes = []
        for i in group_ids:
            poly = np.array(ocr_line_results[i].get("box", []))
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
                ocr_line_results[gid]["merged_text"] = (
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
            ocr_line_results[gid]["merged_text"] = chunk

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

def draw_paragraphs_polys(image, paragraphs, orig_image, padding=2, font_min=5):
    """
    Draw paragraph-level OCR results on an image using bounding polygons.
    Each paragraph is a list of lines, each line is a dict with:
        - 'text': original text
        - 'merged_text': translated/aligned text to draw
        - 'box': [[x,y], ...]
    
    Args:
        image: PIL.Image (RGB)
        paragraphs: list of paragraphs, each paragraph = list of lines
        orig_image: original PIL.Image for color extraction
        padding: inner margin inside paragraph box
        font_min: minimum font size allowed
    Returns:
        PIL.Image with text drawn
    """
    draw = ImageDraw.Draw(image)
    img_cv = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    orig_img_cv = np.array(orig_image.convert("RGB"))[:, :, ::-1].copy()

    for para in paragraphs:
        # Merge all line boxes into paragraph bounding box
        all_pts = np.vstack([np.array(line["box"]) for line in para])
        x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
        x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()
        box_w, box_h = x_max - x_min, y_max - y_min

        num_lines = len(para)
        font_path = "fonts/dejavu-sans.oblique.ttf"

        # Initial font size estimation to fit paragraph height
        font_size = max(int((box_h - 2 * padding) / num_lines * 0.8), font_min)
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # Compute text widths for all lines (use merged_text)
        text_widths = [draw.textlength(line.get("merged_text", line["text"]), font=font) for line in para]
        max_text_width = max(text_widths)

        # Reduce font size if any line is too wide
        while max_text_width > box_w - 2 * padding and font_size > font_min:
            font_size -= 1
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
            text_widths = [draw.textlength(line.get("merged_text", line["text"]), font=font) for line in para]
            max_text_width = max(text_widths)

        # Vertical spacing to evenly distribute lines
        total_text_height = font_size * num_lines
        y_start = y_min + (box_h - total_text_height) / 2  # center vertically

        # Draw each line
        for i, line in enumerate(para):
            line_text = line.get("merged_text", line["text"])  # use translated/aligned text
            line_w = draw.textlength(line_text, font=font)
            x_text = x_min + (box_w - line_w) / 2  # center horizontally
            y_text = y_start + i * font_size

            # Extract line color from original image polygon
            poly_pts = np.array(line["box"], dtype=np.int32)
            color = extract_text_color_from_diff(poly_pts, orig_img_cv, img_cv)

            draw.text((x_text, y_text), line_text, font=font, fill=color)


    return image