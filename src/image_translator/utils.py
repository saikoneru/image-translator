import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import re
from typing import List, Dict
import io

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
    Draw paragraph-level OCR results with consistent alignment (left/center/right).
    Each paragraph is a list of lines, each line is a dict with:
        - 'text': original text
        - 'merged_text': translated/aligned text
        - 'box': [[x,y], ...]
        - 'words' (optional): list of dicts with 'text' and 'box'

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
        if not para:
            continue

        # --- Compute paragraph bounding box ---
        all_pts = np.vstack([np.array(line["box"]) for line in para])
        x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
        x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()
        box_w, box_h = x_max - x_min, y_max - y_min
        num_lines = len(para)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(BASE_DIR, "fonts", "dejavu-sans.oblique.ttf")

        # --- Estimate vertical spacing from OCR lines ---
        line_tops = [np.min(np.array(line["box"])[:, 1]) for line in para]
        line_bottoms = [np.max(np.array(line["box"])[:, 1]) for line in para]
        avg_line_height = np.mean([b - t for t, b in zip(line_tops, line_bottoms)])
        avg_gap = np.mean(np.diff(sorted(line_tops))) if len(line_tops) > 1 else avg_line_height
        avg_gap = max(avg_gap, avg_line_height * 1.1)

        # --- Font sizing ---
        font_size = max(int((box_h - 2 * padding) / num_lines * 0.85), font_min)
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # --- Fit width if needed ---
        text_widths = [draw.textlength(line.get("merged_text", line["text"]), font=font) for line in para]
        while max(text_widths) > (box_w - 2 * padding) and font_size > font_min:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            text_widths = [draw.textlength(line.get("merged_text", line["text"]), font=font) for line in para]

        # --- Detect paragraph alignment (global, not per line) ---
        left_edges = [np.min(np.array(line["box"])[:, 0]) for line in para]
        right_edges = [np.max(np.array(line["box"])[:, 0]) for line in para]
        left_std, right_std = np.std(left_edges), np.std(right_edges)

        if left_std < right_std * 0.6:
            paragraph_align = "left"
        elif right_std < left_std * 0.6:
            paragraph_align = "right"
        else:
            paragraph_align = "center"

        # --- Vertical start (centered within box) ---
        total_text_height = avg_gap * num_lines
        y_start = y_min + (box_h - total_text_height) / 2 + padding

        # --- Draw each line ---
        for i, line in enumerate(para):
            text_to_draw = line.get("merged_text", line["text"]).strip()
            if not text_to_draw:
                continue

            line_box = np.array(line["box"])
            line_left = np.min(line_box[:, 0])
            line_right = np.max(line_box[:, 0])

            line_w = draw.textlength(text_to_draw, font=font)

            # --- Preserve indentation from first word (if available) ---
            indent_offset = 0
            if "words" in line and len(line["words"]) > 1:
                first_word = line["words"][0]
                fw_left = np.min(np.array(first_word["box"])[:, 0])
                indent_offset = max(0, fw_left - line_left)
                indent_offset = min(indent_offset, box_w * 0.2)

            # --- X position (based on global alignment only) ---
            if paragraph_align == "left":
                x_text = x_min + indent_offset + padding
            elif paragraph_align == "right":
                x_text = x_max - line_w - padding
            else:
                x_text = (x_min + x_max - line_w) / 2

            # --- Y position ---
            y_text = y_start + i * avg_gap

            # --- Extract color ---
            poly_pts = np.array(line["box"], dtype=np.int32)
            color = extract_text_color_from_diff(poly_pts, orig_img_cv, img_cv)

            draw.text((x_text, y_text), text_to_draw, font=font, fill=color)

    return image

def merge_translations_enhanced(merged_ocr_results, ocr_line_results):
    """
    Enhanced translation merger with multiple strategies:
    1. Character-level alignment for better accuracy
    2. Punctuation-aware splitting
    3. Special handling for short translations
    4. Fallback strategies for edge cases
    """
    for entry in merged_ocr_results:
        group_ids = entry["group_indices"]
        translated_text = entry["merged_text"].strip()

        if not translated_text or not group_ids:
            continue

        # Single element - direct assignment
        if len(group_ids) == 1:
            ocr_line_results[group_ids[0]]["merged_text"] = translated_text
            continue

        # Choose strategy based on translation characteristics
        strategy = determine_strategy(translated_text, group_ids, ocr_line_results)

        if strategy == "punctuation":
            distribute_by_punctuation(translated_text, group_ids, ocr_line_results)
        elif strategy == "character_ratio":
            distribute_by_character_ratio(translated_text, group_ids, ocr_line_results)
        elif strategy == "word_geometric":
            distribute_by_word_geometry(translated_text, group_ids, ocr_line_results)
        else:  # fallback
            distribute_by_word_geometry(translated_text, group_ids, ocr_line_results)

    return ocr_line_results


def determine_strategy(translated_text, group_ids, ocr_line_results):
    """
    Intelligently choose distribution strategy based on content
    """
    # Count natural break points (punctuation)
    punctuation_count = sum(1 for c in translated_text if c in '.,;:!?—')

    # If punctuation matches or is close to group count, use punctuation strategy
    if abs(punctuation_count - len(group_ids)) <= 1 and punctuation_count > 0:
        return "punctuation"

    # Get original text to compare character counts
    original_chars = sum(len(ocr_line_results[gid].get("text", "")) for gid in group_ids)
    translated_chars = len(translated_text)

    # If similar character counts, use character ratio (works well for similar languages)
    ratio = translated_chars / max(original_chars, 1)
    if 0.5 <= ratio <= 2.0:  # Not too different in length
        return "character_ratio"

    # Default: word-based geometric distribution
    return "word_geometric"


def distribute_by_punctuation(translated_text, group_ids, ocr_line_results):
    """
    Split translation by punctuation marks and assign to elements
    """
    # Split by sentence-ending punctuation while keeping delimiters
    segments = re.split(r'([.!?。！？]+\s*)', translated_text)

    # Merge delimiters back with their segments
    merged_segments = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            merged_segments.append(segments[i] + segments[i + 1])
        elif segments[i].strip():
            merged_segments.append(segments[i])

    # If no segments or mismatch, fall back
    if not merged_segments or len(merged_segments) > len(group_ids):
        return distribute_by_word_geometry(translated_text, group_ids, ocr_line_results)

    # Assign segments to elements
    for i, gid in enumerate(group_ids):
        if i < len(merged_segments):
            ocr_line_results[gid]["merged_text"] = merged_segments[i].strip()
        else:
            ocr_line_results[gid]["merged_text"] = ""


def distribute_by_character_ratio(translated_text, group_ids, ocr_line_results):
    """
    Distribute based on original character count ratios
    More accurate than word count for similar-length languages
    """
    # Get original character counts
    original_lengths = []
    for gid in group_ids:
        original_text = ocr_line_results[gid].get("text", "")
        original_lengths.append(len(original_text.strip()))

    total_original = sum(original_lengths)
    if total_original == 0:
        return distribute_by_word_geometry(translated_text, group_ids, ocr_line_results)

    # Calculate character allocation for each element
    translated_chars = len(translated_text)
    char_allocations = []

    for length in original_lengths:
        ratio = length / total_original
        char_allocations.append(max(1, int(ratio * translated_chars)))

    # Adjust for rounding errors
    total_allocated = sum(char_allocations)
    diff = translated_chars - total_allocated
    if diff != 0:
        # Add/remove from largest allocation
        max_idx = np.argmax(char_allocations)
        char_allocations[max_idx] += diff

    # Split translation by character count, respecting word boundaries
    char_idx = 0
    words = translated_text.split()
    word_idx = 0

    for gid, target_chars in zip(group_ids, char_allocations):
        chunk_words = []
        chunk_length = 0

        # Collect words until we reach target character count
        while word_idx < len(words) and chunk_length < target_chars:
            word = words[word_idx]
            chunk_words.append(word)
            chunk_length += len(word) + 1  # +1 for space
            word_idx += 1

        ocr_line_results[gid]["merged_text"] = " ".join(chunk_words)

    # Assign any remaining words to last element
    if word_idx < len(words):
        last_gid = group_ids[-1]
        remaining = " ".join(words[word_idx:])
        if ocr_line_results[last_gid]["merged_text"]:
            ocr_line_results[last_gid]["merged_text"] += " " + remaining
        else:
            ocr_line_results[last_gid]["merged_text"] = remaining


def distribute_by_word_geometry(translated_text, group_ids, ocr_line_results):
    """
    Your original approach - distribute by geometric proportions
    Enhanced with better edge case handling
    """
    # Gather box geometry
    boxes = []
    for gid in group_ids:
        poly = np.array(ocr_line_results[gid].get("box", []))
        if len(poly) == 0:
            boxes.append((1, 1))
            continue
        x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
        x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
        boxes.append((x_max - x_min, y_max - y_min))

    widths = np.array([w for w, _ in boxes])
    heights = np.array([h for _, h in boxes])

    # Determine orientation
    total_width = widths.sum()
    total_height = heights.sum()
    orientation = "vertical" if total_height > total_width * 1.5 else "horizontal"

    # Word-level distribution
    trans_words = translated_text.split()
    total_words = len(trans_words)

    if total_words == 0:
        return

    if total_words == 1:
        # One-word translation: assign to dominant box
        dominant_idx = np.argmax(heights if orientation == "vertical" else widths)
        for j, gid in enumerate(group_ids):
            ocr_line_results[gid]["merged_text"] = translated_text if j == dominant_idx else ""
        return

    # Multi-word: proportional allocation
    geom_sizes = heights if orientation == "vertical" else widths
    geom_sizes = np.maximum(geom_sizes, 1e-3)
    proportions = geom_sizes / geom_sizes.sum()

    # Calculate word counts per element (ensure at least 1 word per element if possible)
    num_elements = len(group_ids)
    word_counts = np.zeros(num_elements, dtype=int)

    # First pass: proportional distribution
    for i in range(num_elements):
        word_counts[i] = max(1, int(proportions[i] * total_words))

    # Adjust for rounding
    allocated = word_counts.sum()
    if allocated < total_words:
        remaining = total_words - allocated
        top_indices = np.argsort(proportions)[-remaining:]
        for idx in top_indices:
            word_counts[idx] += 1
    elif allocated > total_words:
        excess = allocated - total_words
        for _ in range(excess):
            for idx in np.argsort(proportions):
                if word_counts[idx] > 1:
                    word_counts[idx] -= 1
                    break

    # Assign words
    word_idx = 0
    for gid, count in zip(group_ids, word_counts):
        chunk = " ".join(trans_words[word_idx:word_idx + count])
        word_idx += count
        ocr_line_results[gid]["merged_text"] = chunk


# ============================================================================
# Alternative: Alignment-based approach using dynamic programming
# ============================================================================

def merge_translations_alignment(merged_ocr_results, ocr_line_results):
    """
    Use character-level alignment (similar to sequence alignment in bioinformatics)
    Best for languages with very different character systems (e.g., English → Japanese)
    """
    for entry in merged_ocr_results:
        group_ids = entry["group_indices"]
        translated_text = entry["merged_text"].strip()

        if not translated_text or not group_ids:
            continue

        if len(group_ids) == 1:
            ocr_line_results[group_ids[0]]["merged_text"] = translated_text
            continue

        # Get original texts
        original_texts = [ocr_line_results[gid].get("text", "") for gid in group_ids]
        original_lengths = [len(t) for t in original_texts]
        total_original = sum(original_lengths)

        if total_original == 0:
            # Equal distribution
            words = translated_text.split()
            words_per_element = len(words) // len(group_ids)
            remainder = len(words) % len(group_ids)

            word_idx = 0
            for i, gid in enumerate(group_ids):
                count = words_per_element + (1 if i < remainder else 0)
                chunk = " ".join(words[word_idx:word_idx + count])
                ocr_line_results[gid]["merged_text"] = chunk
                word_idx += count
            continue

        # Use original length ratios to split translation
        translated_words = translated_text.split()
        total_words = len(translated_words)

        word_allocations = []
        for length in original_lengths:
            ratio = length / total_original
            word_allocations.append(max(1, int(ratio * total_words)))

        # Adjust allocations
        total_allocated = sum(word_allocations)
        diff = total_words - total_allocated

        if diff > 0:
            # Distribute extra words to largest elements
            for _ in range(diff):
                max_idx = np.argmax(original_lengths)
                word_allocations[max_idx] += 1
                original_lengths[max_idx] = 0  # Prevent re-selection
        elif diff < 0:
            # Remove words from smallest allocations
            for _ in range(-diff):
                for i in np.argsort(word_allocations):
                    if word_allocations[i] > 1:
                        word_allocations[i] -= 1
                        break

        # Assign words
        word_idx = 0
        for gid, count in zip(group_ids, word_allocations):
            chunk = " ".join(translated_words[word_idx:word_idx + count])
            ocr_line_results[gid]["merged_text"] = chunk
            word_idx += count

def draw_each_box_and_save(img_bytes, ocr_para_trans_results, output_dir="outputs"):
    """
    Draw bounding boxes for OCR/translated text.
    - Handles both single dict entries and list-of-word entries.
    - Saves individual cropped images, final overlay image, and combined merged text.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_cv = np.array(image)

    merged_texts = []
    final_img = img_cv.copy()

    for i, entry in enumerate(ocr_para_trans_results):
        # --- Determine if entry is a list (paragraph) or dict (single box)
        if isinstance(entry, list):
            # Merge all word boxes into one bounding polygon
            all_pts = np.vstack([np.array(word["box"]) for word in entry if "box" in word])
            box = all_pts.astype(np.int32)
            merged_text = " ".join(
                [w.get("merged_text", w.get("text", "")) for w in entry if w.get("text")]
            ).strip()
        elif isinstance(entry, dict):
            box = np.array(entry.get("box", []), dtype=np.int32)
            merged_text = entry.get("merged_text", entry.get("text", "")).strip()
        else:
            print(f"⚠️ Skipping invalid entry at index {i}: {type(entry)}")
            continue

        if box.size == 0:
            continue

        merged_texts.append(merged_text)

        # Compute bounding rectangle for all points (ensures one clean box)
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
        rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

        # --- Individual image with one box ---
        img_copy = img_cv.copy()
        overlay = img_copy.copy()
        cv2.polylines(img_copy, [rect], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.fillPoly(overlay, [rect], (255, 0, 0))
        # cv2.addWeighted(overlay, 0.15, img_copy, 0.85, 0, img_copy)

        out_path = os.path.join(output_dir, f"box_{i+1:03d}.png")
        Image.fromarray(img_copy).save(out_path)

        # --- Add to combined overlay image ---
        cv2.polylines(final_img, [rect], isClosed=True, color=(255, 0, 0), thickness=3)

    # --- Save combined overlay image ---
    final_path = os.path.join(output_dir, "final.png")
    Image.fromarray(final_img).save(final_path)

    # --- Save all merged texts ---
    combined_text = " ".join(merged_texts).strip()
    text_path = os.path.join(output_dir, "merged_texts.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(combined_text + "\n")

    print(f"✅ Saved {len(merged_texts)} boxes, final.png, and merged_texts.txt in '{output_dir}'")
    return combined_text, final_path