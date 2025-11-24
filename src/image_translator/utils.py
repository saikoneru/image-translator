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

def proportional_word_split(text: str, num_parts: int, proportions: np.ndarray) -> List[str]:
    """
    Split text into parts based on proportions, ensuring each part gets at least some text.
    """
    words = text.split()
    if not words:
        return [""] * num_parts

    if len(words) == 1:
        # Single word goes to the largest proportion
        result = [""] * num_parts
        result[np.argmax(proportions)] = text
        return result

    # Normalize proportions
    proportions = np.array(proportions, dtype=float)
    proportions = proportions / proportions.sum()

    # Calculate word counts per part (ensure at least 1 word per part if possible)
    total_words = len(words)
    word_counts = np.round(proportions * total_words).astype(int)

    # Ensure each part gets at least 1 word if we have enough words
    if total_words >= num_parts:
        word_counts = np.maximum(word_counts, 1)

    # Adjust to match total
    diff = total_words - word_counts.sum()
    if diff > 0:
        # Add remaining words to parts with highest proportions
        indices = np.argsort(proportions)[::-1]
        for i in range(diff):
            word_counts[indices[i % len(indices)]] += 1
    elif diff < 0:
        # Remove excess words from parts with lowest proportions
        indices = np.argsort(proportions)
        for i in range(abs(diff)):
            idx = indices[i % len(indices)]
            if word_counts[idx] > 1:
                word_counts[idx] -= 1

    # Split words according to counts
    result = []
    word_idx = 0
    for count in word_counts:
        if count > 0:
            result.append(" ".join(words[word_idx:word_idx + count]))
            word_idx += count
        else:
            result.append("")

    return result

def proportional_word_split(text: str, num_parts: int, proportions: np.ndarray) -> List[str]:
    """
    Split text into parts based on proportions, ensuring each part gets at least some text.
    """
    words = text.split()
    if not words:
        return [""] * num_parts

    if len(words) == 1:
        # Single word goes to the largest proportion
        result = [""] * num_parts
        result[np.argmax(proportions)] = text
        return result

    # Normalize proportions
    proportions = np.array(proportions, dtype=float)
    proportions = proportions / proportions.sum()

    # Calculate word counts per part (ensure at least 1 word per part if possible)
    total_words = len(words)
    word_counts = np.round(proportions * total_words).astype(int)

    # Ensure each part gets at least 1 word if we have enough words
    if total_words >= num_parts:
        word_counts = np.maximum(word_counts, 1)

    # Adjust to match total
    diff = total_words - word_counts.sum()
    if diff > 0:
        # Add remaining words to parts with highest proportions
        indices = np.argsort(proportions)[::-1]
        for i in range(diff):
            word_counts[indices[i % len(indices)]] += 1
    elif diff < 0:
        # Remove excess words from parts with lowest proportions
        indices = np.argsort(proportions)
        for i in range(abs(diff)):
            idx = indices[i % len(indices)]
            if word_counts[idx] > 1:
                word_counts[idx] -= 1

    # Split words according to counts
    result = []
    word_idx = 0
    for count in word_counts:
        if count > 0:
            result.append(" ".join(words[word_idx:word_idx + count]))
            word_idx += count
        else:
            result.append("")

    return result


def merge_translations_smart(merged_ocr_results: List[Dict], ocr_line_results: List[Dict]) -> List[Dict]:
    """
    Geometry-aware translation merger with word-level splitting.
    Properly handles line structure to prevent text concatenation.
    Marks lines that are part of merged groups so they can be drawn together.

    Args:
        merged_ocr_results: List of merged groups with translations
            [{"group_indices": [0, 1], "merged_text": "translated text"}, ...]
        ocr_line_results: List of OCR line results (modified in place)
            [{"text": "...", "box": [[x,y], ...], "merged_text": ""}, ...]
    """
    # Initialize all lines with empty merged_text and group info
    for line in ocr_line_results:
        line["merged_text"] = ""
        line["merge_group_id"] = None  # Track which merge group this belongs to
        line["is_primary_line"] = False  # Only primary line shows text

    for group_idx, entry in enumerate(merged_ocr_results):
        group_ids = entry.get("group_indices", [])
        merged_text = entry.get("merged_text", "").strip()

        if not group_ids:
            continue

        # Mark all lines in this group
        for gid in group_ids:
            if gid < len(ocr_line_results):
                ocr_line_results[gid]["merge_group_id"] = group_idx

        # If no translation, use original text
        if not merged_text:
            for gid in group_ids:
                if gid < len(ocr_line_results):
                    ocr_line_results[gid]["merged_text"] = ocr_line_results[gid].get("text", "")
                    ocr_line_results[gid]["is_primary_line"] = True
            continue

        # Single line → direct assign
        if len(group_ids) == 1:
            gid = group_ids[0]
            if gid < len(ocr_line_results):
                ocr_line_results[gid]["merged_text"] = merged_text
                ocr_line_results[gid]["is_primary_line"] = True
            continue

        # Multi-line group → split based on geometry
        widths, heights = [], []
        for gid in group_ids:
            if gid >= len(ocr_line_results):
                widths.append(1)
                heights.append(1)
                continue

            box = np.array(ocr_line_results[gid].get("box", []))
            if len(box) == 0:
                widths.append(1)
                heights.append(1)
                continue

            x_min, y_min = box[:, 0].min(), box[:, 1].min()
            x_max, y_max = box[:, 0].max(), box[:, 1].max()
            widths.append(max(1, x_max - x_min))
            heights.append(max(1, y_max - y_min))

        widths = np.array(widths)
        heights = np.array(heights)

        # Determine orientation
        orientation = "vertical" if heights.sum() > widths.sum() * 1.5 else "horizontal"
        proportions = heights if orientation == "vertical" else widths

        # Split text proportionally
        split_texts = proportional_word_split(merged_text, len(group_ids), proportions)

        # Assign to OCR lines and mark primary
        for idx, (gid, part) in enumerate(zip(group_ids, split_texts)):
            if gid < len(ocr_line_results):
                ocr_line_results[gid]["merged_text"] = part.strip()
                ocr_line_results[gid]["is_primary_line"] = True  # All split parts are primary

    return ocr_line_results


def extract_text_color_from_diff(poly, orig_cv, inpaint_cv):
    """Extract text color by comparing original and inpainted regions."""
    def ensure_3ch_uint8(img):
        if img is None:
            raise ValueError("Image is None")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.astype(np.uint8)

    orig_cv = ensure_3ch_uint8(orig_cv)
    inpaint_cv = ensure_3ch_uint8(inpaint_cv)

    if orig_cv.shape[:2] != inpaint_cv.shape[:2]:
        inpaint_cv = cv2.resize(inpaint_cv, (orig_cv.shape[1], orig_cv.shape[0]))

    mask = np.zeros(orig_cv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

    orig_region = cv2.bitwise_and(orig_cv, orig_cv, mask=mask)
    inpaint_region = cv2.bitwise_and(inpaint_cv, inpaint_cv, mask=mask)

    diff = cv2.absdiff(orig_region, inpaint_region)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    text_pixels = orig_region[text_mask == 255].reshape(-1, 3)

    if len(text_pixels) < 10:
        # Fallback: invert background
        mean_bg = np.mean(inpaint_region[mask == 255].reshape(-1, 3), axis=0)
        color_rgb = 255 - mean_bg[::-1]
        return tuple(int(c) for c in np.clip(color_rgb, 0, 255))

    med_bgr = np.median(text_pixels, axis=0)
    color_rgb = med_bgr[::-1]

    bright = np.mean(color_rgb)
    if bright < 80:
        color_rgb[:] = 0
    if bright > 200:
        color_rgb[:] = 255

    return tuple(int(c) for c in np.clip(color_rgb, 0, 255))


def draw_paragraphs_polys(image, paragraphs, orig_image, padding=4, font_min=8):
    """
    Draw translated text into original OCR line polygons with proper per-line alignment.
    Lines that are part of the same merge group use the same font size.

    Args:
        image: Inpainted image to draw on
        paragraphs: List of paragraphs, each containing lines
            [[{"text": "...", "merged_text": "...", "box": [[x,y], ...],
               "merge_group_id": int, "is_primary_line": bool}, ...], ...]
        orig_image: Original image for color extraction
        padding: Horizontal padding for text placement
        font_min: Minimum font size
    """
    has_alpha = image.mode == "RGBA"
    draw = ImageDraw.Draw(image)

    # Prepare OpenCV for color extraction
    orig_cv = cv2.cvtColor(np.array(orig_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    inpaint_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Font setup
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        test_font = ImageFont.truetype(font_path, 12)
    except:
        font_path = None

    def make_font(size):
        size = max(font_min, int(size))
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                pass
        return ImageFont.load_default()

    # Process each paragraph
    for para_idx, para in enumerate(paragraphs):
        if not para:
            continue

        # Group lines by merge_group_id
        merge_groups = {}  # {group_id: [line_indices]}
        standalone_lines = []  # Line indices not in any group

        for line_idx, line in enumerate(para):
            group_id = line.get("merge_group_id")
            if group_id is not None:
                if group_id not in merge_groups:
                    merge_groups[group_id] = []
                merge_groups[group_id].append(line_idx)
            else:
                standalone_lines.append(line_idx)

        # Calculate paragraph bounds for alignment
        para_boxes = [np.array(line["box"]) for line in para if line.get("box")]
        if not para_boxes:
            continue

        para_left = min(box[:, 0].min() for box in para_boxes)
        para_right = max(box[:, 0].max() for box in para_boxes)
        para_width = para_right - para_left

        # Process merge groups (lines that were merged together)
        for group_id, line_indices in merge_groups.items():
            group_lines = [para[idx] for idx in line_indices]

            # Calculate combined bounds for the group
            group_boxes = [np.array(line["box"]) for line in group_lines if line.get("box")]
            if not group_boxes:
                continue

            # Get all lines with actual text to draw
            lines_to_draw = [line for line in group_lines if line.get("is_primary_line") and line.get("merged_text", "").strip()]

            if not lines_to_draw:
                continue

            # Calculate average line height for the group
            line_heights = [box[:, 1].max() - box[:, 1].min() for box in group_boxes]
            avg_h = np.mean(line_heights)

            # Start with a font size based on average height
            group_font_size = max(int(avg_h * 0.75), font_min)
            group_font = make_font(group_font_size)

            # Check if all lines fit with this font size
            max_overflow = 0
            for line in lines_to_draw:
                text = line.get("merged_text", "").strip()
                if not text:
                    continue
                box = np.array(line["box"])
                line_width = box[:, 0].max() - box[:, 0].min()
                text_w = draw.textlength(text, font=group_font)
                overflow = text_w - line_width + (2 * padding)
                max_overflow = max(max_overflow, overflow)

            # Reduce font size if needed
            while group_font_size > font_min and max_overflow > 0:
                group_font_size -= 1
                group_font = make_font(group_font_size)
                max_overflow = 0
                for line in lines_to_draw:
                    text = line.get("merged_text", "").strip()
                    if not text:
                        continue
                    box = np.array(line["box"])
                    line_width = box[:, 0].max() - box[:, 0].min()
                    text_w = draw.textlength(text, font=group_font)
                    overflow = text_w - line_width + (2 * padding)
                    max_overflow = max(max_overflow, overflow)

            # Get text height with final font
            sample_bbox = draw.textbbox((0, 0), "Ayg", font=group_font)
            text_h = sample_bbox[3] - sample_bbox[1]

            # Determine group alignment strategy
            # Check if all lines in group are similarly aligned
            group_gaps_left = []
            group_gaps_right = []
            for line in lines_to_draw:
                box = np.array(line["box"])
                x_left = box[:, 0].min()
                x_right = box[:, 0].max()
                group_gaps_left.append(x_left - para_left)
                group_gaps_right.append(para_right - x_right)

            avg_gap_left = np.mean(group_gaps_left)
            avg_gap_right = np.mean(group_gaps_right)

            # Determine group alignment
            if abs(avg_gap_left - avg_gap_right) < 15:
                group_alignment = "center"
            elif avg_gap_left < avg_gap_right:
                group_alignment = "left"
            else:
                group_alignment = "right"

            # Draw each line in the group with the same font and alignment
            for line in lines_to_draw:
                text = line.get("merged_text", "").strip()
                if not text:
                    continue

                box = np.array(line["box"], dtype=np.float32)
                if len(box) < 3:
                    continue

                x_left = box[:, 0].min()
                x_right = box[:, 0].max()
                y_top = box[:, 1].min()
                y_bottom = box[:, 1].max()
                line_width = x_right - x_left
                line_height = y_bottom - y_top

                text_w = draw.textlength(text, font=group_font)

                # Apply consistent group alignment
                if group_alignment == "center":
                    # Center within the entire paragraph width for consistency
                    x = para_left + (para_width - text_w) / 2
                elif group_alignment == "left":
                    # Align to paragraph left edge with padding
                    x = para_left + padding
                else:  # right
                    # Align to paragraph right edge with padding
                    x = para_right - text_w - padding

                # Vertical centering
                y = y_top + (line_height - text_h) / 2

                # Extract text color
                color = extract_text_color_from_diff(box.astype(np.int32), orig_cv, inpaint_cv)
                if has_alpha:
                    color = (*color, 255)

                # Draw the text
                draw.text((x, y), text, font=group_font, fill=color)

        # Process standalone lines (not part of any merge group)
        for line_idx in standalone_lines:
            line = para[line_idx]

            # Skip lines without primary text
            if not line.get("is_primary_line"):
                continue

            text = line.get("merged_text", "").strip()
            if not text:
                text = line.get("text", "").strip()
            if not text:
                continue

            box = np.array(line["box"], dtype=np.float32)
            if len(box) < 3:
                continue

            x_left = box[:, 0].min()
            x_right = box[:, 0].max()
            y_top = box[:, 1].min()
            y_bottom = box[:, 1].max()
            line_width = x_right - x_left
            line_height = y_bottom - y_top

            # Calculate font size for this line
            line_font_size = max(int(line_height * 0.75), font_min)
            line_font = make_font(line_font_size)

            # Scale down if needed
            text_w = draw.textlength(text, font=line_font)
            while line_font_size > font_min and text_w > line_width - (2 * padding):
                line_font_size -= 1
                line_font = make_font(line_font_size)
                text_w = draw.textlength(text, font=line_font)

            # Get text height
            sample_bbox = draw.textbbox((0, 0), "Ayg", font=line_font)
            text_h = sample_bbox[3] - sample_bbox[1]

            # Determine horizontal alignment
            gap_left = x_left - para_left
            gap_right = para_right - x_right

            if abs(gap_left - gap_right) < 15:
                x = para_left + (para_width - text_w) / 2
            elif gap_left < gap_right:
                x = x_left + padding
            else:
                x = x_right - text_w - padding

            # Vertical centering
            y = y_top + (line_height - text_h) / 2

            # Extract text color
            color = extract_text_color_from_diff(box.astype(np.int32), orig_cv, inpaint_cv)
            if has_alpha:
                color = (*color, 255)

            # Draw the text
            draw.text((x, y), text, font=line_font, fill=color)

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
