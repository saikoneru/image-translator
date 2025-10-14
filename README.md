# Image Translation Pipeline

This repository contains a modular **image translation pipeline** that takes images with text (e.g., street signs) and outputs a translated version of the image with original text replaced while preserving style and layout. The pipeline is designed as **independent workers**, each responsible for a specific task. This modular design allows easy scaling, avoids environment conflicts, and provides flexibility for upgrading individual components.

---
## Usage

The main interface (`pipeline.py`) acts as a **gateway** that orchestrates all workers:

1. Upload an image.
2. Specify source and target languages.
3. The gateway calls the workers in sequence and returns the final translated image.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/saikoneru/image-translator.git
cd image-translator

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -e .
```

## Overview

The pipeline consists of the following **workers**:

### 1. OCR Worker
- **Purpose:** Extracts text from the input image along with precise bounding boxes.
- **Input:** Image file.
- **Output:** List of OCR results. Each result contains:
  - `text`: Recognized text string.
  - `box`: Polygon coordinates representing the text region.

### 2. Segmenter Worker
- **Purpose:** Groups OCR results into meaningful phrases or text lines.
- **Why needed:** OCR often returns text at the word or character level. The segmenter identifies which words belong together (e.g., "No" + "Parking" → one phrase).
- **Input:** OCR results and the original image.
- **Output:** Grouped OCR entries with merged text and group indices.

### 3. Inpainting Worker
- **Purpose:** Removes the original text from the image while keeping the background intact.
- **Input:** Original image and bounding boxes of text.
- **Output:** Masked or inpainted image, ready for rendering translated text.

### 4. Translation Worker
- **Purpose:** Translates merged OCR text from a source language to a target language.
- **Input:** List of merged OCR phrases, source language, target language.
- **Output:** Translated text phrases.


## Utility Components (Heuristic-based)

In addition to the workers, the pipeline contains utility components that use heuristics to improve translation rendering.

### 5. Translation Substitution
- **Purpose:** Translates merged OCR text from a source language to a target language.
- **Input:** List of merged OCR phrases, source language, target language.
- **Output:** Translated text phrases.

### 6. Drawing
- **Purpose:** Renders the translated text back onto the inpainted image.
- **Features:**
  - Geometry-aware: Matches the size and position of original text polygons.
  - Auto font sizing and color estimation to preserve the style of the original text.
- **Input:** Inpainted image, translated OCR results, original image for reference.
- **Output:** Final translated image with text visually aligned and styled.


---

---

## Pipeline Flow

```text
Input Image
    │
    ▼
[OCR Worker] --> OCR Results (text + polygon boxes)
    │
    ▼
[Segmenter Worker] --> Merged OCR Groups
    │
    ▼
[Inpainting Worker] --> Image with text removed
    │
    ▼
[Translation Substitution] --> Translated phrases
    │
    ▼
[Translation Merging] --> OCR-level translations distributed
    │
    ▼
[Drawing Worker] --> Final translated image



