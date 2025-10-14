# Requirements:
# pip install paddleocr paddlepaddle pillow opencv-python numpy
import math
import statistics
from collections import defaultdict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from paddleocr import PaddleOCR
import cv2
from PIL import Image, ImageDraw, ImageFont
import ast
import copy
import gradio as gr
from paddle_ocr import PaddleOCRModelManager, create_model
import functools
from segment_vlm import segment_boxes, merge_boxes_from_groups
import re
from PIL import Image
from inpainter import inpaint_with_boxes
from detect_font import predict_fonts_from_boxes, draw_ocr_polys
from download_font import get_font
from seedx_translate import generate, merge_translations
import time
from qwen_vl_omniseed import ocr_image
import numpy as np


ocr = PaddleOCRModelManager(1, functools.partial(create_model, lang="en"))

# Example usage:
def process_image(image_path, src_lang, tgt_lang):
    image = Image.open(image_path).convert("RGB")

    start_time = time.time()
    results = ocr.predict(image_path)
    end_time = time.time()
    print(f"OCR Time: {end_time - start_time} seconds")

    start_time = time.time()
    ocr_line_results = ocr_image(image)
    end_time = time.time()
    print(f"OCR Time: {end_time - start_time} seconds")
    res_dict = results[0]

    # Parse results
    ocr_results = []
    rec_texts_filt = []

    for box, text, conf in zip(res_dict["rec_polys"], res_dict["rec_texts"], res_dict["rec_scores"]):
        if conf > 0.8:
            ocr_results.append({"box": box, "text": text, "conf": conf})
            rec_texts_filt.append(text)


    start_time = time.time()
    ocr_lines_dict = [{"id": str(i), "text": line} for i, line in enumerate(ocr_line_results)]
    segmented_ocr_ids = segment_boxes(ocr_lines_dict, image_path)
    segmented_ocr_ids = ast.literal_eval( re.sub(r'^```json\s*|\s*```$', '', segmented_ocr_ids))
    merged_ocr_results = merge_boxes_from_groups(ocr_line_results, segmented_ocr_ids)
    end_time = time.time()
    print(f"Segmenting Time: {end_time - start_time} seconds")

    masked_image = inpaint_with_boxes(image, [res["box"] for res in ocr_results])
    start_time = time.time()
    #masked_image = inpaint_with_boxes(image, [res["box"] for res in ocr_results])
    end_time = time.time()
    print(f"Inpainting Time: {end_time - start_time} seconds")


    #merged_ocr_results_with_fonts = predict_fonts_from_boxes(image_path, merged_ocr_results)
    # ocr_results_with_fonts = predict_fonts_from_boxes(image_path, ocr_results)
    # for entry in ocr_results_with_fonts:
    #     get_font(entry['font'])

    start_time = time.time()
    trans_texts = generate([res["merged_text"] for res in merged_ocr_results], src_lang, tgt_lang)
    for i, entry in enumerate(merged_ocr_results):
        entry["orig_text"] = entry["merged_text"]
        if "-" not in entry["orig_text"]:
            entry["merged_text"] = trans_texts[i].replace("-", " ")
        else:
            entry["merged_text"] = trans_texts[i]

    ocr_line_results = merge_translations(merged_ocr_results, ocr_line_results)
    for entry in ocr_line_results:
        box = entry["box"]
        poly = np.array([
            [box[0], box[1]],  # top-left
            [box[2], box[1]],  # top-right
            [box[2], box[3]],  # bottom-right
            [box[0], box[3]]   # bottom-left
        ], dtype=np.float32)
        entry["box"] = poly
    end_time = time.time()
    print(f"Translation Time: {end_time - start_time} seconds")
    
    trans_image = draw_ocr_polys(masked_image.copy(), ocr_line_results, image)

    return masked_image, trans_image



if __name__ == "__main__":
    # Gradio interface
    lang_map = {"English": "en", "Chinese": "zh",  "French": "fr", "German": "de", "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt", "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar", "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl", "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms", "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl", "Romanian": "ro", "Turkish": "tr"}
    demo = gr.Interface(
        fn=process_image,
        inputs=[gr.Image(type="filepath", label="Street Sign"), gr.Dropdown(choices=list(lang_map.keys()), label="Source Language"), gr.Dropdown(choices=list(lang_map.keys()), label="Target Language")],
        outputs=[gr.Image(type="pil", label="OCR with bounding boxes"), gr.Image(type="pil", label="Translated Street Sign")],
        title="OCR Segmentation & Translation Demo",
        description="Upload an image. OCR is performed, text is grouped, translated, and aligned, and the result is shown with bounding boxes."
    )
    demo.launch(server_name="0.0.0.0", server_port=8080)
