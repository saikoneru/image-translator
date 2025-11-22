import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

def process_image(image_path, src_lang, tgt_lang):
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://127.0.0.1:5001/translate/auto",
            files={"file": (image_path, f, "image/*")},
            data={"src_lang": src_lang, "tgt_lang": tgt_lang}
        )

    # Raw PNG bytes (NOT base64, NOT JSON)
    img_bytes = resp.content

    return Image.open(BytesIO(img_bytes))

langs = ["English", "Chinese", "Japanese", "French", "German"]

gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath"), gr.Dropdown(langs), gr.Dropdown(langs)],
    outputs=gr.Image(type="pil"),
    title="Local Multi-Worker Image Translator",
    description="Each component runs as a separate FastAPI worker."
).launch(server_name="0.0.0.0", server_port=7860)
