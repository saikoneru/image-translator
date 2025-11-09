import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

def process_image(image, src_lang, tgt_lang):
    files = {"file": open(image, "rb")}
    data = {"src_lang": src_lang, "tgt_lang": tgt_lang}
    resp = requests.post("http://127.0.0.1:5001//translate/image", files=files, data=data)
    img_b64 = resp.json()["image_base64"]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes))
    return img

langs = ["English", "Chinese", "Japanese", "French", "German"]

gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath"), gr.Dropdown(langs), gr.Dropdown(langs)],
    outputs=gr.Image(type="pil"),
    title="Local Multi-Worker Image Translator",
    description="Each component runs as a separate FastAPI worker."
).launch(server_name="0.0.0.0", server_port=7868)
