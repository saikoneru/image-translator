import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile
import os

SERVER_URL = "http://127.0.0.1:8080/process"
REQUEST_TIMEOUT = 300  # seconds


def process_document(file, src_lang, tgt_lang):
    """
    Translate image or multi-page PDF with live page previews.
    """
    if file is None:
        yield None, None
        return

    filename = file.name if hasattr(file, "name") else str(file)
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pdf":
        pdf_bytes = file.read() if hasattr(file, "read") else open(file, "rb").read()
        pages = convert_from_bytes(pdf_bytes, dpi=200)
        num_pages = len(pages)

        translated_images = []

        for i, page in enumerate(pages):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                page.save(tmp_img.name, "PNG")

                files = {"file": open(tmp_img.name, "rb")}
                data = {"src_lang": src_lang, "tgt_lang": tgt_lang}

                try:
                    resp = requests.post(
                        SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT
                    )
                    if resp.status_code == 200:
                        img_b64 = resp.json()["image_base64"]
                        img_bytes = base64.b64decode(img_b64)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        translated_images.append(img)

                        # ✅ keep showing the image until next one is ready
                        yield img, None
                    else:
                        print(f"⚠️ Failed page {i+1}: {resp.status_code}")

                except Exception as e:
                    print(f"❌ Error on page {i+1}: {e}")
                finally:
                    os.unlink(tmp_img.name)

        if translated_images:
            out_path = tempfile.mktemp(suffix="_translated.pdf")
            translated_images[0].save(out_path, save_all=True, append_images=translated_images[1:])
            yield translated_images[-1], out_path
        else:
            yield None, None

    else:
        # Single image case
        files = {"file": open(file, "rb")}
        data = {"src_lang": src_lang, "tgt_lang": tgt_lang}

        try:
            resp = requests.post(
                SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                img_b64 = resp.json()["image_base64"]
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                yield img, None
            else:
                yield None, None
        except Exception as e:
            print("Error:", e)
            yield None, None


# --- UI ---
langs = ["English", "Chinese", "Japanese", "French", "German"]

with gr.Blocks(title="Local Translator") as demo:
    gr.Markdown("### 🌐 Translate Images & PDFs — Live Page Preview")

    with gr.Row():
        file_input = gr.File(label="📄 Upload Image or PDF", file_types=["image", ".pdf"])
        src = gr.Dropdown(langs, label="Source Language", value="English")
        tgt = gr.Dropdown(langs, label="Target Language", value="Chinese")

    img_output = gr.Image(label="🔍 Live Page Preview", type="pil")
    pdf_output = gr.File(label="📘 Download Translated PDF")

    translate_btn = gr.Button("🚀 Translate")

    translate_btn.click(
        fn=process_document,
        inputs=[file_input, src, tgt],
        outputs=[img_output, pdf_output],
        queue=True,
    )

# ✅ just enable queue (no args)
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7868)

