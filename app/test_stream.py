import gradio as gr
import httpx
import json
import base64
import io
from sseclient import SSEClient
import aiohttp
import asyncio

API_URL = "http://127.0.0.1:5001/translate/pptx/sse"  # adjust to your FastAPI URL

async def translate_and_stream(file, src_lang, tgt_lang, translate_master, translate_images):
    """Connect to SSE endpoint and stream translation progress + PDFs."""
    if not file:
        yield "No PPTX uploaded.", None
        return

    # Prepare form data exactly like your production code
    form_data = aiohttp.FormData()
    form_data.add_field(
        'file',
        open(file.name, 'rb'),
        filename='presentation.pptx',
        content_type='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )
    form_data.add_field('src_lang', src_lang)
    form_data.add_field('tgt_lang', tgt_lang)
    form_data.add_field('translate_master', str(translate_master).lower())
    form_data.add_field('translate_images', str(translate_images).lower())

    headers = {"Accept": "text/event-stream"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, data=form_data, headers=headers, timeout=aiohttp.ClientTimeout(total=3600)) as response:
                if response.status != 200:
                    text = await response.text()
                    yield f"❌ API Error: {text}", None
                    return

                buffer = ""
                event_type = None

                async for chunk in response.content.iter_any():
                    decoded = chunk.decode("utf-8")
                    buffer += decoded

                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        event_lines = event_block.split("\n")

                        for line in event_lines:
                            if line.startswith("event:"):
                                event_type = line.split(":", 1)[1].strip()
                            elif line.startswith("data:"):
                                data_json = line.split(":", 1)[1].strip()

                                try:
                                    data = json.loads(data_json)
                                except Exception:
                                    yield f"⚠️ Invalid JSON: {data_json}", None
                                    continue

                                # Handle events
                                if event_type == "slide" and "pdf_base64" in data:
                                    pdf_bytes = base64.b64decode(data["pdf_base64"])
                                    slide = data.get("slide")
                                    total = data.get("total")
                                    yield f"✅ Slide {slide}/{total} translated.", ("slide.pdf", pdf_bytes)
                                elif event_type == "progress":
                                    yield f"⏳ Progress: Slide {data.get('slide')}/{data.get('total')}", None
                                elif event_type == "status":
                                    yield f"ℹ️ Status: {data.get('status')}", None
                                elif event_type == "error":
                                    yield f"❌ Error: {data.get('error')}", None
                                elif event_type == "complete":
                                    yield "✅ Translation complete!", None
                                else:
                                    yield f"📩 {event_type}: {data}", None
    except Exception as e:
        yield f"❌ Exception: {str(e)}", None


with gr.Blocks(title="🧠 PPTX SSE Translation Tester (aiohttp)") as demo:
    gr.Markdown("### Test `/translate/pptx/sse` streaming translation endpoint")

    with gr.Row():
        file = gr.File(label="Upload PPTX File")
        src_lang = gr.Textbox(label="Source Language", value="en")
        tgt_lang = gr.Textbox(label="Target Language", value="fr")
    with gr.Row():
        translate_master = gr.Checkbox(label="Translate Master Slides", value=True)
        translate_images = gr.Checkbox(label="Translate Images", value=True)

    status_box = gr.Textbox(label="Event Log / Status", interactive=False)
    pdf_out = gr.File(label="Latest Slide PDF")

    run_btn = gr.Button("🚀 Start Streaming Translation")

    run_btn.click(
        translate_and_stream,
        inputs=[file, src_lang, tgt_lang, translate_master, translate_images],
        outputs=[status_box, pdf_out]
    )

demo.queue()  # Enable async generator streaming
demo.launch(server_name="0.0.0.0", server_port=7860)

