import gradio as gr
from image_translator.pipeline import process_image

lang_map = {
    "English": "en", "Chinese": "zh", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Spanish": "es"
}

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),
        gr.Dropdown(list(lang_map.keys()), label="Source Language"),
        gr.Dropdown(list(lang_map.keys()), label="Target Language")
    ],
    outputs=[
        gr.Image(type="pil", label="Masked Image"),
        gr.Image(type="pil", label="Translated Image")
    ],
    title="Image Translator",
    description="Performs OCR, inpainting, translation, and overlay."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)