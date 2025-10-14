import time
from PIL import Image

def process_image(image_path, src_lang, tgt_lang):
    image = Image.open(image_path).convert("RGB")
    return image, image