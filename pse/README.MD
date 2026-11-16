# Image Translation Post-Editing Tool

Build a post-editing tool for translated images. The tool starts from OCR boxes
and their translated text. The user can change the text, and **you have to create
a manual drawing interface** that redraws the translated text onto the image so
it looks like the source image.

## Input

An image plus two JSON files with the same structure, one entry per OCR box
(same order, same index):

- `ocr.json`: source text
- `translated.json`: English translation

Each entry has the text, a four-point box (`[x, y]` pixel coordinates), and an
OCR confidence score:

```json
{
  "results": [
    {
      "text": "WIE WOLLEN WIR MORGEN LEBEN?",
      "box": [[54, 423], [830, 423], [830, 467], [54, 467]],
      "confidence": 0.99
    }
  ]
}
```

Boxes can be rotated (see the "PODIUMSDISKUSSION" box) and OCR can be wrong or
noisy, so some boxes will need to be fixed or deleted.

## Task

- Initialise one editable element per OCR box
- Let the user edit the translated text
- Create a **manual drawing interface** on the post-edited text
- Export the final image

## Possible Features

- Add, delete, merge, and split boxes
- Move, resize, and rotate boxes
- Font, size, colour, weight, and alignment controls
- Colour picker that samples from the source image
- Fit text to the box (auto-shrink, line breaks)
- Original vs. edited comparison view
- Highlight low-confidence boxes
- Undo/redo
- Estimate style (font size, colour) from the box and image region
- Inpainting for textured backgrounds
- Export edited boxes and text as JSON

## Cloning

```bash
git clone https://github.com/saikoneru/image-translator
cd image-translator
git checkout pse
cd pse
```
