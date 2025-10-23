# OCR Worker

The OCR (Optical Character Recognition) Worker detects and extracts text from images using PaddleOCR, providing text content and bounding box coordinates for each detected text region.

## 📋 Overview

This worker is the first step in the translation pipeline. It uses PaddleOCR to analyze the input image and returns:
- Detected text strings
- Bounding box coordinates (polygons) for each text region
- Results are filtered by confidence threshold (>0.8)

## 🔧 Implementation Details

- **OCR Engine**: PaddleOCR
- **Model Management**: Thread-based model manager for concurrent requests
- **API Framework**: FastAPI with async support
- **GPU Support**: Enabled by default (set `device="gpu"`)
- **Text Orientation**: Enabled for better accuracy

## 📥 Input Format

### API Endpoint

```
POST /ocr
Content-Type: multipart/form-data
```

### Request Format

Upload an image file as form data:

```bash
curl -X POST http://localhost:8001/ocr \
  -F "file=@image.jpg"
```

### Python Example

```python
import requests

# Upload image file
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8001/ocr", files=files)

result = response.json()
print(result)
```

### Internal Processing Format

The worker internally uses:

```python
input_data = {
    "image_path": "/tmp/tmp_image.png"
}
```

## 📤 Output Format

### Response Schema

```json
{
  "results": [
    {
      "text": "Detected text",
      "box": [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
      ]
    }
  ]
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | List of detected text regions |
| `results[].text` | string | Detected text content |
| `results[].box` | array | Polygon coordinates (4 points) |

### Bounding Box Format

Bounding boxes are polygons with 4 corner points:

```
[x1,y1] -------- [x2,y2]
  |                 |
  |    TEXT         |
  |                 |
[x4,y4] -------- [x3,y3]
```

Points are in image coordinates where (0,0) is top-left.

### Error Response

```json
{
  "error": "Error message description"
}
```

HTTP Status: 500

## 🐳 Docker Usage

### Build the Container

```bash
docker build -t ocr-worker ./workers/ocr
```

### Run Standalone

```bash
docker run -p 8001:8001 ocr-worker
```

### Run with Docker Compose

```yaml
ocr-worker:
  build: ./workers/ocr
  ports:
    - "8001:8001"
  environment:
    - WORKER_HOST=0.0.0.0
    - WORKER_PORT=8001
    - LOG_LEVEL=debug
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_HOST` | `0.0.0.0` | Server host |
| `WORKER_PORT` | `8001` | Server port |
| `LOG_LEVEL` | `debug` | Logging level |

## 🛠️ Configuration

### PaddleOCR Settings

```python
PaddleOCR(
    use_doc_orientation_classify=False,  # Document orientation
    use_doc_unwarping=False,             # Document unwarping
    use_textline_orientation=True,       # Text line orientation
    device="gpu",                        # Use GPU
    lang="en",                           # Language
)
```

### Confidence Threshold

Results are filtered to only include detections with confidence > 0.8:

```python
if conf > 0.8:
    results.append({"text": text, "box": poly})
```

To adjust, modify this threshold in the `process()` method.

### Thread Workers

Control number of concurrent OCR workers:

```python
ocr_worker = PaddleOCRWorker(num_workers=1, lang="en")
```

Increase `num_workers` for higher throughput (requires more GPU memory).

## 🔍 Code Structure

### BaseWorker Implementation

```python
class PaddleOCRWorker(BaseWorker):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR and return standardized results"""
        # 1. Validate input
        if not self.validate_input(input_data):
            raise ValueError("Invalid input format for OCRWorker")
        
        # 2. Run OCR
        image_path = input_data["image_path"]
        raw_results = self.ocr_manager.predict(image_path)[0]
        
        # 3. Format results
        results = []
        for poly, text, conf in zip(...):
            if conf > 0.8:
                results.append({"text": text, "box": poly})
        
        # 4. Validate output
        output = {"results": results}
        if not self.validate_output(output):
            raise ValueError("Invalid output format from OCRWorker")
        
        return output
```

### Model Manager

The `PaddleOCRModelManager` manages threaded OCR workers:

- Creates model instances in separate threads
- Queues requests for processing
- Returns results asynchronously
- Handles errors gracefully

### FastAPI Endpoint

```python
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile):
    # 1. Read uploaded file
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 2. Save temporarily
    tmp_path = "/tmp/tmp_image.png"
    img.save(tmp_path)
    
    # 3. Process with worker
    input_data = {"image_path": tmp_path}
    output_data = ocr_worker.process(input_data)
    
    # 4. Convert numpy arrays to lists for JSON
    for r in output_data["results"]:
        if isinstance(r.get("box"), np.ndarray):
            r["box"] = r["box"].tolist()
    
    return JSONResponse(content=output_data)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"
- **Solution**: Reduce `num_workers` or process smaller images
- **Alternative**: Switch to CPU mode: `device="cpu"`

**Issue**: "No text detected"
- **Solution**: 
  - Check image quality and resolution
  - Try different language: `lang="ch"` for Chinese
  - Lower confidence threshold

**Issue**: "Slow processing"
- **Solution**:
  - Ensure GPU is available and properly configured
  - Check `nvidia-smi` for GPU utilization
  - Reduce image resolution before processing

**Issue**: "Import errors"
- **Solution**: Ensure PaddleOCR and dependencies are installed
  ```bash
  pip install paddlepaddle-gpu paddleocr
  ```

## 🔄 Customizing the Worker

### Change Language

```python
ocr_worker = PaddleOCRWorker(num_workers=1, lang="ch")  # Chinese
ocr_worker = PaddleOCRWorker(num_workers=1, lang="fr")  # French
```

### Adjust Confidence Threshold

Modify the filtering logic in `process()`:

```python
for poly, text, conf in zip(...):
    if conf > 0.6:  # Lower threshold
        results.append({"text": text, "box": poly})
```

### Enable Additional Features

```python
def _create_model(self, lang: str):
    return PaddleOCR(
        use_doc_orientation_classify=True,   # Enable orientation
        use_doc_unwarping=True,              # Enable unwarping
        use_textline_orientation=True,
        device="gpu",
        lang=lang,
    )
```

### Add Post-processing

```python
def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing code ...
    
    # Add custom post-processing
    results = self._filter_results(results)
    results = self._merge_nearby_text(results)
    
    return {"results": results}
```

## 📖 API Documentation

### POST /ocr

Process an image and extract text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file

**Response:**
- Success: 200 OK with results
- Error: 500 with error message

**Example:**
```bash
curl -X POST http://localhost:8001/ocr \
  -F "file=@test.jpg" \
  | jq '.results'
```

## 📚 Dependencies

```txt
paddlepaddle-gpu==2.6.0
paddleocr==2.7.0
fastapi==0.104.1
uvicorn==0.24.0
pillow==10.1.0
numpy==1.24.3
```

For CPU-only:
```txt
paddlepaddle==2.6.0  # Instead of paddlepaddle-gpu
```
