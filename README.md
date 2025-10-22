# Image Translator

A modular image translation pipeline that translates text inside images while preserving layout and style. Built with Docker for easy deployment and scalability.

## 🌟 Features

- **Modular Architecture**: Independent workers for OCR, segmentation, inpainting, and translation
- **Docker-based**: Each worker runs in its own container, avoiding dependency conflicts
- **Extensible**: Easy to add custom workers or replace existing ones and create new pipelines or frontends using this service
- **Language Support**: Translate between multiple languages: Simply replace with your desired huggingface translation model in translator worker
- **Style Preservation**: Maintains original text styling (font size, color, position)

## 📋 Table of Contents

- [Pipeline Design](#Pipeline-Design)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Workers](#workers)
- [Creating Custom Workers](#creating-custom-workers)

## 🏗️ Pipeline Design

```
Input Image
    │
    ▼
[OCR Worker] ──────────────────> OCR Results (text + bounding boxes)
    │
    ▼
[Layout Worker] ───────────────> Paragraph Boundaries
    │
    ▼
[Line Segmentation] ───────────> Lines within Paragraphs
    (combines OCR + Layout)
    │
    ▼
[Segmenter Worker] ────────────> Translation Units (merged text)
    │
    ▼
[Inpainting Worker] ───────────> Image with Text Removed
    │
    ▼
[Translation] ──────────────────> Translated Text
    │
    ▼
[Drawer Worker] ───────────────> Final Translated Image
```


1. **OCR Worker**: Extracts text and bounding boxes from the image. They are not always line-level with PaddleOCR.
2. **Layout Worker**: Analyzes image layout and identifies paragraph boundaries
3. **Line Segmentation**: Combines OCR results with layout to group entries into lines within each paragraph
4. **Segmenter Worker**: Merges text within paragraphs into optimal translation units
5. **Inpainting Worker**: Removes original text from the image
6. **Translation**: Translates merged text to target language
7. **Drawer Worker**: Renders translated text back onto the image

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+ (for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saikoneru/image-translator.git
cd image-translator
```

2. Build and start all services:
```bash
docker compose up --build
```

3. Access the API:
```bash
curl -X POST http://localhost:5000/process \
  -F "image=@test_image.jpg" \
  -F "source_lang=en" \
  -F "target_lang=de"
```

## 🐳 Running with Docker

### Starting All Services

Every worker will download the models from the internet except the Layout worker models. For this, you need to download the models from [HiSAM](https://github.com/ymy-k/Hi-SAM/tree/main?tab=readme-ov-file#pushpin-checkpoints) and put it in `models/` at the root
directory of the project. Make you sure download the SAM's ViT image encoder mentioned as NOTE in the README.md. After this you can simply build!

```bash
# Build and start all containers
docker compose up --build

# Run in detached mode
docker compose up -d

# View logs
docker compose logs -f
```

### Starting Individual Workers

```bash
# Start only OCR worker
docker compose up ocr-worker

# Start multiple workers
docker compose up ocr-worker segmenter-worker
```

### Stopping Services

```bash
# Stop all containers
docker compose down

# Stop and remove volumes
docker compose down -v
```
## 👷 Workers

Each worker is an independent service with its own Docker container. For detailed information about each worker:

- [OCR Worker](workers/ocr/README.md) - Text detection and recognition
- [Layout Worker](workers/layout/README.md) - Paragraph boundary detection
- [Segmenter Worker](workers/segmenter/README.md) - Line grouping and translation unit creation
- [Inpainting Worker](workers/inpainting/README.md) - Text removal and background reconstruction
- [Drawer Worker](workers/drawer/README.md) - Translated text rendering

### Worker Communication

Workers communicate via REST APIs with standardized JSON formats. Each worker:
- Accepts POST requests with specific input format
- Returns JSON responses with standardized output format

## 🔧 Creating Custom Workers (Not Yet Implemented)

### Step 1: Understand the Worker Interface

All workers must implement the `BaseWorker` interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseWorker(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return output in standardized format.
        
        Args:
            input_data: Dictionary with worker-specific input
            
        Returns:
            Dictionary with standardized output format
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input format"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output format"""
        pass
```

### Step 2: Create Your Worker

Create a new directory in `workers/`:

```bash
workers/
├── your_worker/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   ├── worker.py
│  
```

Example `worker.py`:

```python
from base_worker import BaseWorker
from typing import Dict, Any

class YourWorker(BaseWorker):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Your processing logic here
        result = self.your_custom_logic(input_data)
        
        # Return standardized format
        return {
            "success": True,
            "data": result,
            "metadata": {
                "worker": "your_worker",
                "version": "1.0.0"
            }
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = ["field1", "field2"]
        return all(field in input_data for field in required_fields)
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        return "success" in output_data and "data" in output_data

app = FastAPI(title="Your Custom Worker")
worker = YourWorker()

@app.post("/custom_process")
async def custom_endpoint(field1: UploadFile, field2: int):
    try:
        ### YOUR LOGIC, modify input###
        output_data = worker.process(input_Data)
        return JSONResponse(content=output_data)

    except Exception as e:
        print("Error during custom processing:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8010"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
```

### Step 3: Create Docker Configuration

Example `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Step 4: Update docker compose.yml

Add your worker to `docker compose.yml`:

```yaml
your-worker:
  build: ./workers/your_worker
  ports:
    - "5005:5000"
  environment:
    - WORKER_NAME=your_worker
  volumes:
    - ./workers/your_worker:/app
```

### Step 5: Update Pipeline

Modify `pipeline.py` to include your worker:

```python
# Add worker endpoint
YOUR_WORKER_URL = "http://your-worker:5000/process"

# Add to pipeline
your_result = requests.post(YOUR_WORKER_URL, json=input_data)
```

### Step 6: Test Your Worker

```bash
# Build and start your worker
docker compose up --build your-worker

# Test independently
curl -X POST http://localhost:5005/process \
  -H "Content-Type: application/json" \
  -d '{"field1": "value1", "field2": "value2"}'
```


## 📚 API Documentation

### Main Pipeline Endpoint

**POST** `/process`

Translates text in an image from source to target language.

**Request:**
- `image` (file): Image file to translate
- `source_lang` (string): Source language code (e.g., "en", "es"). See Seedx LLM language codes
- `target_lang` (string): Target language code

**Response:**
```json
{
  "success": true,
  "image_base64": "base64_encoded_image",
  }
}
```

You will get the base64 encoded translated image.

### Individual Worker Endpoints

Each worker exposes their specific endpoint. See individual worker READMEs for the link and their function.

### Local Development (Without Docker)

This is not recommended. The tools used have their own individual environement setups and will lead to conflics. You can look at the Dockerfile for each worker and setup an environment with those specifications.

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run gateway
python -m image_translator --port 5000

# Run individual worker
cd workers/ocr
python app.py
```

## 🙏 Acknowledgments

- PaddleOCR
- Simple Lama
- HiSAM
- huggingface
- python-pptx
- All coding agent providers

## 📧 Contact

For questions or support, please open an issue on GitHub. We are currently developing




