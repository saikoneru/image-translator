FROM paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY src/image_translator/ocr/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m pip install paddleocr

# Copy the project
COPY . .

# Default port (can be overridden by docker-compose)
ENV WORKER_PORT=8001
ENV PYTHONUNBUFFERED=1

# Expose that port dynamically
EXPOSE ${WORKER_PORT}

# Run your app using the environment variable
CMD ["sh", "-c", "python  src/image_translator/ocr/ocr_worker.py --host 0.0.0.0 --port ${WORKER_PORT}"]