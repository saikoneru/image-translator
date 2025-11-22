FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
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

# Copy only the worker requirements first
COPY src/image_translator/layout/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch GPU version
RUN pip install --no-cache-dir torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Clone Hi-SAM and install dependencies
RUN git clone https://github.com/ymy-k/Hi-SAM.git /app/Hi-SAM
RUN pip install --no-cache-dir -r /app/Hi-SAM/requirements.txt

# Now copy the rest of the project
COPY . .

# ✅ Create pretrained checkpoints directory and copy models
# FIND MODELS HERE https://github.com/ymy-k/Hi-SAM


# Add Hi-SAM to PYTHONPATH
ENV PYTHONPATH=/app/Hi-SAM:$PYTHONPATH

# Environment variables
ENV WORKER_PORT=8005
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE ${WORKER_PORT}

# ✅ Run worker, using checkpoint path from /app/pretrained_checkpoints
CMD ["sh", "-c", "python src/image_translator/layout/layout_worker.py --host 0.0.0.0 --port ${WORKER_PORT} --checkpoint pretrained_checkpoint/hi_sam_l.pth"]
