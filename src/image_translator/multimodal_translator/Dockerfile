FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY src/image_translator/multimodal_translator/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
# Copy the latest project
COPY . .

ARG CACHE_BUSTER=1
RUN git clone https://github.com/saikoneru/OmniFusion 
RUN cp -r OmniFusion/* ./src/image_translator/multimodal_translator/

# Default port (can be overridden by docker-compose)
ENV WORKER_PORT=8004
ENV PYTHONUNBUFFERED=1

# Expose that port dynamically
EXPOSE ${WORKER_PORT}
# Run your app using the environment variable
CMD ["sh", "-c", "python3  src/image_translator/multimodal_translator/translator_worker.py --host 0.0.0.0 --port ${WORKER_PORT}"]
