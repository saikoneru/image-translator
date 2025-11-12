# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including LibreOffice
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
    libreoffice \
    libreoffice-impress \
    libreoffice-writer \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Expose the port for the API gateway
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "image_translator", "--port", "5000", "--host", "0.0.0.0"]
