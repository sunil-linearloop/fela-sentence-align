# Use RunPod base image with CUDA 12.4.1
FROM runpod/base:0.6.2-cuda12.4.1

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# Update and upgrade the system packages, install Python 3.10
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv swig build-essential git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy requirements file first and remove googletrans
COPY requirements.txt /app/requirements.txt
RUN grep -v "googletrans" /app/requirements.txt > /app/requirements_no_googletrans.txt

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements_no_googletrans.txt && \
    python -m pip install runpod

# Install Hugging Face transfer optimization for faster downloads
RUN python -m pip install "huggingface_hub[hf_transfer]"

# Install compatible versions for googletrans
RUN python -m pip install httpcore==0.9.1 httpx==0.13.3 && \
    python -m pip install googletrans==4.0.0rc1

# Copy application files
COPY app/ /app/app
COPY bertalign/ /app/bertalign
COPY download_model.py /app/download_model.py
COPY handler.py /app/handler.py

# Create model cache directory
RUN mkdir -p /app/models

# Enable HF transfer for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download the LaBSE model
RUN python /app/download_model.py --model "sentence-transformers/LaBSE" --cache-dir "/app/models"

# Set permissions
RUN chmod -R 777 /app/models

# Verify installation
RUN python -c "from googletrans import Translator; print('Googletrans successfully imported')"

# Set environment variables for RunPod serverless
ENV PYTHONPATH=/app
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV RUNPOD_WORKER_PORT=8000
ENV TRANSFORMERS_CACHE="/app/models"
ENV HF_HOME="/app/models"

# Expose the port that the worker runs on
EXPOSE 8000

# Start the worker
CMD ["python", "-u", "/app/handler.py"]