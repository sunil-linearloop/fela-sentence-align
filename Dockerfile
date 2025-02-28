# Use NVIDIA CUDA base image with Python support
FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# Update and upgrade the system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y swig build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first and remove googletrans
COPY requirements.txt /app/requirements.txt
RUN grep -v "googletrans" /app/requirements.txt > /app/requirements_no_googletrans.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements_no_googletrans.txt && \
    pip install runpod==1.6.0

# Install compatible versions for googletrans
RUN pip install httpcore==0.9.1 httpx==0.13.3 && \
    pip install googletrans==4.0.0rc1

# Verify installation
RUN python -c "from googletrans import Translator; print('Googletrans successfully imported')"

# Copy application files
COPY app/ /app/app
COPY bertalign/ /app/bertalign
COPY handler.py /app/handler.py

# Create an rp_handler.py file that imports your handler for RunPod compatibility
RUN echo "import runpod; from handler import handler; runpod.serverless.start({'handler': handler})" > /app/rp_handler.py

# Set environment variables for RunPod
ENV RUNPOD_SERVERLESS_FUNCTION_PORT=8080
ENV RUNPOD_WEBHOOK_GET_BODY=true
ENV PYTHONPATH=/app

# Expose port 8080
EXPOSE 8080

# Run the handler
CMD ["python", "-u", "/app/rp_handler.py"]