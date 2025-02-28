# Use CUDA base image since Bertalign can use GPU
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
COPY setup.sh /workspace/
RUN chmod +x /workspace/setup.sh && /workspace/setup.sh

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Copy application code
COPY . /workspace/

# Install the Bertalign package
RUN pip install -e .

# Start the handler
CMD ["python", "handler.py"] 