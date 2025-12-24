# Your custom Dockerfile for in-orbit inference
FROM tm2space/aicube-base:latest

# # Only add packages NOT in base image (check library docs first!)
# # The base image already includes: PyTorch, TensorFlow, GDAL, Rasterio, etc.
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf ~/.cache/pip

# Copy source files directly to /workspace/
COPY src/ /workspace/

# Models and config at workspace level
COPY models/ /workspace/models/
COPY config/ /workspace/config/

WORKDIR /workspace

# Entry point for your application
CMD ["python3", "main.py"]