FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system utilities and multi-threading runtimes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies matrix
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime tracking data folders
RUN mkdir -p /app/utils /app/data

# Copy app code components natively
COPY app/config.py ./
COPY app/database.py ./
COPY app/inference.py ./
COPY app/cluster_handling.py ./
COPY app/app.py ./
COPY app/cluster_refinement.py ./
COPY app/pipeline.py ./
COPY app/cluster_engine.py ./
COPY app/custom_clustering.py ./
# copy this test file
COPY test/pdw_labeled.csv test/pdw_labeled.csv
COPY test/gof_sband_cta_1_PDWTable.csv test/gof_sband_cta_1_PDWTable.csv
# FIX: Copy utils from its new location outside of the app directory
COPY utils/create_features.py ./utils/

# Copy the ONNX model from your root model directory
COPY model/transformer_model_known.onnx model/transformer_model_known.onnx

CMD ["python", "app.py"]
