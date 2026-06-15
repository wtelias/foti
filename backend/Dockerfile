# Foti backend — CPU image. Works on any machine; no GPU required.
# For NVIDIA GPU face acceleration, see docs/DEPLOY.md (install the [gpu]
# extra on a CUDA-12 base image and pass --gpus all).
FROM python:3.12-slim

# Runtime libraries: libheif (HEIC/HEIF), libraw (RAW previews), libgl/libglib
# (OpenCV, pulled in by insightface). build-essential + cmake for packages that
# compile a native extension at install time (insightface).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libheif1 libraw20 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# CPU ONNX Runtime + Foti. torch CPU wheels are large — expect a chunky image.
RUN pip install --no-cache-dir ".[cpu,raw]"

# Catalog, thumbnails and the model cache live here — mount a volume to persist.
ENV FOTI_DATA_DIR=/data \
    FOTI_HOST=0.0.0.0 \
    FOTI_PORT=7777
VOLUME ["/data"]
EXPOSE 7777

# Mount your photos read-only at /photos, then `foti-backend scan /photos`.
ENTRYPOINT ["foti-backend"]
CMD ["serve"]
