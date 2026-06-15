# Deploying Foti

Foti is a local-first daemon. By default it binds to `127.0.0.1:7777` and stores
everything under `~/.local/share/foti/`. This guide covers GPU acceleration,
exposing it safely, and running it as a service.

## Configuration

All settings take an environment variable prefixed `FOTI_`:

| Variable | Default | Meaning |
|---|---|---|
| `FOTI_DATA_DIR` | `~/.local/share/foti` | catalog, thumbnails, model cache |
| `FOTI_HOST` | `127.0.0.1` | bind address (`0.0.0.0` to expose) |
| `FOTI_PORT` | `7777` | HTTP port |
| `FOTI_CLIP_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `FOTI_BASIC_USER` / `FOTI_BASIC_PASS` | unset | enable HTTP Basic auth (see below) |

A `.env` file in the working directory is read automatically.

## GPU (NVIDIA, CUDA 12)

Install the GPU extra and put the CUDA-12 runtime libraries on the loader path.
ONNX Runtime's CUDA execution provider links `libcublasLt.so.12` and friends; if
they aren't found, face indexing **silently falls back to CPU** (works, just
much slower).

```bash
pipx install "foti-backend[gpu]"

# Point the loader at the cu12 wheels pip just installed:
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:\
$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:\
$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
```

Confirm the GPU is in use — the face model should report
`providers=['CUDAExecutionProvider', 'CPUExecutionProvider']` in the logs.

## Exposing it (reverse proxy + auth)

Foti has **no built-in authentication beyond optional HTTP Basic.** Never bind
`0.0.0.0` on an untrusted network without putting auth in front of it.

```bash
export FOTI_BASIC_USER=you
export FOTI_BASIC_PASS="$(openssl rand -base64 24)"   # use a long random value
export FOTI_HOST=127.0.0.1                              # keep it local; proxy in front
```

Then terminate TLS and proxy with Caddy/nginx/Tailscale Serve. Basic auth is a
thin guard, not a substitute for TLS — always pair it with HTTPS.

## systemd (user service)

```ini
# ~/.config/systemd/user/foti-backend.service
[Unit]
Description=Foti backend
After=network-online.target

[Service]
Type=simple
Environment=FOTI_DATA_DIR=%h/.local/share/foti
# For GPU, add an Environment=LD_LIBRARY_PATH=... line (see GPU section).
ExecStart=%h/.local/bin/foti-backend serve
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now foti-backend
loginctl enable-linger "$USER"   # keep it running after you log out
```

## Docker

```bash
docker build -t foti backend/
docker run -d --name foti -p 127.0.0.1:7777:7777 \
  -v "$HOME/Pictures:/photos:ro" \
  -v foti-data:/data \
  foti

docker exec foti foti-backend scan /photos
```

The bundled Dockerfile is **CPU-only** (works everywhere). For GPU, base the
image on `nvidia/cuda:12-runtime`, install `".[gpu,raw]"`, and run with
`--gpus all`.
