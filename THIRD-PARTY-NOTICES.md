# Third-Party Notices

Foti's own source code is licensed **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)).

Foti **bundles no third-party code or model weights** in this repository. The
libraries below are pulled in by your package manager at install time, and the
ML models are downloaded into a local cache on first run. You are responsible
for honouring their licenses for your use case.

This file is provided in good faith for convenience. It is **not legal advice**
and may lag upstream license changes — verify against each project before
relying on it commercially.

## ⚠️ The one that matters for commercial use

**InsightFace pretrained models (e.g. `buffalo_l`)** are released for
**non-commercial / academic research use only.** The InsightFace *Python
package* is MIT-licensed, but the downloaded face-recognition **model weights**
are not free for commercial use. If you run Foti commercially, you must obtain a
commercial license from InsightFace or substitute a face model whose license
permits your use. Foti loads the model by name at runtime, so swapping it is a
configuration change, not a code change.

## Python dependencies (installed via pip/uv)

| Package | License |
|---|---|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic, pydantic-settings | MIT |
| sqlite-vec | Apache-2.0 / MIT |
| pillow | MIT-CMU (HPND) |
| pillow-heif | BSD-3-Clause (wraps libheif — LGPL-3.0; HEVC patent licensing may apply) |
| exifread | BSD-3-Clause |
| open-clip-torch | MIT (code) |
| torch, torchvision | BSD-3-Clause |
| numpy | BSD-3-Clause |
| typer | MIT |
| rich | MIT |
| watchdog | Apache-2.0 |
| rawpy | MIT (wraps LibRaw — LGPL-2.1 / CDDL-1.0 / commercial) |
| imagehash | BSD-2-Clause |
| insightface | MIT (code) — **see model-weights note above** |
| onnxruntime / onnxruntime-gpu | MIT |
| nvidia-\*-cu12 (gpu extra only) | NVIDIA proprietary EULA (redistributable CUDA runtime components) |

## Models downloaded at runtime

| Model | Use | License of the weights |
|---|---|---|
| OpenCLIP **ViT-L/14** | content + text + similarity search, aesthetic scoring | Open — the OpenAI CLIP weights are MIT; LAION-trained variants are openly licensed. Verify the specific checkpoint you fetch. |
| InsightFace **buffalo_l** | face detection + embedding | **Non-commercial / research only** (see above) |

## Why this is clean to open-source

- None of the above is **redistributed** by this repository — it is fetched by
  the user's own tooling at install/first-run.
- Foti's code is an **independent implementation**: no reverse-engineering of,
  and no code shared with, any commercial product it is compared to.
- The AGPL-3.0 of Foti's code is compatible with the permissive
  (MIT/BSD/Apache) licenses of its code dependencies.
