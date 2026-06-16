# Foti Desktop

A native desktop shell for **Foti**, built with [Tauri](https://tauri.app).

It is intentionally thin: on launch it starts a local `foti-backend serve` on a
private loopback port (no auth — it only ever listens on `127.0.0.1`), waits for
the daemon's `/health` to come up, then points the webview at it. The entire
product UI is Foti's existing single-page web app, so the desktop build reuses
it verbatim — there is no second UI to keep in sync. When the window closes, the
backend it started is shut down with it.

The heavy Python/ML dependencies (PyTorch, OpenCLIP, InsightFace) are **not**
bundled — they are multi-gigabyte and GPU-specific. The desktop app discovers an
installed `foti-backend` on `PATH` (e.g. via `pipx install "foti-backend[gpu]"`).

## Prerequisites

- Rust + Cargo
- Tauri CLI v2: `cargo install tauri-cli --version '^2.0.0' --locked`
- Linux system deps: `webkit2gtk-4.1`, `libappindicator`, `librsvg` (see the
  [Tauri prerequisites](https://tauri.app/start/prerequisites/))
- A `foti-backend` reachable on `PATH` at runtime

## Develop

```bash
cd ui/desktop
cargo tauri dev
```

During development, point the shell at a backend that isn't on `PATH` with
`FOTI_BACKEND_CMD` (it is split on spaces — program first, then leading args):

```bash
# use the project venv's console script
FOTI_BACKEND_CMD="$PWD/../../backend/.venv/bin/foti-backend" cargo tauri dev
```

## Build (AppImage / deb / rpm)

```bash
cd ui/desktop
cargo tauri build
```

Bundles are written to `src-tauri/target/release/bundle/`.

## Runtime configuration

| Env var             | Effect                                                              |
| ------------------- | ------------------------------------------------------------------ |
| `FOTI_BACKEND_CMD`  | Command used to launch the backend (default: `foti-backend`).       |
| `FOTI_PORT`         | Force a fixed backend port (default: an OS-assigned free port).      |

The shell always starts the backend **without** HTTP Basic auth (it binds
loopback only), so no login prompt appears in the desktop window.
