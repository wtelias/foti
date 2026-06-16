// Splash boot loop: wait for the local Foti daemon to answer /health, then
// hand the whole webview over to its web UI. All product UI lives in the
// backend's single-page app, so once we navigate there this shell is done.
const { invoke } = window.__TAURI__.core;

const POLL_MS = 500;
const TIMEOUT_MS = 120000; // backends loading ML models can take a while on first run

async function boot() {
  const statusEl = document.getElementById("status");
  const spinnerEl = document.getElementById("spinner");
  const deadline = Date.now() + TIMEOUT_MS;

  while (Date.now() < deadline) {
    let ready = false;
    try {
      ready = await invoke("backend_ready");
    } catch (_) {
      ready = false;
    }
    if (ready) {
      const url = await invoke("backend_url");
      window.location.replace(url);
      return;
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }

  // Timed out — most likely the daemon isn't installed or failed to start.
  if (spinnerEl) spinnerEl.style.display = "none";
  if (statusEl) {
    statusEl.innerHTML =
      "Couldn't reach the Foti engine.<br/>Make sure the Foti backend is installed, then reopen the app.";
    statusEl.classList.add("error");
  }
}

window.addEventListener("DOMContentLoaded", boot);
