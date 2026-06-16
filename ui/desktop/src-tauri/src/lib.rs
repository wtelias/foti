// Foti desktop shell.
//
// This is a thin Tauri wrapper around Foti's existing web UI. On launch it
// starts a local `foti-backend serve` on a private loopback port (no auth —
// it only ever listens on 127.0.0.1), waits for the daemon's /health to come
// up, then points the webview at it. The whole product UI is the backend's
// own single-page app, so the desktop build reuses it verbatim — no second
// implementation to keep in sync.
//
// The heavy Python/ML dependencies are NOT bundled (multi-GB, GPU-specific);
// `foti-backend` is installed separately (pipx) and discovered on PATH. The
// command can be overridden with FOTI_BACKEND_CMD for development.

use std::net::TcpListener;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, State};

struct Backend {
    child: Mutex<Option<Child>>,
    port: u16,
}

/// Ask the OS for a free loopback port by binding :0 and reading it back.
fn pick_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(7777)
}

/// The base URL the webview should load once the backend is healthy.
#[tauri::command]
fn backend_url(state: State<Backend>) -> String {
    format!("http://127.0.0.1:{}", state.port)
}

/// True once the local daemon answers GET /health. The splash page polls this.
#[tauri::command]
fn backend_ready(state: State<Backend>) -> bool {
    let url = format!("http://127.0.0.1:{}/health", state.port);
    match ureq::get(&url).timeout(Duration::from_millis(800)).call() {
        Ok(resp) => resp.status() == 200,
        Err(_) => false,
    }
}

/// Split FOTI_BACKEND_CMD (or the default) into program + leading args, so an
/// override like "uv run --no-sync foti-backend" works as well as a bare
/// "foti-backend" on PATH.
fn backend_command() -> Command {
    let raw = std::env::var("FOTI_BACKEND_CMD")
        .unwrap_or_else(|_| "foti-backend".to_string());
    let mut parts = raw.split_whitespace();
    let program = parts.next().unwrap_or("foti-backend");
    let mut cmd = Command::new(program);
    for arg in parts {
        cmd.arg(arg);
    }
    cmd
}

fn spawn_backend(port: u16) -> std::io::Result<Child> {
    let mut cmd = backend_command();
    cmd.arg("serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string());
    // A desktop instance binds loopback only, so it runs without HTTP Basic
    // auth — strip any inherited credentials so no login prompt appears.
    cmd.env_remove("FOTI_BASIC_USER");
    cmd.env_remove("FOTI_BASIC_PASS");
    // When packaged as an AppImage, the AppRun wrapper injects its own runtime
    // environment (PYTHONHOME/PYTHONPATH point into the read-only mount, and
    // LD_LIBRARY_PATH/LD_PRELOAD point at bundled GUI libraries). Inheriting
    // those breaks the separately-installed Python backend — most visibly
    // "No module named 'encodings'" from a hijacked PYTHONHOME. Strip them so
    // the backend starts in a clean environment using its own interpreter.
    for var in [
        "PYTHONHOME",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "GTK_PATH",
        "GTK_EXE_PREFIX",
        "GDK_PIXBUF_MODULE_FILE",
        "GI_TYPELIB_PATH",
    ] {
        cmd.env_remove(var);
    }
    cmd.spawn()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = std::env::var("FOTI_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or_else(pick_port);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Backend {
            child: Mutex::new(None),
            port,
        })
        .setup(move |app| {
            match spawn_backend(port) {
                Ok(child) => {
                    *app.state::<Backend>().child.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    // The splash page reports the failure to the user after its
                    // poll loop times out; log for diagnostics.
                    eprintln!("foti-desktop: failed to start backend: {e}");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_url, backend_ready])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Tear the backend down with the app so we never orphan a daemon.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(mut child) = app_handle
                    .state::<Backend>()
                    .child
                    .lock()
                    .unwrap()
                    .take()
                {
                    let _ = child.kill();
                }
            }
        });
}
