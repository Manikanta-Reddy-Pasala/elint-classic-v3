import subprocess

"""Rust engine bridge for the pytest harness (AI_ENGINE=rust).

The Rust engine (ai_processor --csv) does everything itself: wipes the PG/CH
tables, loads the CSV into ClickHouse, runs the pipeline and writes results
back to PG/CH.  This module only drives the binary and waits for it to finish.

Env:
    RUST_AI_PROCESSOR  path to the ai_processor binary (default:
                       /home/sysadm/ai_elint_rust/target/release/ai_processor)
    RUST_AI_TIMEOUT    per-file timeout in seconds (default 600)
"""

import os


def _run_rust_engine(csv_file_path: str, source: str = "omni", timeout: int = 600) -> None:
    """Run the Rust engine once for one CSV, then verify data landed in PG.

    The engine itself wipes PG/CH, loads the CSV into ClickHouse csv_input,
    runs the pipeline and writes emitters/signals/clusters back.
    """
    import helpers.config as config

    binary = os.environ.get(
        "RUST_AI_PROCESSOR",
        "/home/sysadm/ai_elint_rust/target/release/ai_processor",
    )
    env = os.environ.copy()
    env.update(
        {
            "PG_URL": os.environ.get("PG_URL", "postgresql://elint:elint@127.0.0.1:15432/sigint"),
            "CH_URL": os.environ.get("CH_URL", "http://127.0.0.1:18123"),
            "ONNX_PATH": "/home/sysadm/models/transformer_model_known_v16.onnx",
            # RUST_ENGINE=classic (default) | onnx  — onnx needs ONNX_URL (model service)
            "CLASSIC": "0" if os.environ.get("RUST_ENGINE", "classic") == "onnx" else "1",
            "ONNX_SESSION_COUNT": "1",
            "ONNX_THREADS": "1",
        }
    )

    result = subprocess.run(
        [binary, "--csv", str(csv_file_path)],
        env=env,
        timeout=int(os.environ.get("RUST_AI_TIMEOUT", timeout)),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(f"=== RUST STDOUT ({os.path.basename(str(csv_file_path))}) ===")
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Rust engine failed (rc={result.returncode}) on {csv_file_path}\n"
            f"STDERR:\n{result.stderr}"
        )
