"""Persistent isolated multi-device worker for the optional shared VAE test."""
from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimax_h3_sol_upscaler.protocol import parse_worker_request
from minimax_h3_sol_upscaler.shared_vae import run


def serve() -> None:
    responses = sys.stdout
    for line in sys.stdin:
        try:
            request = parse_worker_request(json.loads(line))
            with redirect_stdout(sys.stderr):
                report = run(request)
            output = Path(request["output"]).resolve()
            (output.parent / "worker-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            payload = {"status": "succeeded", "report": report}
        except Exception:  # noqa: BLE001 - request errors must remain in the JSON protocol
            payload = {"status": "failed", "error": traceback.format_exc()}
        print(json.dumps(payload, sort_keys=True), file=responses, flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: shared_vae_worker.py REQUEST.json | --serve")
    if sys.argv[1] == "--serve":
        serve()
        return
    request = parse_worker_request(json.loads(Path(sys.argv[1]).read_text()))
    with redirect_stdout(sys.stderr):
        report = run(request)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
