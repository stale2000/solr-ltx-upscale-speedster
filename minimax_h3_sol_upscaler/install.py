"""ComfyUI-Manager bridge for the refiner's independent native worker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def prepare_worker(worker_dir: Path, *, source_root: Path | None = None) -> None:
    """Copy the standalone host interpreter, excluding its installed packages.

    A normal venv records the build host's absolute interpreter/stdlib path. The
    platform relocates its archive, so this worker needs its own interpreter and
    stdlib. Package installs below use this copy and never share host Torch.
    """
    source = (source_root or Path(sys.base_prefix)).resolve()
    worker_dir = worker_dir.resolve()
    marker = worker_dir / ".portable-runtime"
    if marker.is_file() and (worker_dir / "bin/python").is_file():
        return
    if worker_dir.exists():
        raise RuntimeError(f"Refusing to reuse a nonportable worker runtime: {worker_dir}")
    if worker_dir.is_relative_to(source):
        raise RuntimeError("The copied worker must be outside the host interpreter tree")
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if not (source / "bin" / minor).is_file() or not (source / "lib" / minor / "os.py").is_file():
        raise RuntimeError("Managed builds require a self-contained standalone Python interpreter")
    worker_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sol-worker-", dir=worker_dir.parent) as temp_dir:
        staged = Path(temp_dir) / "runtime"
        # Dereference interpreter/library aliases: absolute symlinks into the build
        # tree would undo the relocation guarantee. Never copy host site-packages.
        shutil.copytree(
            source, staged, symlinks=False,
            ignore=shutil.ignore_patterns("site-packages", "dist-packages", "__pycache__", "pyvenv.cfg"),
        )
        interpreter = staged / "bin" / minor
        stable = staged / "bin/python"
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(interpreter.name)
        # uv honors this marker when it writes relocatable console entry points.
        (staged / "pyvenv.cfg").write_text("relocatable = true\n", encoding="utf-8")
        (staged / ".portable-runtime").write_text("standalone-python\n", encoding="utf-8")
        staged.rename(worker_dir)


def main() -> None:
    if sys.argv[1:2] == ["--prepare-worker"] and len(sys.argv) == 3:
        prepare_worker(Path(sys.argv[2]))
        return
    if len(sys.argv) != 1:
        raise SystemExit("Usage: install.py [--prepare-worker DIRECTORY]")
    environment = dict(os.environ, PYTHON_BIN=sys.executable)
    subprocess.run(
        ["bash", str(Path(__file__).with_name("install_runtime.sh")), "--build-only"],
        check=True, env=environment,
    )


if __name__ == "__main__":
    main()
