"""Execute notebooks in the current checkout using this Python's clean kernel.

Notebooks 00-04 write historical outputs: use only an isolated copy of the project.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("notebooks", nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel_root = Path(tempfile.mkdtemp(prefix="btl_kernel_"))
    spec = kernel_root / "kernels" / "btl-acceptance"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "BTL acceptance", "language": "python",
    }), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    outcomes = []
    for name in args.notebooks:
        path = Path(name).resolve()
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        row = {"notebook": path.name, "python": sys.executable}
        try:
            NotebookClient(notebook, timeout=1800, kernel_name="btl-acceptance",
                           resources={"metadata": {"path": str(path.parent)}}).execute()
            row["status"] = "passed"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        nbformat.write(notebook, args.output_dir / path.name)
        outcomes.append(row)
        (args.output_dir / "execution_results.json").write_text(
            json.dumps(outcomes, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k != "error"}), flush=True)
    if any(row["status"] != "passed" for row in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
