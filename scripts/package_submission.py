"""Build the academic submission with an explicit allowlist and SHA-256 manifest.

Output must be outside the checkout. No Git, venv, caches or cloud links are needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "2411969-2313598-2310409-2412432"
BASELINE = "c0ae203"
ROOT_FILES = {"README.md", "requirements.txt", "requirements-dev.txt",
              "requirements-docs.txt", "Dockerfile", ".dockerignore", ".gitignore",
              "app.py", "run_pipeline.py"}
FOLDERS = {"src", "tests", "scripts", "notebooks", "figures", "models", "reports", "docs"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "superpowers"}
MANDATORY = ["docs/Report.pdf", "docs/Slides.pptx", "docs/Slides.pdf",
             "reports/report_draft.docx", "src/predict.py", "data/sample_test.csv",
             "models/champion_model.pkl", "models/preprocessing_pipeline.pkl",
             "models/best_hyperparameters.json", "notebooks/00_EDA_Final.ipynb",
             "notebooks/04_features_pipeline.ipynb", "notebooks/05_model_experiments.ipynb",
             "notebooks/06_evaluation.ipynb"]


def build(output_dir, size_limit_mb=None):
    output_dir = Path(output_dir).resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("Use an output directory outside the project checkout.")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if (relative.as_posix() in ROOT_FILES or relative.parts[0] in FOLDERS
                or relative.as_posix() in {"data/raw/bank.csv", "data/sample_test.csv"}
                or (relative.parent.as_posix() == "data/processed" and path.suffix == ".parquet")):
            files.append(path)
    names = {p.relative_to(ROOT).as_posix() for p in files}
    missing = set(MANDATORY) - names
    if missing:
        raise FileNotFoundError(f"Missing mandatory files: {sorted(missing)}")
    manifest = {
        "submission_name": NAME, "baseline_commit": BASELINE,
        "student_id_order": "Person 1 Loi, Person 2 Tri, Person 3 Danh, Person 4 Nhan",
        "size_limit_mb": size_limit_mb,
        "exclusions": [".git", "caches", "virtual environments", "docs/superpowers",
                       "redundant raw archive.zip", "duplicate sample prediction CSVs"],
        "files": {p.relative_to(ROOT).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        } for p in files},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (NAME + ".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            info = zipfile.ZipInfo(NAME + "/" + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 11, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo(NAME + "/SUBMISSION_MANIFEST.json", date_time=(2026, 11, 5, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    size_mb = target.stat().st_size / 1_000_000
    if size_limit_mb is not None and size_mb > size_limit_mb:
        raise ValueError(f"Archive {size_mb:.2f} MB exceeds requested limit {size_limit_mb} MB")
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size,
                      "decimal_mb": size_mb, "files_including_manifest": len(files)+1,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size-limit-mb", type=float)
    args = parser.parse_args()
    build(args.output_dir, args.size_limit_mb)
