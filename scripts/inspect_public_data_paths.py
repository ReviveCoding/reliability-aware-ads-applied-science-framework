from pathlib import Path
import os
import json
import pandas as pd
import gzip
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

ENV_PATHS = {
    "criteo_uplift": os.environ.get("CRITEO_UPLIFT_PATH"),
    "esci": os.environ.get("ESCI_DATA_DIR"),
    "open_bandit": os.environ.get("OPEN_BANDIT_DIR"),
    "criteo_click": os.environ.get("CRITEO_CLICK_DIR"),
}

def list_files(path, max_files=80):
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        return [p]
    files = []
    for ext in ["*.csv", "*.tsv", "*.txt", "*.gz", "*.zip", "*.parquet", "*.json", "*.jsonl"]:
        files.extend(p.rglob(ext))
    return files[:max_files]

def safe_preview(path, nrows=5):
    path = Path(path)
    info = {
        "file": str(path),
        "suffix": path.suffix,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
        "preview_status": "not_attempted",
    }
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            info["shape"] = list(df.shape)
            info["columns"] = list(df.columns)
            info["head"] = df.head(nrows).astype(str).to_dict(orient="records")
            info["preview_status"] = "ok"
        elif path.suffix.lower() in [".csv", ".tsv", ".txt", ".gz"]:
            sep = "\t" if path.suffix.lower() in [".tsv", ".txt"] else ","
            # Criteo click logs are often tab-separated even when .gz/.txt.
            try:
                df = pd.read_csv(path, nrows=nrows, sep=sep)
                if df.shape[1] <= 2:
                    df = pd.read_csv(path, nrows=nrows, sep="\t", header=None)
            except Exception:
                df = pd.read_csv(path, nrows=nrows, sep="\t", header=None)
            info["shape_preview"] = list(df.shape)
            info["columns"] = [str(c) for c in df.columns]
            info["head"] = df.head(nrows).astype(str).to_dict(orient="records")
            info["preview_status"] = "ok"
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                info["zip_members_sample"] = z.namelist()[:30]
                info["preview_status"] = "zip_listed"
        else:
            info["preview_status"] = "skipped_extension"
    except Exception as e:
        info["preview_status"] = "error"
        info["error"] = repr(e)
    return info

def main():
    result = {}
    for name, path in ENV_PATHS.items():
        if not path:
            result[name] = {"status": "missing_env"}
            continue
        p = Path(path)
        files = list_files(p)
        result[name] = {
            "path": str(p),
            "exists": p.exists(),
            "is_file": p.is_file(),
            "file_count_sample": len(files),
            "files": [str(f) for f in files[:40]],
            "previews": [safe_preview(f) for f in files[:10]],
        }

    out_path = OUT / "public_data_inventory.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    md = ["# Public Dataset Inventory", ""]
    for name, item in result.items():
        md.append(f"## {name}")
        md.append("")
        md.append(f"- Path: `{item.get('path', '')}`")
        md.append(f"- Exists: `{item.get('exists', '')}`")
        md.append(f"- File count sample: `{item.get('file_count_sample', '')}`")
        md.append("")
        for prev in item.get("previews", [])[:5]:
            md.append(f"### {Path(prev['file']).name}")
            md.append(f"- Size MB: `{prev.get('size_mb')}`")
            md.append(f"- Preview status: `{prev.get('preview_status')}`")
            md.append(f"- Columns: `{prev.get('columns', prev.get('zip_members_sample', []))}`")
            if prev.get("error"):
                md.append(f"- Error: `{prev['error']}`")
            md.append("")
    report_path = REPORTS / "public_data_inventory.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {report_path}")
    print(json.dumps({k: {"exists": v.get("exists"), "file_count_sample": v.get("file_count_sample")} for k, v in result.items()}, indent=2))

if __name__ == "__main__":
    main()
