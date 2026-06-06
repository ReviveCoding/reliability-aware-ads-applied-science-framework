from pathlib import Path
import json
import pandas as pd


def ensure_dirs(root="."):
    root = Path(root)
    for d in ["reports", "outputs", "artifacts"]:
        (root / d).mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_markdown(path, title, sections):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for header, body in sections:
        lines += [f"## {header}", ""]
        if isinstance(body, pd.DataFrame):
            lines.append(body.to_markdown(index=False))
        else:
            lines.append(str(body))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
