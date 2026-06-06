from pathlib import Path
import json
import shutil
import pandas as pd

class ArtifactManager:
    def __init__(self, root="."):
        self.root = Path(root)
        for d in ["reports", "outputs", "sample_outputs"]:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def json(self, rel, obj):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def csv(self, rel, df):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def md(self, rel, title, sections):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {title}", ""]
        for header, body in sections:
            lines += [f"## {header}", ""]
            lines.append(body.to_markdown(index=False) if isinstance(body, pd.DataFrame) else str(body))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def snapshot_examples(self):
        for dname in ["reports", "outputs"]:
            src = self.root / dname
            dst = self.root / "sample_outputs" / dname
            if dst.exists(): shutil.rmtree(dst)
            if src.exists(): shutil.copytree(src, dst)
