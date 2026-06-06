from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config
from src.pipelines.all import run_all

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_all(load_config("configs/smoke.yaml"), root=root)
    print(f"Smoke pipeline finished. Release decision: {result['decision']['status']}")
