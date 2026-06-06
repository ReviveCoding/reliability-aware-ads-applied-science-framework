import subprocess, sys
from pathlib import Path

def test_track_ctr_cli_runs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--mode", "smoke", "--track", "ctr", "--n-rows", "300"], cwd=root, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "CTR track finished" in result.stdout
