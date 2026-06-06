# Local Run Guide

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
python scripts/run_pipeline.py --mode smoke --track all
python -m pytest tests -q
```

## macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python scripts/run_pipeline.py --mode smoke --track all
python -m pytest tests -q
```

## Track-level runs

```bash
python scripts/run_pipeline.py --mode smoke --track ctr
python scripts/run_pipeline.py --mode smoke --track relevance
python scripts/run_pipeline.py --mode smoke --track uplift
python scripts/run_pipeline.py --mode smoke --track ope
```
