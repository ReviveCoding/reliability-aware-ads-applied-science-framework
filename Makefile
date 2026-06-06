.PHONY: smoke test all ctr relevance uplift ope clean
smoke:
	python scripts/run_pipeline.py --mode smoke --track all
test:
	python -m pytest tests -q
all: smoke test
ctr:
	python scripts/run_pipeline.py --mode smoke --track ctr
relevance:
	python scripts/run_pipeline.py --mode smoke --track relevance
uplift:
	python scripts/run_pipeline.py --mode smoke --track uplift
ope:
	python scripts/run_pipeline.py --mode smoke --track ope
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
