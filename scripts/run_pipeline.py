import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config
from src.pipelines.all import run_all
from src.pipelines.ctr import run_ctr_track
from src.pipelines.relevance import run_relevance_track
from src.pipelines.uplift import run_uplift_track
from src.pipelines.ope import run_ope_track

def main():
    parser = argparse.ArgumentParser(description="Run the ads applied science framework.")
    parser.add_argument("--mode", default="smoke", choices=["smoke","sample","full"])
    parser.add_argument("--track", default="all", choices=["all","ctr","relevance","uplift","ope"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-rows", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config or f"configs/{args.mode}.yaml", overrides={"mode": args.mode, "n_rows": args.n_rows})
    root = Path(__file__).resolve().parents[1]
    if args.track == "all":
        result = run_all(cfg, root=root); print(f"Pipeline finished. Release decision: {result['decision']['status']}"); print(f"Reports written to: {root/'reports'}"); print(f"Outputs written to: {root/'outputs'}")
    elif args.track == "ctr":
        out = run_ctr_track(cfg); print(f"CTR track finished. Rows: {len(out['test'])}")
    elif args.track == "relevance":
        out = run_relevance_track(cfg); print(f"Relevance track finished. Variants: {list(out['metrics'].keys())}")
    elif args.track == "uplift":
        out = run_uplift_track(cfg); print(f"Uplift track finished. AUUC: {out['metrics']['auuc']:.6f}")
    elif args.track == "ope":
        out = run_ope_track(cfg); print(f"OPE track finished. Estimators: {list(out['metrics'].keys())}")
if __name__ == "__main__": main()
