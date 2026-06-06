from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
metrics_path = root / "outputs/metrics_summary.json"
decision_path = root / "outputs/release_decision.json"

if not metrics_path.exists() or not decision_path.exists():
    raise SystemExit("Run scripts/run_smoke_pipeline.py first.")

metrics = json.loads(metrics_path.read_text())
decision = json.loads(decision_path.read_text())

bullets = [
    "Built a local-runnable, public-data-compatible ads applied-science framework with synthetic fallback to evaluate CTR/CVR prediction, empirical-Bayes calibration, product-query relevance, uplift measurement, OPE, ranking-policy simulation, and release gates.",
    "Implemented reliability-aware upgrades over vanilla baselines, including PHT-style lifecycle features, mechanism-informed residual pCTR correction, ABC-style candidate expansion with bias-corrected reranking, and slice-wise reliability metrics.",
    f"Generated model/data cards, policy-comparison artifacts, drift/update diagnostics, and PASS/ITERATE/BLOCK release reports; smoke-mode release decision was {decision['status']} under explicit non-production claim boundaries."
]
out = root / "outputs/resume_bullets.md"
out.write_text("# Resume Bullets\n\n" + "\n".join(f"- {b}" for b in bullets), encoding="utf-8")
print(out)
