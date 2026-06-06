from pathlib import Path

DEFAULT_CONFIG = {
    "mode": "smoke",
    "n_rows": 600,
    "n_products": 80,
    "n_queries": 32,
    "random_seed": 42,
    "top_fraction": 0.2,
    "enable_dnn": False,
    "release": {
        "max_ece": 0.12,
        "max_sparse_ece": 0.22,
        "max_false_high_rank_irrelevant_rate": 0.35,
        "min_long_tail_exposure": 0.08,
        "min_ope_lcb_lift": -0.01,
    },
}

def _coerce(value):
    v = str(value).strip().strip('"').strip("'")
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v

def load_config(path=None, overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["release"] = dict(DEFAULT_CONFIG["release"])
    if path:
        p = Path(path)
        if p.exists():
            section = None
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].rstrip()
                if not line.strip():
                    continue
                if not line.startswith(" ") and ":" in line:
                    k, val = line.split(":", 1)
                    k = k.strip(); val = val.strip()
                    if val == "":
                        section = k
                        cfg.setdefault(k, {})
                    else:
                        cfg[k] = _coerce(val)
                        section = None
                elif section and ":" in line:
                    k, val = line.split(":", 1)
                    cfg[section][k.strip()] = _coerce(val)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg
