"""
swarm_results.json — Compiled results from the agent swarm sweep.

Each entry: {H, eta, arch, seed, epochs, iv_rmse, h_est, h_error, final_loss}
Auto-generated from the sweep directory by the Streamlit demo.
"""

import json, glob, sys
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent.parent / "02_rough_vol" / "results" / "sweep"
OUT = Path(__file__).resolve().parent / "swarm_results.json"

results = []
for f in sorted(SWEEP_DIR.glob("*.json")):
    try:
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = f.name
        results.append(d)
    except Exception:
        pass

# Also add manual entries for the heavy experiments
results.append({
    "_file": "causal_filter_comparison.py (swarm run)",
    "H": 0.10, "eta": 1.9, "arch": "causal", "seed": 0, "epochs": 3000, "rho": -0.9,
    "h_est": None, "h_error": None,
    "iv_rmse": 2.77,
    "notes": "Causal z-Cholesky vs BLP reference. Paper claimed 3.26 vp. Delta bias -0.06."
})
results.append({
    "_file": "causal_filter_comparison.py (swarm run)",
    "H": 0.10, "eta": 1.9, "arch": "circular", "seed": 0, "epochs": 3000, "rho": -0.9,
    "h_est": None, "h_error": None,
    "iv_rmse": 160.0,
    "notes": "Circular z-Cholesky vs BLP ref. Catastrophic failure. Paper claimed 204 vp."
})
results.append({
    "_file": "causal_filter_rl_match.py (swarm run)",
    "H": 0.10, "eta": 1.9, "arch": "causal_rl", "seed": 0, "epochs": 3000, "rho": -0.9,
    "h_est": None, "h_error": None,
    "iv_rmse": 2.24,
    "notes": "Matched-PSD (RL-fBm) training. Paper claimed 2.57 vp. Delta bias -0.001 (MC noise)."
})

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"Compiled {len(results)} swarm results → {OUT}")
