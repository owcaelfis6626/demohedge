"""
causal_filter_diagnostics.py  —  inference speed and H recovery for
CausalSpectralFilter vs (circular) SpectralFilter.

Question: is the causal architecture a drop-in replacement on the core
metrics (speed, Hurst recovery) or does it pay a price for causality?
"""

import time
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bochner_pinn import (SpectralFilter, CausalSpectralFilter,
                           get_exact_fgn_psd, train_one, evaluate_model)

H, N, T = 0.10, 252, 1.0
DT      = T / N
EPOCHS  = 3_000
N_GEN   = 10_000
N_TIME  = 5         # repeat timing runs and take median

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ─── Train both filters ───────────────────────────────────────────────────────
freqs, psd_exact = get_exact_fgn_psd(H, N, DT)

causal_model = CausalSpectralFilter(N, H=H, dt=DT)
causal_model, _ = train_one(H, N, DT, psd_exact, 'CausalSF', n_epochs=EPOCHS,
                             device=device, model=causal_model)

circ_model, _   = train_one(H, N, DT, psd_exact, 'CircularSF', n_epochs=EPOCHS,
                             device=device)

# ─── Inference timing: generate N_GEN paths, median over N_TIME runs ──────────
def bench(model, label):
    model.eval()
    times = []
    with torch.no_grad():
        # warm-up
        _ = model(torch.randn(N_GEN, N, device=device))
        torch.cuda.synchronize() if device.type == 'cuda' else None
        for _ in range(N_TIME):
            z = torch.randn(N_GEN, N, device=device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            _ = model(z)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.perf_counter() - t0)
    med_ms = 1e3 * np.median(times)
    print(f"  {label:<20}  gen {N_GEN:,} paths: {med_ms:6.2f} ms (median of {N_TIME})")
    return med_ms

print(f"\n--- Inference timing (gen {N_GEN:,} paths) ---")
t_causal = bench(causal_model, 'CausalSpectralFilter')
t_circ   = bench(circ_model,   'CircularSpectralFilter')

# ─── Parameter counts ─────────────────────────────────────────────────────────
n_causal = sum(p.numel() for p in causal_model.parameters())
n_circ   = sum(p.numel() for p in circ_model.parameters())
print(f"\n--- Parameter counts ---")
print(f"  CausalSpectralFilter:    {n_causal:>5} params")
print(f"  CircularSpectralFilter:  {n_circ:>5} params")

# ─── H recovery via structure function ───────────────────────────────────────
print(f"\n--- Hurst recovery (structure function on generated paths) ---")
res_causal = evaluate_model(causal_model, H, N, DT, device, n_eval=N_GEN, label='Causal')
res_circ   = evaluate_model(circ_model,   H, N, DT, device, n_eval=N_GEN, label='Circ ')

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("                    CAUSAL vs CIRCULAR DIAGNOSTICS")
print("=" * 70)
print(f"{'Filter':<22} | {'Params':>6} | {'Time/10k (ms)':>14} | {'H_est':>7} | {'|err|':>6}")
print("-" * 70)
print(f"{'CircularSpectralFilter':<22} | {n_circ:>6} | {t_circ:>14.2f} | "
      f"{res_circ['H_est']:>7.4f} | {abs(res_circ['H_est'] - H):>6.4f}")
print(f"{'CausalSpectralFilter':<22} | {n_causal:>6} | {t_causal:>14.2f} | "
      f"{res_causal['H_est']:>7.4f} | {abs(res_causal['H_est'] - H):>6.4f}")
print("=" * 70)
