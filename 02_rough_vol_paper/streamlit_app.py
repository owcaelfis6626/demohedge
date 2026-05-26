"""
streamlit_app.py — Spectral Rough Vol Demo (updated May 2026)

Covers the full pipeline: SpectralFilter (ρ=0), CausalSpectralFilter (ρ≠0),
VolterraSurrogate (cross-model), SPX calibration, Markovian speedup.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from rbm_reference import simulate_rbm_variance, simulate_rbm_blp, simulate_fbm_increments, compute_psd
from bochner_pinn import (SpectralFilter, CausalSpectralFilter,
                           get_exact_fgn_psd, asymptotic_fgn_psd, bochner_loss, train_one)
from volterra_surrogate import VolterraSurrogate
from implied_vol import (compute_iv_surface, compare_iv_surfaces,
                          _STRIKES_DEFAULT, _MATURITIES_DEFAULT,
                          FIGURES as IV_FIG, RESULTS as IV_RES)

st.set_page_config(page_title="Spectral Rough Vol Demo", page_icon="📈", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Model Parameters")
H = st.sidebar.slider("Hurst H", 0.02, 0.50, 0.10, 0.01)
eta = st.sidebar.slider("Vol-of-vol η", 0.5, 3.0, 1.9, 0.1)
xi0 = st.sidebar.slider("Initial variance ξ₀", 0.01, 0.10, 0.04, 0.01)
rho = st.sidebar.selectbox("Leverage ρ", [-0.9, -0.7, -0.5, 0.0], index=3)

N = 252; T_val = 1.0; dt = T_val / N
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.caption(f"Device: {device}")

n_epochs = st.sidebar.select_slider("Training epochs", [500, 1000, 2000, 3000], value=500)
batch_size = st.sidebar.slider("Batch size", 16, 256, 64, 16)
n_mc_paths = st.sidebar.slider("MC reference paths", 1000, 20000, 5000, 1000)
n_surr_paths = st.sidebar.slider("Surrogate pricing paths", 1000, 10000, 5000, 1000)

# ── Tabs ────────────────────────────────────────────────────────────────────
st.title("📈 Spectral Surrogate for Rough Volatility")
st.caption("Cramér PINN — match the PSD, skip the simulation.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Train Surrogate", "⚡ Causal ρ≠0", "🔬 Volterra Cross-Model", "🧠 Theory"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Train SpectralFilter / BochnerMLP (ρ=0)
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("SpectralFilter — Exact PSD Matching")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SpectralFilter (126 params, exact PSD)")
        if st.button("⚡ Train SpectralFilter", type="primary", key="btn_sf"):
            with st.spinner(f"Training ({n_epochs} epochs)..."):
                freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
                m = SpectralFilter(N, psd_init=psd_exact, dt=dt).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=3e-3)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=1e-5)
                psd_t = torch.tensor(psd_exact, dtype=torch.float32, device=device)
                progress = st.progress(0)
                status = st.empty()
                losses = []
                t0 = time.time()
                for ep in range(n_epochs):
                    z = torch.randn(batch_size, N, device=device)
                    loss = bochner_loss(m(z), psd_t, dt)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    losses.append(loss.item())
                    if ep % max(1, n_epochs//50) == 0:
                        progress.progress((ep+1)/n_epochs)
                        status.text(f"Epoch {ep+1}/{n_epochs} — loss {loss.item():.5f}")
                train_time = time.time() - t0
                progress.progress(1.0)
                status.text(f"Done! Loss {losses[-1]:.5f} ({train_time:.1f}s)")

                # Evaluate H recovery
                m.eval()
                with torch.no_grad():
                    dW = m(torch.randn(4000, N, device=device)).cpu().numpy()
                W_eval = np.zeros((4000, N+1))
                W_eval[:,1:] = np.cumsum(dW, axis=1)
                from rbm_reference import estimate_hurst_structure_function
                H_est = estimate_hurst_structure_function(W_eval, dt)["H_est"]

                st.session_state["m_exact"] = m.cpu()
                st.session_state["H_est"] = H_est
                st.session_state["losses"] = losses
                st.session_state["train_time"] = train_time
                st.session_state["freqs"] = freqs
                st.session_state["psd_exact"] = psd_exact
                st.session_state["dW_eval"] = dW

                st.success(f"Ĥ={H_est:.4f}  |Ĥ−H|={abs(H_est-H):.4f}  Loss={losses[-1]:.5f}  Time={train_time:.1f}s")

    with col2:
        st.subheader("BochnerMLP (41k params, exact PSD)")
        if st.button("⚡ Train BochnerMLP", key="btn_mlp"):
            with st.spinner(f"Training ({n_epochs} epochs)..."):
                freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
                m = torch.nn.Sequential(
                    torch.nn.Linear(N, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, N),
                ).to(device)
                def mlp_forward(z):
                    out = m(z)
                    return out - out.mean(1, keepdim=True)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=1e-5)
                psd_t = torch.tensor(psd_exact, dtype=torch.float32, device=device)
                progress = st.progress(0)
                status = st.empty()
                losses_mlp = []
                t0 = time.time()
                for ep in range(n_epochs):
                    z = torch.randn(batch_size, N, device=device)
                    loss = bochner_loss(mlp_forward(z), psd_t, dt)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    losses_mlp.append(loss.item())
                    if ep % max(1, n_epochs//50) == 0:
                        progress.progress((ep+1)/n_epochs)
                        status.text(f"Epoch {ep+1}/{n_epochs} — loss {loss.item():.5f}")
                train_time = time.time() - t0
                progress.progress(1.0)
                status.text(f"Done! Loss {losses_mlp[-1]:.5f} ({train_time:.1f}s)")

                st.session_state["m_mlp"] = m.cpu()
                st.session_state["mlp_forward"] = mlp_forward
                st.session_state["losses_mlp"] = losses_mlp
                n_params = sum(p.numel() for p in m.parameters())
                st.success(f"Params={n_params:,}  Loss={losses_mlp[-1]:.5f}  Time={train_time:.1f}s")

    # Loss curves
    if "losses" in st.session_state:
        st.markdown("---")
        st.subheader("Training Curves")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.semilogy(st.session_state["losses"], label="SpectralFilter (126p)", color="#e74c3c")
        if "losses_mlp" in st.session_state:
            ax.semilogy(st.session_state["losses_mlp"], label="BochnerMLP (41k)", color="#2ecc71")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Bochner loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # IV surface (ρ=0)
    if "m_exact" in st.session_state:
        st.markdown("---")
        st.subheader("IV Surface (ρ=0)")
        if st.button("💹 Compare IV Surfaces", key="btn_iv"):
            with st.spinner("Simulating reference and surrogate paths..."):
                # Reference
                V_ref, _, _ = simulate_rbm_variance(H, eta, xi0, N, min(n_mc_paths, 5000), T_val, seed=1)
                iv_ref, K_arr, T_arr = compute_iv_surface(V_ref, dt, rho=0.0)

                # Surrogate
                m = st.session_state["m_exact"]
                with torch.no_grad():
                    dW_s = m(torch.randn(n_surr_paths, N)).cpu().numpy()
                W_s = np.zeros((n_surr_paths, N+1))
                W_s[:,1:] = np.cumsum(dW_s, axis=1)
                var_W = W_s.var(axis=0)
                drift_corr = -0.5 * eta**2 * var_W
                V_s = xi0 * np.exp(eta * W_s + drift_corr[None,:])
                iv_s, _, _ = compute_iv_surface(V_s, dt, rho=0.0)

                s = compare_iv_surfaces(iv_ref, iv_s, K_arr, T_arr)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            moneyness = _STRIKES_DEFAULT
            for i, (ax, Tm) in enumerate(zip(axes, _MATURITIES_DEFAULT)):
                ax.plot(moneyness, iv_ref[i]*100, "ko-", ms=4, lw=2, label="MC truth")
                ax.plot(moneyness, iv_s[i]*100, "s--", color="#e74c3c", ms=4, lw=1.5, label="Surrogate")
                ax.set_xlabel("K/S₀"); ax.set_title(f"T={Tm:.2f} yr")
                if i==0: ax.set_ylabel("IV (%)")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            fig.suptitle(f"rBergomi IV Smile (H={H}, η={eta}, ρ=0)", fontweight="bold")
            st.pyplot(fig); plt.close(fig)

            st.metric("IV RMSE (vol pts)", f"{s['rmse_overall']:.3f}")
            per_mat = " | ".join(f"T={t:.2f}: {v:.3f}" for t, v in zip(T_arr, s["rmse_by_maturity"].values()))
            st.caption(per_mat)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CausalSpectralFilter (ρ≠0)
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("CausalSpectralFilter — ρ≠0 via z-Cholesky")

    st.markdown("""
    The circular FFT SpectralFilter fails catastrophically at ρ≠0 (204 vp, delta > 1) because
    the circular convolution makes $V_k$ depend on future innovations $z_{k+1},...,z_{N-1}$,
    violating the martingale condition.

    The **CausalSpectralFilter** replaces the circular FFT with a lower-triangular Toeplitz
    convolution: $\\Delta\\hat{W}_k = \\sum_{j=0}^k h_{k-j} z_j$. Output at step $k$ depends
    only on $z_0,...,z_k$, so $V_k \\perp z_k$ and the $\\rho=-0.9$ Cholesky is valid.
    """)

    if st.button("⚡ Train CausalSpectralFilter (ρ≠0)", type="primary", key="btn_causal"):
        with st.spinner(f"Training causal filter ({n_epochs} epochs)..."):
            freqs, psd_exact = get_exact_fgn_psd(H, N, dt)
            m = CausalSpectralFilter(N).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=1e-5)
            psd_t = torch.tensor(psd_exact, dtype=torch.float32, device=device)
            progress = st.progress(0)
            status = st.empty()
            losses = []
            t0 = time.time()
            for ep in range(n_epochs):
                z = torch.randn(batch_size, N, device=device)
                loss = bochner_loss(m(z), psd_t, dt)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                losses.append(loss.item())
                if ep % max(1, n_epochs//50) == 0:
                    progress.progress((ep+1)/n_epochs)
                    status.text(f"Epoch {ep+1}/{n_epochs} — loss {loss.item():.5f}")
            train_time = time.time() - t0
            progress.progress(1.0)
            status.text(f"Done! Loss {losses[-1]:.5f} ({train_time:.1f}s)")

            # Evaluate
            m.eval()
            with torch.no_grad():
                dW = m(torch.randn(4000, N, device=device)).cpu().numpy()
            W_eval = np.zeros((4000, N+1))
            W_eval[:,1:] = np.cumsum(dW, axis=1)
            from rbm_reference import estimate_hurst_structure_function
            H_est = estimate_hurst_structure_function(W_eval, dt)["H_est"]

            st.session_state["m_causal"] = m.cpu()
            st.session_state["H_est_causal"] = H_est

            st.success(
                f"CausalSpectralFilter trained ({train_time:.1f}s)\n\n"
                f"Params: {N} (vs 126 for circular)  |  Ĥ={H_est:.4f}  "
                f"|Ĥ−H|={abs(H_est-H):.4f}\n\n"
                f"Inference: ~12ms/10k paths (6.7× slower than circular, 29× faster than Wood-Chan)"
            )

    if "m_causal" in st.session_state:
        st.markdown("---")
        st.subheader("Expected Performance at ρ=−0.9")
        st.markdown("""
        | Architecture | IV RMSE (vp) | ATM Δ bias | Notes |
        |-------------|-------------|-----------|-------|
        | Circular SF + z-Cholesky | 204 | Δ>1 | Martingale violated |
        | Causal SF (ρ=0) | 6.8 | −0.17 | Level ok, no skew |
        | Causal SF + z-Cholesky | **3.26** | **−0.07** | 63× improvement |
        | Causal SF + matched PSD | **2.57** | **≤0.011** | MC noise floor |
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Volterra Cross-Model
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("VolterraSurrogate — Cross-Model Gap Closure")

    st.markdown("""
    The CausalSpectralFilter resolves the $\\rho\\neq 0$ wrap-around for rBergomi, but
    cross-model (rBergomi surrogate → rough Heston reference) still has a residual
    delta bias of −0.15. The issue: rBergomi's log-normal $V$ distribution doesn't
    match rough Heston's mean-reverting dynamics.

    The **VolterraSurrogate** is an autoregressive module with V-conditional diffusion:
    $$V_{k+1} = V_0 + c_d\\sum w_{k-j}(\\theta-V_j) + c_s\\sum w_{k-j}\\sqrt{V_j}z_j$$

    Paired with the same z-Cholesky, this closes the cross-model gap to Monte Carlo noise
    (0.80 vp, delta bias ≤0.005) — identifying $\\sqrt{V_j}$ in the diffusion as the
    precise architectural requirement.
    """)

    st.markdown("---")
    st.subheader("Cross-Model Performance (rBergomi surr → rough Heston ref)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Spectral", "4.57 vp", "Δ bias −0.15")
    with col2:
        st.metric("VolterraSurrogate", "0.80 vp", "Δ bias +0.002")
    with col3:
        st.metric("Improvement", "5.7×", "MC noise level")

    st.markdown("---")
    st.subheader("Markovian Speedup")
    st.markdown("""
    Approximating the RL kernel with $M$ exponentials turns the $O(N^2)$ Volterra recursion
    into $O(MN)$ via recursively updatable states:

    | Variant | Fwd time (ms) | Speedup | IV RMSE (vp) | Δ bias |
    |---------|-------------|---------|-------------|--------|
    | O(N²) Volterra | 280 | 1× | 0.91 | −0.004 |
    | M=3 Markovian | 47 | 6.0× | 0.76 | −0.001 |
    | M=5 Markovian | **39** | **7.2×** | **0.73** | **+0.001** |

    The 7× speedup comes at zero accuracy cost — the Markovian kernel-fit error (~4% relative)
    is below the MC noise threshold.
    """)

    st.markdown("---")
    st.subheader("Live SPY Calibration")
    st.markdown("""
    The same VolterraSurrogate trained end-to-end from a misspecified init calibrates to
    live SPY option chains across three expirations simultaneously:

    | Expiry | Strikes | IV RMSE (vp) |
    |--------|---------|-------------|
    | +32d (Jun 26) | 8 | 2.27 |
    | +88d (Aug 21) | 9 | 3.79 |
    | +179d (Nov 20) | 9 | 5.17 |
    | **Overall** | **26** | **3.97** |

    Training: 3 minutes on GTX 950. All parameters shared across maturities.
    Short-end within 2× bid/ask uncertainty. Long-end residual (~5 vp) reflects the
    stationary-kernel limitation — a known property of rough vol calibrations.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Theory
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("How It Works")

    st.markdown(r"""
    ### The Problem
    The rough Bergomi model is state-of-the-art for options pricing but requires
    expensive fBm simulation ($O(N\log N)$ per path via Wood-Chan).

    ### The Solution
    Instead of simulating fBm, train a neural network to generate fGn increments
    whose PSD matches the exact fGn PSD. The surrogate is:

    - **SpectralFilter** (126 params): $\\Delta W = \\text{IRFFT}(\\exp(\\alpha) \\cdot \\text{RFFT}(z))$
    - **CausalSpectralFilter** (252 params): $\\Delta W_k = \\sum_{j\\leq k} h_{k-j} z_j$
    - **VolterraSurrogate** (255 params): autoregressive with $\\sqrt{V_j}$ diffusion

    ### Why It Works
    1. fGn is **stationary** → its PSD fully characterizes the Gaussian law (Cramér rep.)
    2. Matching the **exact PSD** (not just the asymptotic $C|f|^{1-2H}$) forces
       the surrogate's distribution to converge to the reference (W₂ bound)
    3. The **causal** architecture fixes the circular FFT wrap-around that breaks
       ρ≠0 correlation
    4. **√V_j diffusion** closes the remaining cross-model gap

    ### The Guarantee
    $$W_2(\\mathcal{L}(u_\\theta), \\mathcal{L}(u_{\\text{ref}})) \\leq C_1\\sqrt{\\varepsilon} + C_2 N_s^{-\\beta_H} + C_3 \\omega_0^{-(1-2H)/2}$$
    """)

    st.info(
        "📄 **Paper:** *Distributional Learning of Rough Volatility via Spectral "
        "Surrogates: a Cramér PINN for fBm-Driven SDEs* — Hubert Lipski, 2026\n\n"
        "Based on the Bochner PINN framework from: *Closing the Variance Gap: "
        "Bochner PINN for Lévy-Driven SPDEs*"
    )

st.markdown("---")
st.caption("Built with Streamlit · Spectral Surrogate uses PyTorch · MC reference uses Wood-Chan circulant embedding")
