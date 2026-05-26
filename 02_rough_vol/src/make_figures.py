"""
make_figures.py  —  Publication-quality visualisations for Paper 2.

Figures generated
─────────────────
  roughness_gallery.png    fBm + variance paths for H ∈ {0.05,0.1,0.3,0.5}
  roughness_animation.gif  fBm ensemble morphing H: 0.5 → 0.05
  psd_comparison.png       exact fGn PSD vs asymptotic power-law (log-log)
  iv_surface_3d.png        3D reference MC implied-vol surface
  iv_surrogate_panel.png   per-maturity smile: MC vs three surrogates
  rmse_heatmap.png         RMSE (vol pts) summary heatmap
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3d projection
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from rbm_reference import simulate_fbm_paths, simulate_rbm_variance
from bochner_pinn import get_exact_fgn_psd, asymptotic_fgn_psd

FIGURES = Path(__file__).parent.parent / 'figures'
RESULTS = Path(__file__).parent.parent / 'results'

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       12,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ── Palette ──────────────────────────────────────────────────────────────────
C = dict(ref='#111111', exact='#c0392b', asymp='#2471a3', mlp='#1e8449',
         gold='#f1c40f')
H_CMAP = plt.cm.plasma   # colour-codes H values across figures


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Roughness gallery
# ═══════════════════════════════════════════════════════════════════════════════

def fig_roughness_gallery():
    H_vals  = [0.5, 0.3, 0.1, 0.05]
    H_norms = [0.0, 0.33, 0.67, 1.0]   # position in plasma cmap
    eta, xi0, N, T = 1.9, 0.04, 252, 1.0
    dt = T / N
    t  = np.linspace(0, T, N + 1)
    n  = 12   # paths per panel

    fig = plt.figure(figsize=(18, 8), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(2, 4, hspace=0.08, wspace=0.07,
                            top=0.88, bottom=0.10, left=0.05, right=0.97)

    for col, (H, hn) in enumerate(zip(H_vals, H_norms)):
        colour = H_CMAP(hn)
        W = simulate_fbm_paths(H, N, n, T, seed=col * 7 + 1)

        # ── fBm row ─────────────────────────────────────────────────
        ax_w = fig.add_subplot(gs[0, col], facecolor='#0d0d0d')
        for i in range(n):
            ax_w.plot(t, W[i], color=colour, alpha=0.55, lw=0.9)
        ax_w.axhline(0, color='white', lw=0.4, ls='--', alpha=0.3)
        ax_w.set_xlim(0, T);  ax_w.set_xticks([])
        ax_w.tick_params(colors='#aaaaaa')
        for sp in ax_w.spines.values():
            sp.set_color('#333333')
        if col == 0:
            ax_w.set_ylabel('W$^H_t$', color='#aaaaaa')
        ax_w.set_title(f'H = {H}', color=colour, fontsize=15, fontweight='bold',
                       pad=6)

        # ── Variance row ─────────────────────────────────────────────
        drift = -0.5 * eta**2 * t**(2 * H)
        V = xi0 * np.exp(eta * W + drift[None, :])

        ax_v = fig.add_subplot(gs[1, col], facecolor='#0d0d0d')
        for i in range(n):
            ax_v.plot(t, np.sqrt(V[i]) * 100, color=colour, alpha=0.55, lw=0.9)
        ax_v.axhline(np.sqrt(xi0) * 100, color='white', lw=0.6, ls='--',
                     alpha=0.4, label=f'{np.sqrt(xi0)*100:.0f}% base vol')
        ax_v.set_xlim(0, T);  ax_v.set_xlabel('$t$', color='#aaaaaa')
        ax_v.set_ylim(bottom=0)
        ax_v.tick_params(colors='#aaaaaa')
        for sp in ax_v.spines.values():
            sp.set_color('#333333')
        if col == 0:
            ax_v.set_ylabel('Spot vol $\\sigma_t$ (%)', color='#aaaaaa')

    fig.suptitle(
        'Rough Bergomi: volatility becomes rougher as H → 0',
        color='white', fontsize=16, fontweight='bold', y=0.96)

    fpath = FIGURES / 'roughness_gallery.png'
    fig.savefig(fpath, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f'  roughness_gallery.png  →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Roughness animation  (H: 0.5 → 0.05)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_roughness_animation(n_paths=25, N=252, T=1.0, n_frames=36):
    H_seq = np.concatenate([
        np.linspace(0.5, 0.05, n_frames // 2),
        np.linspace(0.05, 0.5, n_frames // 2)])   # loop back
    t = np.linspace(0, T, N + 1)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, T)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel('$t$', color='#aaaaaa', fontsize=13)
    ax.set_ylabel('$W^H_t$', color='#aaaaaa', fontsize=13)
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values():
        sp.set_color('#333333')

    palette = plt.cm.plasma(np.linspace(0.1, 0.9, n_paths))
    lines = [ax.plot([], [], lw=0.9, alpha=0.55, color=palette[i])[0]
             for i in range(n_paths)]
    title_obj = ax.set_title('', color='white', fontsize=15, pad=8)
    hline = ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.25)   # noqa

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        H = H_seq[frame]
        W = simulate_fbm_paths(H, N, n_paths, T, seed=42)
        hn = 1.0 - (H - 0.05) / 0.45           # 1 at H=0.05 (rough), 0 at H=0.5
        colour = H_CMAP(0.15 + 0.7 * hn)
        for i, ln in enumerate(lines):
            ln.set_data(t, W[i])
            ln.set_color(colour)
        title_obj.set_text(
            f'Fractional Brownian Motion    H = {H:.3f}   '
            f'{"(very rough — like SPX vol)" if H < 0.12 else ""}')
        title_obj.set_color(colour)
        return lines + [title_obj]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=120, blit=True)
    fpath = FIGURES / 'roughness_animation.gif'
    anim.save(fpath, writer=PillowWriter(fps=8),
              savefig_kwargs={'facecolor': fig.get_facecolor()})
    plt.close(fig)
    print(f'  roughness_animation.gif →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PSD comparison (log-log)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_psd_comparison():
    H, N, T = 0.1, 252, 1.0
    dt = T / N
    freqs, psd_ex = get_exact_fgn_psd(H, N, dt)
    psd_as = asymptotic_fgn_psd(freqs, H, psd_ex)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(freqs, psd_ex, color=C['exact'], lw=2.5,
              label='Exact fGn PSD  (Monte Carlo, 100k paths)')
    ax.loglog(freqs, psd_as, color=C['asymp'], lw=2.0, ls='--',
              label=f'Asymptotic  $C|f|^{{1-2H}} = C|f|^{{0.8}}$')

    # Reference slope annotation
    f0, f1 = freqs[5], freqs[60]
    c_ref   = psd_ex[5]
    ax.loglog([f0, f1], [c_ref, c_ref * (f1 / f0) ** (1 - 2 * H)],
              color='#999999', lw=1.2, ls=':', alpha=0.8)
    mid = (f0 * f1) ** 0.5
    ax.annotate(f'slope = {1 - 2*H:.1f}  (=1−2H)',
                xy=(mid, c_ref * (mid / f0) ** (1 - 2 * H)),
                xytext=(mid * 2, c_ref * (mid / f0) ** (1 - 2 * H) * 2.5),
                color='#999999', fontsize=10, arrowprops=dict(
                    arrowstyle='->', color='#999999', lw=1.0))

    # Show where asymptotic diverges from exact
    rel_err = np.abs(psd_as - psd_ex) / psd_ex
    ax2 = ax.inset_axes([0.55, 0.08, 0.42, 0.30])
    ax2.semilogx(freqs, rel_err * 100, color='#e74c3c', lw=1.5)
    ax2.axhline(10, color='#aaaaaa', lw=0.8, ls='--')
    ax2.set_xlabel('frequency', fontsize=9)
    ax2.set_ylabel('|asymp−exact|/exact (%)', fontsize=8)
    ax2.set_title('Relative error', fontsize=9)
    ax2.tick_params(labelsize=8)

    ax.set_xlabel('Frequency (cycles / year)')
    ax.set_ylabel('Power spectral density')
    ax.set_title(f'Fractional Gaussian Noise PSD  (H = {H},  N = {N})')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()

    fpath = FIGURES / 'psd_comparison.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  psd_comparison.png     →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  3-D IV surface
# ═══════════════════════════════════════════════════════════════════════════════

def fig_iv_surface_3d():
    data = np.load(RESULTS / 'surrogate_comparison.npz')
    K_arr, T_arr, iv_ref = data['K_arr'], data['T_arr'], data['iv_ref']
    iv_ex  = data['iv_exact']
    S0 = 1.0
    moneyness = K_arr / S0

    K_mesh, T_mesh = np.meshgrid(moneyness, T_arr)   # (n_mat, n_str)

    fig = plt.figure(figsize=(13, 5))

    for idx, (iv, title, cmap) in enumerate([
            (iv_ref, 'MC Reference', 'plasma'),
            (iv_ex,  'SpectralFilter (exact PSD)', 'cividis')]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        surf = ax.plot_surface(K_mesh, T_mesh, iv * 100,
                               cmap=cmap, alpha=0.92,
                               linewidth=0, antialiased=True,
                               vmin=14, vmax=28)
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12,
                     label='IV (%)', pad=0.12)
        ax.set_xlabel('K/S₀', labelpad=6)
        ax.set_ylabel('T (yr)', labelpad=6)
        ax.set_zlabel('IV (%)', labelpad=6)
        ax.set_zlim(14, 28)
        ax.set_title(title, pad=10)
        ax.view_init(elev=22, azim=-50)

    fig.suptitle('rBergomi Implied Volatility Surface  (H=0.1, η=1.9, ρ=0)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fpath = FIGURES / 'iv_surface_3d.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  iv_surface_3d.png      →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Surrogate smile panel  (publication quality)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_iv_surrogate_panel():
    data = np.load(RESULTS / 'surrogate_comparison.npz')
    K_arr, T_arr = data['K_arr'], data['T_arr']
    iv_ref  = data['iv_ref']
    iv_ex   = data['iv_exact']
    iv_as   = data['iv_asymp']
    iv_mlp  = data['iv_mlp']
    moneyness = K_arr   # S0=1

    # RMSE per maturity
    def rmse(a, b):
        d = (a - b) * 100
        v = np.isfinite(d)
        return np.sqrt(np.mean(d[v] ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.subplots_adjust(wspace=0.30, top=0.88, bottom=0.13)

    for i, (ax, T_val) in enumerate(zip(axes, T_arr)):
        m = np.isfinite(iv_ref[i])

        ax.plot(moneyness[m], iv_ref[i][m] * 100,
                color=C['ref'], lw=2.8, zorder=5,
                marker='o', ms=5, label='MC reference')
        ax.plot(moneyness[m], iv_ex[i][m] * 100,
                color=C['exact'], lw=1.8, ls='-', zorder=4,
                marker='s', ms=4,
                label=f'SpectralFilter exact  ({rmse(iv_ex[i], iv_ref[i]):.2f} vp)')
        ax.plot(moneyness[m], iv_as[i][m] * 100,
                color=C['asymp'], lw=1.8, ls='--', zorder=3,
                marker='^', ms=4,
                label=f'SpectralFilter asymp  ({rmse(iv_as[i], iv_ref[i]):.2f} vp)')
        ax.plot(moneyness[m], iv_mlp[i][m] * 100,
                color=C['mlp'], lw=1.8, ls=':', zorder=3,
                marker='D', ms=4,
                label=f'BochnerMLP exact        ({rmse(iv_mlp[i], iv_ref[i]):.2f} vp)')

        ax.set_xlabel('Moneyness  K/S₀')
        ax.set_title(f'T = {T_val:.2f} yr', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Implied vol (%)')
        ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.78, 1.22)

    fig.suptitle(
        'rBergomi Implied Volatility Smile — MC vs Bochner PINN Surrogates',
        fontsize=13, fontweight='bold')

    fpath = FIGURES / 'iv_surrogate_panel.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  iv_surrogate_panel.png →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  RMSE heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rmse_heatmap():
    data = np.load(RESULTS / 'surrogate_comparison.npz')
    iv_ref = data['iv_ref']
    T_arr  = data['T_arr']
    models = {
        'SpectralFilter\n(exact PSD)': data['iv_exact'],
        'SpectralFilter\n(asymptotic)': data['iv_asymp'],
        'BochnerMLP\n(exact PSD)':   data['iv_mlp'],
    }

    col_labels = [f'T={T:.2f}' for T in T_arr] + ['Overall']
    row_labels  = list(models.keys())

    mat = np.zeros((len(models), len(col_labels)))
    for r, (_, iv_s) in enumerate(models.items()):
        for c, T_i in enumerate(range(len(T_arr))):
            d = (iv_s[T_i] - iv_ref[T_i]) * 100
            v = np.isfinite(d)
            mat[r, c] = np.sqrt(np.mean(d[v] ** 2))
        # Overall
        d = (iv_s - iv_ref) * 100
        v = np.isfinite(d)
        mat[r, -1] = np.sqrt(np.mean(d[v] ** 2))

    fig, ax = plt.subplots(figsize=(9, 4))

    vmax = np.ceil(mat.max())
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=vmax, interpolation='nearest')

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            text_color = 'white' if val > vmax * 0.6 else '#111111'
            ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(col_labels)));  ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)));  ax.set_yticklabels(row_labels,
                                                                fontsize=11)

    # Highlight "Overall" column
    ax.axvline(len(col_labels) - 1.5, color='white', lw=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE (vol points)', fontsize=11)

    ax.set_title('IV Surface RMSE: Bochner PINN vs MC Reference  (vol points)',
                 fontsize=13, pad=12)
    fig.tight_layout()

    fpath = FIGURES / 'rmse_heatmap.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  rmse_heatmap.png       →  {fpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating figures...\n')

    print('[1/6] Roughness gallery...')
    fig_roughness_gallery()

    print('[2/6] Roughness animation...')
    fig_roughness_animation()

    print('[3/6] PSD comparison...')
    fig_psd_comparison()

    print('[4/6] 3-D IV surface...')
    fig_iv_surface_3d()

    print('[5/6] Surrogate smile panel...')
    fig_iv_surrogate_panel()

    print('[6/6] RMSE heatmap...')
    fig_rmse_heatmap()

    print(f'\nAll figures saved to {FIGURES}')
