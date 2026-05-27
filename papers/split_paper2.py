"""
split_paper2.py — Split the unified rough vol paper into 2a and 2b.

Paper 2a: Spectral Surrogates for rBergomi
  - Sections 1-5.4 (through Runtime comparison)
  - Section 5.6 (CausalSpectralFilter for rBergomi: BLP + matched PSD)
  - Excludes rough Heston content, VolterraSurrogate, SPY calibration
  - Trimmed Discussion and Conclusion

Paper 2b: Closing the Cross-Model Gap
  - New short intro referencing Paper 2a
  - Section 5.5 (Leverage effect + rough Heston)
  - Cross-model parts of 5.6 (rough Heston transfer)
  - VolterraSurrogate sections
  - SPY calibration
  - Markovian speedup
  - Trimmed Discussion and Conclusion
"""

src = "/home/hubi/spde/papers/paper2a_cramer_rbergomi/main.tex"

with open(src) as f:
    lines = f.readlines()

# ── Paper 2a: everything except section 5.5 ──────────────────────────────
paper2a = []
skip_until_section_end = False
for i, line in enumerate(lines):
    # Skip Section 5.5 (Leverage effect) entirely
    if i >= 926 and i < 1105:  # lines 927-1105
        continue
    
    # After CausalSpectralFilter section ends (~line 1670), skip Volterra/SPY content
    # The Discussion starts at line 1673
    if i >= 1106 and i < 1673:
        # Keep the rBergomi-only parts of Section 5.6
        # Skip the cross-model rough Heston paragraph and Volterra prototype
        if "Cross-model: causal $z$-Cholesky on rough Heston" in line:
            skip_until_section_end = True
        if skip_until_section_end:
            if line.strip().startswith("\\paragraph{") and "Cross-model" not in line:
                skip_until_section_end = False
                paper2a.append(line)
            continue
    
    # Skip Volterra-aware prototype and everything after
    if "Volterra-aware prototype closes" in line:
        skip_until_section_end = True
    if skip_until_section_end and i >= 1673:
        # Reached Discussion — stop skipping
        skip_until_section_end = False
    
    if not skip_until_section_end:
        paper2a.append(line)

# Write Paper 2a
with open("/home/hubi/spde/papers/paper2a_cramer_rbergomi/main.tex", "w") as f:
    f.writelines(paper2a)

print(f"Paper 2a: {len(paper2a)} lines (from {len(lines)})")

# ── Paper 2b: Section 5.5 + cross-model + Volterra + SPY ──────────────────
paper2b_header = []
paper2b_body = []

# Extract preamble and abstract
in_preamble = True
for line in lines:
    if in_preamble:
        paper2b_header.append(line)
        if "\\end{abstract}" in line:
            paper2b_header.append("\n% ═══════════════════════════════════════════\n")
            paper2b_header.append("\\section{Introduction}\n")
            paper2b_header.append("\\label{sec:intro}\n")
            paper2b_header.append("\n")
            paper2b_header.append("This paper is the second part of a study on spectral surrogates for\n")
            paper2b_header.append("rough volatility. Paper~2a \\citep{TODO-paper2a} introduced the\n")
            paper2b_header.append("Cramér PINN, a neural-network surrogate for the rough Bergomi (rBergomi)\n")
            paper2b_header.append("model trained to match the exact PSD of fractional Gaussian noise.\n")
            paper2b_header.append("It established a $W_2$ convergence theorem, validated SpectralFilter and\n")
            paper2b_header.append("CausalSpectralFilter at $\\rho=0$ and $\\rho=-0.9$, and demonstrated\n")
            paper2b_header.append("that matching the training PSD to the reference simulator eliminates\n")
            paper2b_header.append("residual delta bias to Monte Carlo noise.\n")
            paper2b_header.append("\n")
            paper2b_header.append("In this second part, we address the cross-model problem: the CausalSpectralFilter\n")
            paper2b_header.append("resolves $\\rho\\neq 0$ within rBergomi, but when the reference process is\n")
            paper2b_header.append("rough Heston (with mean-reverting $V$ dynamics), a residual delta bias of\n")
            paper2b_header.append("$-0.15$ remains. We introduce the VolterraSurrogate, an autoregressive\n")
            paper2b_header.append("architecture with $\\sqrt{V_j}$ diffusion, and show that it closes the\n")
            paper2b_header.append("cross-model gap to Monte Carlo noise ($0.80$~vp, delta bias $\\leq 0.005$).\n")
            paper2b_header.append("End-to-end training from misspecified initialisation, Markovian kernel\n")
            paper2b_header.append("approximation ($7\\times$ speedup), and live SPY calibration ($3.97$~vp,\n")
            paper2b_header.append("$3$~minutes on a single GPU) confirm the method's practical viability.\n")
            paper2b_header.append("\n")
            paper2b_header.append("\\tableofcontents\n")
            paper2b_header.append("\\newpage\n")
            in_preamble = False
        continue

# Body: Section 5.5 through the end
found_sec55 = False
for line in lines:
    if "\\subsection{Leverage effect and Greek bias" in line:
        found_sec55 = True
    if found_sec55:
        paper2b_body.append(line)

# Write Paper 2b
with open("/home/hubi/spde/papers/paper2b_volterra_crossmodel/main.tex", "w") as f:
    f.writelines(paper2b_header)
    f.writelines(paper2b_body)

print(f"Paper 2b: {len(paper2b_header)} header + {len(paper2b_body)} body lines")
print("Done.")
