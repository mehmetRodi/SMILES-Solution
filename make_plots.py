import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from task_and_baseline import (
    MAX_UNEXPLAINED_TO_RESIDUAL,
    MIN_EXPLAIN_RATIO,
    baseline,
    build_task_helpers,
)
from applicant_solution import ALPHA_A, ALPHA_B, _rank1_inband

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)
score_filter = helpers["score_filter"]
fit_tx_prediction = helpers["fit_tx_prediction"]

print("Running pipeline stages…")
tx_pred = fit_tx_prediction(rx)
rank1_a = _rank1_inband(rx - tx_pred, score_filter)
rank1_b = _rank1_inband(rx - tx_pred - rank1_a, score_filter)

stages = {
    "raw rx": rx,
    "baseline (TX only)": rx - tx_pred,
    "TX + rank-1": rx - tx_pred - rank1_a,
    "TX + 2× rank-1 (final)": rx - tx_pred - ALPHA_A * rank1_a - ALPHA_B * rank1_b,
}


def in_band_db(sig):
    p = np.mean(
        np.abs(np.column_stack([score_filter(sig[:, c]) for c in range(4)])) ** 2,
        axis=0,
    )
    return 10 * np.log10(p + 1e-30)


print("Figure 1: PSD before/after per channel")
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
nperseg = 8192
colors = {
    "raw rx": "#999999",
    "baseline (TX only)": "#2b8cbe",
    "TX + rank-1": "#fdbb84",
    "TX + 2× rank-1 (final)": "#d7301f",
}
for ch, ax in enumerate(axes.ravel()):
    for label, sig in stages.items():
        f, psd = welch(
            sig[:, ch],
            fs=Fs,
            nperseg=nperseg,
            return_onesided=False,
            detrend=False,
        )
        idx = np.argsort(f)
        ax.plot(
            f[idx] / 1e6,
            10 * np.log10(psd[idx] + 1e-30),
            label=label,
            color=colors[label],
            lw=1.1,
        )
    ax.axvspan(1.6, 2.2, color="#fee08b", alpha=0.35, label="score band")
    ax.set_title(f"RX channel {ch}")
    ax.set_xlim(-Fs / 2 / 1e6, Fs / 2 / 1e6)
    if ch >= 2:
        ax.set_xlabel("Frequency (MHz)")
    if ch % 2 == 0:
        ax.set_ylabel("PSD (dB/Hz)")
axes[0, 0].legend(loc="lower center", fontsize=8, framealpha=0.9)
fig.suptitle(
    "Power spectral density per RX channel — each stage cancels in-band content",
    fontsize=12,
)
fig.savefig(f"{FIG_DIR}/psd_per_channel.png")
plt.close(fig)

print("Figure 2: per-channel reduction by method")
methods_for_bars = [
    ("Baseline (TX only)", baseline(tx_n, rx, fit_tx_prediction)),
    ("TX + rank-1", rx - tx_pred - rank1_a),
    ("TX + 2× rank-1, α=1.0", rx - tx_pred - rank1_a - rank1_b),
    ("TX + 2× rank-1 (final)", rx - tx_pred - ALPHA_A * rank1_a - ALPHA_B * rank1_b),
]
p0 = np.array([np.mean(np.abs(score_filter(rx[:, c])) ** 2) for c in range(4)])
fig, ax = plt.subplots(figsize=(10, 4.5))
width = 0.2
x = np.arange(4)
for i, (label, sig) in enumerate(methods_for_bars):
    p1 = np.array([np.mean(np.abs(score_filter(sig[:, c])) ** 2) for c in range(4)])
    reds = 10 * np.log10(p0 / (p1 + 1e-30))
    bars = ax.bar(x + (i - 1.5) * width, reds, width, label=f"{label} (avg {np.mean(reds):.2f} dB)")
    for b, r in zip(bars, reds):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.1, f"{r:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"ch{c}" for c in range(4)])
ax.set_ylabel("In-band reduction (dB)")
ax.set_title("Per-channel cancellation by method")
ax.axhline(8.0, color="#666", lw=0.8, ls="--", alpha=0.6)
ax.text(3.45, 8.05, '"Good" tier (8 dB)', fontsize=8, color="#666")
ax.legend(loc="upper left", fontsize=8)
fig.savefig(f"{FIG_DIR}/per_channel_bars.png")
plt.close(fig)

print("Figure 3: eigenvalue spectrum of band covariance at each stage")
fig, ax = plt.subplots(figsize=(7, 4))
eig_stages = [
    ("after TX only", rx - tx_pred),
    ("after TX + rank-1_a", rx - tx_pred - rank1_a),
    ("after TX + 2× rank-1 (final)", rx - tx_pred - ALPHA_A * rank1_a - ALPHA_B * rank1_b),
]
markers = ["o", "s", "^"]
for (label, sig), m in zip(eig_stages, markers):
    band = np.column_stack([score_filter(sig[:, c]) for c in range(4)])
    cov = band.conj().T @ band / band.shape[0]
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ax.semilogy(range(1, 5), eigvals, marker=m, lw=1.5, ms=8, label=label)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Eigenvalue rank")
ax.set_ylabel("Eigenvalue (band-covariance)")
ax.set_title("In-band 4×4 covariance spectrum across stages\n(each rank-1 stage knocks out one dominant mode)")
ax.legend(fontsize=8)
fig.savefig(f"{FIG_DIR}/eigvals.png")
plt.close(fig)

print("Figure 4: α grid heatmap")


def evaluate(rx_hat):
    removed_band = np.column_stack(
        [score_filter(rx[:, c] - rx_hat[:, c]) for c in range(4)]
    )
    tx_part = fit_tx_prediction(rx - rx_hat)
    rd = removed_band - tx_part
    cov = rd.conj().T @ rd / rd.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = rd @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    rank1_part = np.column_stack(
        [(np.vdot(shared, rd[:, c]) / denom) * shared for c in range(4)]
    )
    err = rd - rank1_part
    total_pow = np.mean(np.abs(removed_band) ** 2) + 1e-30
    explain = 1.0 - np.mean(np.abs(err) ** 2) / total_pow
    residual_band = np.column_stack([score_filter(rx_hat[:, c]) for c in range(4)])
    err_pow = np.mean(np.abs(err) ** 2, axis=0)
    res_pow = np.mean(np.abs(residual_band) ** 2, axis=0) + 1e-30
    valid = explain >= MIN_EXPLAIN_RATIO and np.all(
        err_pow <= MAX_UNEXPLAINED_TO_RESIDUAL * res_pow
    )
    if not valid:
        return 0.0
    return float(np.mean(10 * np.log10(p0 / res_pow)))


a_vals = np.linspace(0.5, 1.3, 9)
b_vals = np.linspace(0.5, 1.3, 9)
grid = np.zeros((len(a_vals), len(b_vals)))
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        grid[i, j] = evaluate(rx - tx_pred - a * rank1_a - b * rank1_b)
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(
    grid,
    origin="lower",
    extent=(b_vals[0], b_vals[-1], a_vals[0], a_vals[-1]),
    aspect="auto",
    cmap="viridis",
)
ax.set_xlabel(r"$\alpha_b$ (second rank-1 scaling)")
ax.set_ylabel(r"$\alpha_a$ (first rank-1 scaling)")
ax.set_title("Score (dB) over the (α_a, α_b) plane\nblack = INVALID per scorer's validity check")
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        v = grid[i, j]
        col = "white" if v < grid.max() * 0.6 else "black"
        ax.text(b, a, f"{v:.1f}" if v > 0 else "✗", ha="center", va="center", color=col, fontsize=7)
ax.scatter([ALPHA_B], [ALPHA_A], marker="*", s=220, color="red", edgecolor="white", lw=1.2, label="chosen (0.85, 0.95)")
ax.legend(loc="lower left")
fig.colorbar(im, ax=ax, label="avg in-band reduction (dB)")
fig.savefig(f"{FIG_DIR}/alpha_grid.png")
plt.close(fig)

print("Figure 5: validity margins at each stage")
margin_stages = [
    ("TX only", rx - tx_pred),
    ("TX + rank-1", rx - tx_pred - rank1_a),
    ("TX + 2× rank-1 (α=1.0)", rx - tx_pred - rank1_a - rank1_b),
    ("TX + 2× rank-1 (final)", rx - tx_pred - ALPHA_A * rank1_a - ALPHA_B * rank1_b),
]
exp_ratios = []
worst_ratios = []
scores = []
labels = []
for label, sig in margin_stages:
    removed_band = np.column_stack(
        [score_filter(rx[:, c] - sig[:, c]) for c in range(4)]
    )
    tx_part = fit_tx_prediction(rx - sig)
    rd = removed_band - tx_part
    cov = rd.conj().T @ rd / rd.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = rd @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    rank1_part = np.column_stack(
        [(np.vdot(shared, rd[:, c]) / denom) * shared for c in range(4)]
    )
    err = rd - rank1_part
    total_pow = np.mean(np.abs(removed_band) ** 2) + 1e-30
    exp_ratios.append(1.0 - np.mean(np.abs(err) ** 2) / total_pow)
    residual_band = np.column_stack([score_filter(sig[:, c]) for c in range(4)])
    err_pow = np.mean(np.abs(err) ** 2, axis=0)
    res_pow = np.mean(np.abs(residual_band) ** 2, axis=0) + 1e-30
    worst_ratios.append(float(np.max(err_pow / res_pow)))
    scores.append(float(np.mean(10 * np.log10(p0 / res_pow))))
    labels.append(label)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
xpos = np.arange(len(labels))
axes[0].bar(xpos, scores, color="#2b8cbe")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
axes[0].axhline(8.0, color="#888", ls="--", lw=0.8)
axes[0].set_ylabel("Average reduction (dB)")
axes[0].set_title("Score")
for x, s in zip(xpos, scores):
    axes[0].text(x, s + 0.1, f"{s:.2f}", ha="center", fontsize=8)

axes[1].bar(xpos, exp_ratios, color="#fdbb84")
axes[1].axhline(MIN_EXPLAIN_RATIO, color="red", ls="--", lw=1, label=f"min {MIN_EXPLAIN_RATIO}")
axes[1].set_xticks(xpos)
axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
axes[1].set_ylim(0.9, 1.005)
axes[1].set_ylabel("explain_ratio")
axes[1].set_title("Explainability (≥ 0.95)")
axes[1].legend(fontsize=8)
for x, s in zip(xpos, exp_ratios):
    axes[1].text(x, s + 0.001, f"{s:.3f}", ha="center", fontsize=8)

axes[2].bar(xpos, worst_ratios, color="#d7301f")
axes[2].axhline(MAX_UNEXPLAINED_TO_RESIDUAL, color="red", ls="--", lw=1, label=f"max {MAX_UNEXPLAINED_TO_RESIDUAL}")
axes[2].set_xticks(xpos)
axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
axes[2].set_ylabel("worst-channel err / residual")
axes[2].set_title("Residual guard (≤ 0.80)")
axes[2].legend(fontsize=8)
for x, s in zip(xpos, worst_ratios):
    axes[2].text(x, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
fig.suptitle("Score and validity margins at each cancellation stage", fontsize=12)
fig.savefig(f"{FIG_DIR}/validity_margins.png")
plt.close(fig)

print(f"All figures written to {FIG_DIR}/")
