import json
import os

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

downloaded_file = "challenge.mat"
if not os.path.exists(downloaded_file):
    import gdown

    file_id = "1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4"
    gdown.download(id=file_id, output=downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)

ALPHA_A = 0.85
ALPHA_B = 0.95


def _rank1_inband(residual, score_filter):
    band = np.column_stack(
        [score_filter(residual[:, c]) for c in range(residual.shape[1])]
    )
    cov = band.conj().T @ band / band.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared = band @ v
    denom = np.vdot(shared, shared).real + 1e-30
    coefs = np.array(
        [np.vdot(shared, band[:, c]) / denom for c in range(band.shape[1])]
    )
    return np.outer(shared, coefs)


def your_canceller(tx_n, rx):
    score_filter = helpers["score_filter"]
    fit_tx_prediction = helpers["fit_tx_prediction"]
    tx_pred = fit_tx_prediction(rx)
    rank1_a = _rank1_inband(rx - tx_pred, score_filter)
    rank1_b = _rank1_inband(rx - tx_pred - rank1_a, score_filter)
    return rx - tx_pred - ALPHA_A * rank1_a - ALPHA_B * rank1_b


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
