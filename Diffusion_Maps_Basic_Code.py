import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


# ============================================================
# 1. CONFIGURATION — CHANGE P HERE
# ============================================================

P = np.inf                 # Examples: 1, 1.5, 2, 3, 4, np.inf
DATA_FILE = "ctap_dataset.npz"

CHANNELS = {
    "QS_anticorrelated": "Non-Markovian anticorrelated",
    "QS_correlated":     "Non-Markovian correlated",
    "white_correlated":  "Markovian correlated",
}

N_DC = 3
N_CLUSTERS = len(CHANNELS)
DIFFUSION_TIME = 1
SEED = 11

if not P >= 1:
    raise ValueError("Choose a Minkowski norm with p >= 1 or p = np.inf.")

p_text = "∞" if np.isinf(P) else f"{P:g}"


# ============================================================
# 2. FILTER THE DATA BEFORE BUILDING THE KERNEL
# ============================================================

with np.load(DATA_FILE) as data:
    keep = np.isin(data["label5"], list(CHANNELS))
    X = data["X"][keep]
    labels = data["label5"][keep]

print(f"Minkowski p = {p_text}")
print("Feature matrix shape:", X.shape)
print(pd.Series(labels).value_counts().to_string())


# ============================================================
# 3. ALL-PAIRS MINKOWSKI DISTANCES
# ============================================================

# Finite p: d(i,j) = (sum_a |X[i,a] - X[j,a]|**p)**(1/p)
# p = inf:  d(i,j) = max_a |X[i,a] - X[j,a]|
if np.isinf(P):
    pair_dist = pdist(X, metric="chebyshev")
else:
    pair_dist = pdist(X, metric="minkowski", p=P)

# For every p, keep the same kernel convention: exp(-d_p**2 / epsilon).
dist2 = squareform(pair_dist) ** 2


# ============================================================
# 4. DIFFUSION MAP + K-MEANS FOR ONE EPSILON
# ============================================================

def fit_at_epsilon(epsilon):
    # Full affinity matrix, with one entry for every pair of samples.
    S = np.exp(-dist2 / epsilon)

    # Density correction with alpha = 1.
    q = S.sum(axis=1)
    S /= q[:, None]
    S /= q[None, :]

    # Symmetric matrix similar to the row-stochastic diffusion matrix.
    root_d = np.sqrt(S.sum(axis=1))
    S /= root_d[:, None]
    S /= root_d[None, :]

    # Stationary mode + three retained modes + one extra for the gap check.
    n = len(X)
    n_eigs = N_DC + 2
    lam, vec = eigh(
        S,
        subset_by_index=[n - n_eigs, n - 1],
        check_finite=False,
    )
    lam = lam[::-1]
    vec = vec[:, ::-1]

    # Recover right eigenvectors and drop the stationary mode.
    retained = slice(1, N_DC + 1)
    psi = vec[:, retained] / root_d[:, None]
    psi /= np.linalg.norm(psi, axis=0)

    dc = psi * lam[retained] ** DIFFUSION_TIME

    # K-means uses all three diffusion coordinates.
    groups = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=30,
        random_state=SEED,
    ).fit_predict(dc)

    # Check that the stationary mode and the DC3–DC4 boundary are resolved.
    gap = min(lam[0] - lam[1], lam[N_DC] - lam[N_DC + 1])

    return dc, groups, gap


# ============================================================
# 5. COARSE + FINE EPSILON SEARCH USING THE PHYSICAL LABELS
# ============================================================

scores = {}


def sweep(epsilon_grid):
    for eps in epsilon_grid:
        if eps in scores:
            continue

        _, groups, gap = fit_at_epsilon(eps)
        ari = adjusted_rand_score(labels, groups)
        scores[eps] = (ari, gap)


def best_epsilon():
    eligible = [eps for eps, (_, gap) in scores.items() if gap > 1e-12]
    if not eligible:
        raise ValueError("No numerically resolved bandwidth. Extend the grid.")

    # Highest ARI first. For exact ties, prefer the larger boundary gap.
    return max(eligible, key=lambda eps: scores[eps])


with threadpool_limits(limits=1):
    coarse_grid = np.geomspace(1e-5, 1.0, 31)
    sweep(coarse_grid)

    i = np.searchsorted(coarse_grid, best_epsilon())
    lo = coarse_grid[max(0, i - 1)]
    hi = coarse_grid[min(len(coarse_grid) - 1, i + 1)]
    sweep(np.geomspace(lo, hi, 21))

    epsilon = best_epsilon()
    dc, groups, _ = fit_at_epsilon(epsilon)

ari = adjusted_rand_score(labels, groups)
cluster_id = groups + 1


# ============================================================
# 6. CLUSTER TABLE, MATCHED CONFUSION MATRIX, AND ACCURACY
# ============================================================

channel_order = list(CHANNELS)

counts = pd.crosstab(
    pd.Series(labels, name="True channel"),
    pd.Series(cluster_id, name="Cluster"),
).reindex(
    index=channel_order,
    columns=range(1, N_CLUSTERS + 1),
    fill_value=0,
)

# Match arbitrary cluster IDs to distinct physical channels.
matrix = counts.to_numpy()
rows, cols = linear_sum_assignment(-matrix)

cluster_to_channel = {
    counts.columns[c]: counts.index[r]
    for r, c in zip(rows, cols)
}
predicted_labels = pd.Series(cluster_id).map(cluster_to_channel).to_numpy()

confusion = pd.crosstab(
    pd.Series(labels, name="True channel"),
    pd.Series(predicted_labels, name="Predicted channel"),
).reindex(index=channel_order, columns=channel_order, fill_value=0)

correct = np.count_nonzero(predicted_labels == labels)
accuracy = correct / len(labels)

print(f"\nBest tested epsilon = {epsilon:.10g}")
print(f"ARI                 = {ari:.4f}")
print(f"Matched accuracy    = {correct}/{len(labels)} = {accuracy:.2%}")

print("\nRaw cluster table:")
print(counts.to_string())

print("\nMatched confusion matrix (rows = true, columns = predicted):")
print(confusion.to_string())


# ============================================================
# 7. PLOT ARI VERSUS EPSILON
# ============================================================

scan = pd.DataFrame.from_dict(
    scores, orient="index", columns=["ARI", "gap"]
).sort_index()

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.semilogx(scan.index, scan["ARI"], ".-", lw=1)
ax.scatter(
    epsilon, ari, color="crimson", zorder=3,
    label=f"Selected ε = {epsilon:.3g}",
)
ax.set(
    xlabel="Gaussian kernel bandwidth ε",
    ylabel="Adjusted Rand index",
    title=f"Three-channel bandwidth search · Minkowski p = {p_text}",
)
ax.legend()
plt.show()


# ============================================================
# 8. PLOT DC1, DC2, DC3 WITH MATCHED COLORS
# ============================================================

colors = {
    "QS_anticorrelated": "#D55E00",
    "QS_correlated":     "#0072B2",
    "white_correlated":  "#009E73",
}

fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)

for panel, (values, title) in enumerate([
    (labels, "Physical noise channels"),
    (predicted_labels, f"Matched clusters · accuracy = {accuracy:.2%}"),
], start=1):
    ax = fig.add_subplot(1, 2, panel, projection="3d")

    for key, name in CHANNELS.items():
        mask = values == key
        ax.scatter(
            *dc[mask].T,
            s=15, alpha=0.7,
            color=colors[key], label=name,
        )

    ax.set(xlabel="DC1", ylabel="DC2", zlabel="DC3", title=title)
    ax.view_init(elev=22, azim=-55)
    ax.legend(fontsize=8)

fig.suptitle(f"Minkowski p = {p_text} · ε = {epsilon:.3g}")
plt.show()