"""
Per-figure rendering for the reference-free deployment protocol (thesis).

Reads the CSVs written by DeploymentProtocol._dump() and renders each as a
matplotlib figure, written to FIG_DIR as vector PDF (for LaTeX) + PNG (for
VS Code markdown preview). Nothing here reads a ground-truth transcription:
the inputs are the deployment protocol's own reference-free tables. A true
ranking may be passed in explicitly to `render_all(true_rank=...)`, in which
case it is drawn as an annotation and never used to compute anything.

Figure order:
  0  ranking              reference-free ordering, WER against the consensus
  1  filtration           f-bar per model, spread across subsets
  2  rank distribution    P(model at rank r) heatmap
  3  confidence           p_at_full_rank + normalized entropy per model
  4  pair stability       p_supports_full, ordered, threshold-marked
  5  pool size            weight and entropy by k
  6  cross-WER            pairwise hypothesis disagreement matrix
  7  configuration        two runs side by side (lineage applied vs suspended)

Usage:
    import deployment_analysis as da
    da.render_all("deployment_out/rdi_voters", suffix="rdi_voters",
                  fig_dir="figures/deploy_rdi", prefix="rdi_")

    da.figure7_configs("deployment_out/rdi_voters", "deployment_out/rdi_all_voting",
                       suffix_a="rdi_voters", suffix_b="rdi_all_voting",
                       label_a="lineage rule applied", label_b="lineage rule suspended")
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- config -------------------------------------------------------
FIG_DIR = "figures"
FORMATS = ("pdf", "png")
DPI = 300
FS = 9.0
PAD = 7.0

C_ROW = "#dde5f0"; C_HDR = "#f7f7f7"; C_W = "#ffffff"
BLUE = "#3b6fb0"; GREEN = "#3f8b57"; RED = "#b4413e"; GREY = "#8a8a8a"
INK = "#222222"; LINE = "#b8b8b8"
AMBER = "#c98a2e"


def set_output(fig_dir=FIG_DIR, formats=FORMATS, dpi=DPI):
    global FIG_DIR, FORMATS, DPI
    FIG_DIR = fig_dir; FORMATS = tuple(formats); DPI = dpi
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = []
    for ext in FORMATS:
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        out.append(p)
    plt.close(fig)
    print("saved: " + " | ".join(out))
    return out


def _rc():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
                         "grid.linewidth": 0.6, "font.size": FS, "axes.titlesize": FS + 2,
                         "axes.titleweight": "600", "axes.labelsize": FS + 0.5})


# ---- loading ------------------------------------------------------

_INDEXED = ("cross_wer", "rank_distribution")


def load(indir, suffix="", names=None):
    """Load the deployment tables from a directory. Missing files are skipped."""
    sfx = f"_{suffix.strip('_')}" if suffix else ""
    want = names or ("ranking", "model_filtration", "rank_distribution", "model_confidence",
                     "pair_probability", "by_pool_size", "cross_wer",
                     "subpool_filtration", "subpool_ranks")
    out = {}
    for n in want:
        p = os.path.join(indir, f"{n}{sfx}.csv")
        if not os.path.exists(p):
            warnings.warn(f"{n}{sfx}.csv not found in {indir}")
            continue
        out[n] = pd.read_csv(p, index_col=0 if n in _INDEXED else None)
    if not out:
        raise ValueError(f"no deployment tables found in {indir}")
    return out


def _order(conf):
    """Models in reported-ranking order."""
    return list(conf.sort_values("rank_pseudo_full")["model"])


def _lum(rgba):
    r, g, b, _ = rgba
    return 0.299 * r + 0.587 * g + 0.114 * b


# ==== figure 0: the reference-free ranking ==========================

def figure0_ranking(t, true_rank=None, name="deploy0_ranking"):
    """WER against the leave-one-out consensus, in reported order. `true_rank`,
    if given, is drawn as an annotation only."""
    _rc()
    d = t["ranking"].sort_values("rank_pseudo").reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.9 * n + 2.4), 3.9))
    x = np.arange(n)
    ax.bar(x, d.wer_pseudo_loo, 0.62, color=GREY, edgecolor="white")
    for i, v in enumerate(d.wer_pseudo_loo):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=FS - 2)
    ax.set_xticks(x)
    ax.set_xticklabels(d.model, rotation=38, ha="right", fontsize=FS - 1)
    ax.set_ylabel("WER against consensus (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.22, lw=.6)
    sub = "leave-one-out consensus per system; no human reference used"
    if true_rank is not None:
        agree = sum(1 for m in d.model if true_rank.get(m) == int(d.loc[d.model == m, "rank_pseudo"].iloc[0]))
        for i, m in enumerate(d.model):
            tr = true_rank.get(m)
            if tr is None:
                continue
            ok = tr == int(d.loc[d.model == m, "rank_pseudo"].iloc[0])
            ax.text(i, -0.045 * d.wer_pseudo_loo.max(), f"{tr}", ha="center", va="top",
                    fontsize=FS - 2, color=GREEN if ok else RED, fontweight="bold")
        sub += f"   |   annotations = reference-based rank ({agree}/{n} match)"
    ax.set_title("Deployment 0 - reference-free ranking", loc="left", pad=24)
    ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=FS - 1.4, color="#555", va="bottom")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 1: filtration ==========================================

def figure1_filtration(t, name="deploy1_filtration"):
    """Mean exclusion rate per model with its range across subsets. A model's
    filtration is a property of the pool it sits in, so the spread matters as
    much as the mean; the weighting uses the mean."""
    _rc()
    d = t["model_filtration"].sort_values("f_bar").reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * n + 2.4))
    y = np.arange(n)
    ax.hlines(y, d.f_min, d.f_max, color="#d6d6d6", lw=3.2, zorder=1)
    ax.scatter(d.f_bar, y, s=66, color=BLUE, zorder=3, edgecolor="white", lw=.9)
    for i in range(n):
        ax.text(d.f_max[i] + 0.015, y[i], f"{d.f_bar[i]:.3f}", va="center",
                fontsize=FS - 2, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels(d.model, fontsize=FS - 0.5)
    ax.set_xlabel(r"exclusion rate  $\bar{f}_m$  (bar = range over subsets)")
    ax.set_xlim(-0.02, min(1.02, d.f_max.max() + 0.14))
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.22, lw=.6)
    ax.set_title("Deployment 1 - filtration by model", loc="left", pad=24)
    ax.text(0, 1.02, f"averaged over the {int(d.n_subpools.iloc[0])} subsets containing each model",
            transform=ax.transAxes, fontsize=FS - 1.4, color="#555", va="bottom")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 2: rank distribution ===================================

def figure2_rank_distribution(t, name="deploy2_rank_distribution"):
    """P(model at rank r) over the enumeration of subsets with an utterance
    bootstrap inside each. Rows sum to one; the reported rank is outlined."""
    _rc()
    dist = t["rank_distribution"]
    conf = t["model_confidence"]
    order = _order(conf)
    dist = dist.loc[order]
    M = dist.values.astype(float)
    n, k = M.shape
    fig, ax = plt.subplots(figsize=(0.68 * k + 3.4, 0.46 * n + 2.6))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    rf = conf.set_index("model")["rank_pseudo_full"].to_dict()
    for i, m in enumerate(order):
        for j in range(k):
            if M[i, j] >= 0.005:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=FS - 2.4,
                        color=INK if _lum(cmap(M[i, j])) > 0.6 else "white")
        ax.add_patch(Rectangle((rf[m] - 1.5, i - .5), 1, 1, fill=False, edgecolor=RED, lw=2.0))
    ax.set_xticks(range(k))
    ax.set_xticklabels([c.replace("rank_", "") for c in dist.columns], fontsize=FS - 1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=FS - 1)
    ax.set_xlabel("rank")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="P(rank)")
    ax.set_title("Deployment 2 - rank distribution under pool perturbation", loc="left", pad=24)
    ax.text(0, 1.02, "outline = rank reported by the full pool; rows sum to one",
            transform=ax.transAxes, fontsize=FS - 1.4, color="#555", va="bottom")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 3: confidence and uncertainty ==========================

def figure3_confidence(t, true_rank=None, name="deploy3_confidence"):
    """Stability of the reported rank, and the entropy of the distribution it
    came from. These are stability under pool perturbation, not a probability
    that the ranking is correct."""
    _rc()
    conf = t["model_confidence"].sort_values("rank_pseudo_full").reset_index(drop=True)
    n = len(conf)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.42 * n + 2.8), sharey=True)
    y = np.arange(n)[::-1]

    ax = axes[0]
    cols = [GREY] * n
    if true_rank is not None:
        cols = [GREEN if true_rank.get(m) == r else RED
                for m, r in zip(conf.model, conf.rank_pseudo_full)]
    ax.barh(y, conf.p_at_full_rank, 0.62, color=cols, edgecolor="white")
    for i in range(n):
        ax.text(conf.p_at_full_rank[i] + .012, y[i], f"{conf.p_at_full_rank[i]:.3f}",
                va="center", fontsize=FS - 2, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  (#{r})" for m, r in zip(conf.model, conf.rank_pseudo_full)],
                       fontsize=FS - 1)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("P(reported rank)")
    ax.set_axisbelow(True); ax.grid(axis="x", alpha=.22, lw=.6)
    ax.set_title("(a) rank stability", loc="left", pad=8)

    ax = axes[1]
    ax.barh(y, conf.entropy_norm.clip(lower=0), 0.62, color=BLUE, edgecolor="white")
    for i in range(n):
        ax.text(max(conf.entropy_norm[i], 0) + .012, y[i], f"{max(conf.entropy_norm[i],0):.3f}",
                va="center", fontsize=FS - 2, color="#555")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("normalized entropy of the rank distribution")
    ax.set_axisbelow(True); ax.grid(axis="x", alpha=.22, lw=.6)
    ax.set_title("(b) rank uncertainty   [0 = position settled]", loc="left", pad=8)

    sub = "stability under perturbation of the model pool; not a probability of correctness"
    if true_rank is not None:
        sub += "   |   bar colour = reported rank matches the reference-based rank"
    fig.suptitle("Deployment 3 - confidence in the reported ranking",
                 y=1.075, fontsize=FS + 3, fontweight="bold", x=0.005, ha="left")
    fig.text(0.005, 1.028, sub, fontsize=FS - 1.4, color="#555", ha="left", va="top")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 4: pairwise stability ==================================

def figure4_pair_stability(t, top=None, bar=0.95, name="deploy4_pair_stability"):
    """Fraction of replicates preserving each pair's reported ordering. The
    analogue of the utterance bootstrap's preserved fraction, but with the
    reference varied rather than the sample."""
    _rc()
    d = t["pair_probability"].sort_values("p_supports_full").reset_index(drop=True)
    if top:
        d = d.head(top)
    n = len(d)
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * n + 2.4))
    y = np.arange(n)[::-1]
    cols = [RED if v < 0.5 else (AMBER if v < bar else GREY) for v in d.p_supports_full]
    ax.barh(y, d.p_supports_full, 0.66, color=cols, edgecolor="white")
    ax.axvline(bar, color=BLUE, ls="--", lw=1.2)
    ax.axvline(0.5, color=RED, ls=":", lw=1.0)
    for i in range(n):
        ax.text(d.p_supports_full[i] + .006, y[i], f"{d.p_supports_full[i]:.3f}",
                va="center", fontsize=FS - 2.4, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("|", " vs ") for p in d.pair], fontsize=FS - 2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("P(reported ordering preserved)")
    ax.set_axisbelow(True); ax.grid(axis="x", alpha=.22, lw=.6)
    ax.text(bar, -0.8, f"{bar:g}", color=BLUE, fontsize=FS - 2, ha="center", va="top")
    ax.set_title("Deployment 4 - pairwise stability under pool perturbation", loc="left", pad=24)
    ax.text(0, 1.016, "below 0.5 the perturbations more often give the opposite order",
            transform=ax.transAxes, fontsize=FS - 1.4, color="#555", va="bottom")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 5: pool size ===========================================

def figure5_pool_size(t, name="deploy5_pool_size"):
    """How weight and uncertainty distribute over subset size. The weighting is
    derived from the filter's own decisions, so its influence scales with how
    much the filter is firing."""
    _rc()
    d = t["by_pool_size"].sort_values("k").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))

    ax = axes[0]
    ax.bar(d.k, d.total_weight, 0.56, color=BLUE, edgecolor="white")
    unif = d.pools / d.pools.sum()
    ax.plot(d.k, unif, "o--", color=GREY, lw=1.2, ms=5, label="uniform")
    for i in range(len(d)):
        ax.text(d.k[i], d.total_weight[i], f"{d.total_weight[i]:.3f}",
                ha="center", va="bottom", fontsize=FS - 2.4)
    ax.set_xticks(d.k)
    ax.set_xlabel("subset size k"); ax.set_ylabel("total weight")
    ax.legend(frameon=False, fontsize=FS - 2)
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_title("(a) weight by subset size", loc="left", pad=8)

    ax = axes[1]
    ax.plot(d.k, d.mean_entropy, "o-", color=BLUE, lw=1.5, ms=5, label="mean")
    ax.plot(d.k, d.max_entropy, "s--", color=GREY, lw=1.1, ms=4, label="max")
    ax.set_xticks(d.k)
    ax.set_xlabel("subset size k"); ax.set_ylabel("normalized entropy")
    ax.legend(frameon=False, fontsize=FS - 2)
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_title("(b) uncertainty by subset size", loc="left", pad=8)

    fig.suptitle("Deployment 5 - subset size", y=1.03, fontsize=FS + 3,
                 fontweight="bold", x=0.005, ha="left")
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 6: cross-WER ===========================================

def figure6_cross_wer(t, name="deploy6_cross_wer"):
    """Pairwise disagreement between hypotheses. One system's output serves as
    the metric's reference for another's; no human transcript is involved."""
    _rc()
    cw = t["cross_wer"]
    names = list(cw.index)
    M = cw.values.astype(float)
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.66 * n + 3.6, 0.56 * n + 2.6))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=np.nanmax(M))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j] / max(np.nanmax(M), 1e-9)
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=FS - 2.4,
                    color=INK if _lum(cmap(v)) > 0.6 else "white")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=42, ha="right", fontsize=FS - 2)
    ax.set_yticklabels(names, fontsize=FS - 2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="cross-WER (%)")
    ax.set_title("Deployment 6 - pairwise hypothesis disagreement", loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, name)


# ==== figure 7: configuration comparison ============================

def figure7_configs(dir_a, dir_b, suffix_a="", suffix_b="",
                    label_a="configuration A", label_b="configuration B",
                    true_rank=None, fig_dir=None, name="deploy7_configs"):
    """Two deployment configurations side by side, typically the lineage rule
    applied against the same pool with it suspended. Rank counts differ between
    configurations, so the entropies are not directly comparable in magnitude;
    the orderings and the reported-rank stabilities are."""
    if fig_dir:
        set_output(fig_dir, FORMATS, DPI)
    _rc()
    a = load(dir_a, suffix_a, ("model_confidence",))["model_confidence"]
    b = load(dir_b, suffix_b, ("model_confidence",))["model_confidence"]
    a = a.sort_values("rank_pseudo_full"); b = b.sort_values("rank_pseudo_full")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 0.42 * max(len(a), len(b)) + 3.0))
    for ax, d, lab in ((axes[0], a, label_a), (axes[1], b, label_b)):
        n = len(d); y = np.arange(n)[::-1]
        cols = [GREY] * n
        if true_rank is not None:
            cols = [GREEN if true_rank.get(m) == r else RED
                    for m, r in zip(d.model, d.rank_pseudo_full)]
        ax.barh(y, d.p_at_full_rank, 0.62, color=cols, edgecolor="white")
        for i, (v, m, r) in enumerate(zip(d.p_at_full_rank, d.model, d.rank_pseudo_full)):
            ax.text(v + .012, y[i], f"{v:.3f}", va="center", fontsize=FS - 2.4, color="#555")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m}  (#{r})" for m, r in zip(d.model, d.rank_pseudo_full)],
                           fontsize=FS - 1.5)
        ax.set_xlim(0, 1.16)
        ax.set_xlabel("P(reported rank)")
        ax.set_axisbelow(True); ax.grid(axis="x", alpha=.22, lw=.6)
        agree = int(d.agrees_with_full.sum()) if "agrees_with_full" in d else n
        ax.set_title(f"{lab}   [{n} systems, modal matches reported on {agree}]",
                     loc="left", pad=8, fontsize=FS + 0.5)
    fig.suptitle("Deployment 7 - configuration comparison", y=1.04,
                 fontsize=FS + 3, fontweight="bold", x=0.005, ha="left")
    fig.text(0.005, 0.995,
             "rank counts differ between configurations, so stabilities are comparable "
             "in ordering rather than in magnitude",
             fontsize=FS - 1.4, color="#555", ha="left", va="top")
    fig.tight_layout()
    return _save(fig, name)


# ==== driver ========================================================

def render_all(indir, suffix="", fig_dir=FIG_DIR, formats=FORMATS, dpi=DPI,
               prefix="", true_rank=None, pair_top=None):
    """Render every deployment figure from one output directory.

    `true_rank` is an optional {model: rank} mapping from the evaluation run. It
    is drawn as an annotation only; nothing here computes with it.
    """
    set_output(fig_dir, formats, dpi)
    t = load(indir, suffix)
    p = []
    if "ranking" in t:
        p += figure0_ranking(t, true_rank, prefix + "deploy0_ranking")
    if "model_filtration" in t:
        p += figure1_filtration(t, prefix + "deploy1_filtration")
    if "rank_distribution" in t and "model_confidence" in t:
        p += figure2_rank_distribution(t, prefix + "deploy2_rank_distribution")
    if "model_confidence" in t:
        p += figure3_confidence(t, true_rank, prefix + "deploy3_confidence")
    if "pair_probability" in t:
        p += figure4_pair_stability(t, pair_top, 0.95, prefix + "deploy4_pair_stability")
    if "by_pool_size" in t:
        p += figure5_pool_size(t, prefix + "deploy5_pool_size")
    if "cross_wer" in t:
        p += figure6_cross_wer(t, prefix + "deploy6_cross_wer")
    return p
