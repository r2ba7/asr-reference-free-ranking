#!/usr/bin/env python3
"""M2: does the rank-retention warning weaken as a correlated pair's share of the
surviving vote rises?

Reads the reference-free outputs of a suspended-rule deployment run and, over the
subsets that seat BOTH members of a documented lineage, reports how the reported
ordering holds up as that pair's share of the expected surviving vote grows.

    share(S) = sum over pair members of (1 - f_bar)
               / sum over all members of (1 - f_bar)

f_bar is the model-level mean filtration rate the weighting already uses, so this
reuses a quantity the protocol computes rather than introducing a new one.

Three controls, because share is partly a proxy for other things:

  * within-k       share is correlated with pool size; holding k fixed asks
                   whether composition matters independently of size.
  * focus-member   high-share subsets tend to exclude the strongest system;
                   splitting on its membership separates the two mechanisms.
  * control pair   an accuracy-matched pair that shares no documented lineage.
                   If the lineage curve is not steeper than the control's, the
                   effect is general small-pool instability, not lineage.

Usage:
  python m2_stratify.py <deployment_out_dir> -s SUFFIX
                        [--pair whisper_v3 whisper_v2]
                        [--control fastconformer seamless_m4t]
                        [--focus cohere] [--bins 0.35 0.45]

Writes m2_stratification<SUFFIX>.csv, m2_within_k<SUFFIX>.csv and
m2_per_subset<SUFFIX>.csv next to the inputs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(d, name, sfx):
    p = d / f"{name}{sfx}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return pd.read_csv(p)


def per_subset(ranks, surv, reported, pair, focus):
    """One row per subset seating both members of `pair`."""
    p1, p2 = pair
    rows = []
    for pool, g in ranks.groupby("pool"):
        members = pool.split("+")
        if p1 not in members or p2 not in members:
            continue
        denom = sum(surv[m] for m in members)
        share = (surv[p1] + surv[p2]) / denom

        n_rep = g.groupby("model")["count"].sum().max()
        held = {}
        for model, gm in g.groupby("model"):
            r = reported.get(model)
            held[model] = gm.loc[gm["rank"] == r, "count"].sum() / n_rep if n_rep else np.nan

        gf = g[g.model == focus]
        modal = int(gf.loc[gf["count"].idxmax(), "rank"]) if len(gf) else np.nan

        rows.append({
            "pool": pool,
            "k": len(members),
            "share": share,
            "focus_in_pool": int(focus in members),
            "ret_focus": held.get(focus, np.nan),
            "ret_pair_1": held.get(p1, np.nan),
            "ret_pair_2": held.get(p2, np.nan),
            "modal_focus": modal,
            "holds_reported": int(modal == reported.get(focus)),
        })
    return pd.DataFrame(rows).sort_values("share").reset_index(drop=True)


def binned(per, edges):
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    per = per.copy()
    per["bin"] = pd.cut(per.share, bins=edges, labels=labels, include_lowest=True)
    agg = (per.groupby("bin", observed=True)
              .agg(subsets=("pool", "size"),
                   share_min=("share", "min"),
                   share_max=("share", "max"),
                   k_min=("k", "min"), k_max=("k", "max"),
                   focus_in=("focus_in_pool", "sum"),
                   holds=("holds_reported", "sum"),
                   mean_ret_focus=("ret_focus", "mean"),
                   mean_ret_pair=("ret_pair_1", "mean"))
              .reset_index())
    agg["frac_holds"] = agg.holds / agg.subsets
    return agg


def corr(per, col="ret_focus"):
    if per.share.nunique() < 3 or per[col].nunique() < 3:
        return np.nan, np.nan
    return per.share.corr(per[col]), per.share.corr(per[col], method="spearman")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("-s", "--suffix", default="")
    ap.add_argument("--pair", nargs=2, default=["whisper_v3", "whisper_v2"])
    ap.add_argument("--control", nargs=2, default=None,
                    help="accuracy-matched pair with no documented lineage")
    ap.add_argument("--focus", default="cohere")
    ap.add_argument("--bins", nargs="*", type=float, default=[0.35, 0.45])
    a = ap.parse_args()

    d = Path(a.directory).expanduser().resolve()
    sfx = f"_{a.suffix.strip('_')}" if a.suffix else ""

    ranks = load(d, "subpool_ranks", sfx)
    filt = load(d, "model_filtration", sfx)
    rank_full = load(d, "ranking", sfx)

    surv = dict(zip(filt.model, 1.0 - filt.f_bar))
    reported = dict(zip(rank_full.model, rank_full.rank_pseudo))
    edges = [0.0] + sorted(a.bins) + [1.0]

    per = per_subset(ranks, surv, reported, a.pair, a.focus)
    if per.empty:
        raise SystemExit(f"no subset seats both {a.pair[0]} and {a.pair[1]}")

    print(f"\n=== lineage pair: {a.pair[0]} + {a.pair[1]}   focus: {a.focus} ===")
    print(f"subsets seating both: {len(per)} of {ranks.pool.nunique()}")
    print(f"share range: {per.share.min():.3f} to {per.share.max():.3f}\n")
    agg = binned(per, edges)
    print(agg.to_string(index=False))
    r, rho = corr(per)
    print(f"\ncorrelation(share, ret_{a.focus}): Pearson {r:+.3f}, Spearman {rho:+.3f}")

    # --- control 1: hold pool size fixed ------------------------------------
    print("\n--- within pool size (share varies by composition, not by k) ---")
    wk = []
    for k, g in per.groupby("k"):
        r_k, rho_k = corr(g)
        wk.append({"k": k, "subsets": len(g),
                   "share_min": g.share.min(), "share_max": g.share.max(),
                   "mean_ret_focus": g.ret_focus.mean(),
                   "frac_holds": g.holds_reported.mean(),
                   "pearson": r_k, "spearman": rho_k})
    wkdf = pd.DataFrame(wk)
    print(wkdf.to_string(index=False))

    # --- control 2: is the strongest system even in the pool? ---------------
    print(f"\n--- split on whether {a.focus} is a member of the subset ---")
    sp = []
    for inpool, g in per.groupby("focus_in_pool"):
        r_s, rho_s = corr(g)
        sp.append({f"{a.focus}_in_pool": bool(inpool), "subsets": len(g),
                   "share_min": g.share.min(), "share_max": g.share.max(),
                   "mean_ret_focus": g.ret_focus.mean(),
                   "frac_holds": g.holds_reported.mean(),
                   "pearson": r_s, "spearman": rho_s})
    print(pd.DataFrame(sp).to_string(index=False))

    # --- control 3: a matched pair with no documented lineage ---------------
    if a.control:
        ctl = per_subset(ranks, surv, reported, a.control, a.focus)
        if ctl.empty:
            print(f"\n(control pair {a.control} seated in no subset)")
        else:
            print(f"\n=== control pair (no documented lineage): "
                  f"{a.control[0]} + {a.control[1]} ===")
            print(f"subsets seating both: {len(ctl)}   "
                  f"share range: {ctl.share.min():.3f} to {ctl.share.max():.3f}\n")
            print(binned(ctl, edges).to_string(index=False))
            r_c, rho_c = corr(ctl)
            print(f"\ncorrelation(share, ret_{a.focus}): "
                  f"Pearson {r_c:+.3f}, Spearman {rho_c:+.3f}")
            print(f"\nlineage Spearman {rho:+.3f} vs control {rho_c:+.3f} "
                  f"-> lineage is {'steeper' if rho < rho_c else 'NOT steeper'}")
            ctl.to_csv(d / f"m2_per_subset_control{sfx}.csv", index=False)

    agg.to_csv(d / f"m2_stratification{sfx}.csv", index=False)
    wkdf.to_csv(d / f"m2_within_k{sfx}.csv", index=False)
    per.to_csv(d / f"m2_per_subset{sfx}.csv", index=False)
    print(f"\nwrote m2_stratification{sfx}.csv, m2_within_k{sfx}.csv, "
          f"m2_per_subset{sfx}.csv")


if __name__ == "__main__":
    main()
