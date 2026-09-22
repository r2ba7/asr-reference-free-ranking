"""
Reference-free ranking, pool diagnostics, and rank-uncertainty quantification.

This module never reads a ground-truth transcription. It imports nothing from
evaluation_protocol. `normalized_transcription` is not read, `true_ref` does not
exist, and every table is screened on the way in and on the way out against
FORBIDDEN_SUBSTRINGS. Every quantity produced here is computable on unlabeled
audio.

Two entry points, one internal state:

  DeploymentProtocol.from_hypotheses(...)   fuse from 1-best hypotheses
  DeploymentProtocol.from_run(indir, ...)   rehydrate a completed run

Either may be given `indir` to load previously written reference-free tables for
merging; it may be None.

What it answers:
  ranking()             the reference-free ordering
  cross_wer()           pairwise hypothesis disagreement
    model_filtration()    f-bar: each model's mean exclusion rate across subsets
  rank_distribution()   P(model at rank r), from an exhaustive enumeration of
                        voter subsets with an utterance bootstrap inside each

What it cannot answer: whether the ranking is correct. That needs references.

Contract for `ensemble_factory`: zero-arg callable returning an object with
  .combine_models_transcriptions(*samples_dicts)
  .main()
  .samples_info -> {audio_path: {"normalized_prediction": str, "metadata": {...}}}

Contract for `model_samples`: {name: {audio_path: {"normalized_prediction": str}}}
Any other key present is ignored; none is required.
"""

import inspect
import itertools
import json
import os
from collections import defaultdict
import time

import numpy as np
import pandas as pd

from . import metrics, logger, decorators

LOGGER = logger.Logger.get_logger(module_name=__name__)

FORBIDDEN_SUBSTRINGS = ("true", "compression", "vs_true", "_wrong",
                        "resolvable", "transcription", "correct")

ALLOWED_FILES = {
    "cross_wer": None,
    "ranking": ["model", "role", "n_samples", "wer_pseudo_loo", "cer_pseudo_loo", "rank_pseudo"],
    "diagnostics_runs": None,
    "diagnostics_dropout": None,
    "rank_distribution": None,
    "model_confidence": None,
    "pair_probability": None,
    "by_pool_size": None,
    "model_filtration": None,
    "subpool_filtration": None,
}


def has_pred(s):
    return isinstance(s, str) and s.strip() != ""


def word_counts(m):
    w = m["word_error_rate"]
    return w["word_edit_distance"], w["ref_length"]


def char_counts(m):
    c = m["character_error_rate"]
    return c["char_edit_distance"], c["ref_length"]


def micro_wer(per_sample, paths):
    e = sum(word_counts(per_sample[p])[0] for p in paths)
    n = sum(word_counts(per_sample[p])[1] for p in paths)
    return 100.0 * e / n if n else float("nan")


def micro_cer(per_sample, paths):
    e = sum(char_counts(per_sample[p])[0] for p in paths)
    n = sum(char_counts(per_sample[p])[1] for p in paths)
    return 100.0 * e / n if n else float("nan")


def _assert_clean(df, where):
    bad = [c for c in df.columns if any(f in str(c).lower() for f in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise ValueError(f"{where}: reference-derived columns rejected: {bad}")
    return df


def _load_dir(indir, suffix):
    """Load whitelisted tables from a directory, dropping every column whose name
    marks it as reference-derived before the data reaches the object."""
    sfx = f"_{suffix.strip('_')}" if suffix else ""
    out = {}
    for name, keep in ALLOWED_FILES.items():
        path = os.path.join(indir, f"{name}{sfx}.csv")
        if not os.path.exists(path):
            continue
        idx = 0 if name in ("cross_wer", "rank_distribution") else None
        df = pd.read_csv(path, index_col=idx)
        if keep is not None:
            df = df[[c for c in keep if c in df.columns]]
        else:
            df = df[[c for c in df.columns
                     if not any(f in str(c).lower() for f in FORBIDDEN_SUBSTRINGS)]]
        out[name] = _assert_clean(df, name)
    return out


class DeploymentProtocol:

    MODE_LIVE = "live"
    MODE_AUDIT = "audit"

    # ---------- construction ----------

    def __init__(self, mode, voters, ranked_only=(), model_samples=None,
                 ensemble_factory=None, main_kwargs=None, min_pool_size=3,
                 n_boot=1000, seed=42, tables=None,
                 outdir="deployment_out", suffix=""):
        self.mode = mode
        self.voters = list(voters)
        self.ranked_only = list(ranked_only)
        self.ms = model_samples
        self.factory = ensemble_factory
        self.min_pool_size = min_pool_size
        self.n_boot = n_boot
        self.seed = seed
        self.outdir = outdir
        self.suffix = f"_{suffix.strip('_')}" if suffix else ""
        self.tables = tables or {}
        self._fusion_cache = {}
        self._score_cache = {}
        self.report = {}
        os.makedirs(outdir, exist_ok=True)
        if mode == self.MODE_LIVE:
            self.main_kwargs = self._filter_main_kwargs(main_kwargs or {})
            names = list(self.ms)
            common = sorted(set.intersection(*[set(s.keys()) for s in self.ms.values()]))
            self.paths = [p for p in common
                          if all(has_pred(self.ms[n][p].get("normalized_prediction")) for n in names)]
            if not self.paths:
                raise ValueError("No audio path carries a hypothesis from every system.")
            self.report["n_samples"] = len(self.paths)
            self.report["main_kwargs_applied"] = dict(self.main_kwargs)
        else:
            self.main_kwargs = {}
            self.paths = []

    @classmethod
    def from_hypotheses(cls, model_samples, voters, ensemble_factory,
                        main_kwargs=None, min_pool_size=3, n_boot=1000, seed=42,
                        indir=None, outdir="deployment_out", suffix=""):
        """Fuse from 1-best hypotheses. `indir` optionally loads previously written
        reference-free tables for merging; it may be None."""
        ro = [n for n in model_samples if n not in voters]
        tables = _load_dir(indir, suffix) if indir else {}
        return cls(cls.MODE_LIVE, voters, ro, model_samples=model_samples,
                   ensemble_factory=ensemble_factory, main_kwargs=main_kwargs,
                   min_pool_size=min_pool_size, n_boot=n_boot, seed=seed,
                   tables=tables, outdir=outdir, suffix=suffix)

    @classmethod
    def from_run(cls, indir, voters, suffix="", outdir=None):
        """Rehydrate a completed run. Whitelisted files only, reference-derived
        columns dropped at load."""
        tables = _load_dir(indir, suffix)
        if not tables:
            raise ValueError(f"No whitelisted tables found in {indir}.")
        return cls(cls.MODE_AUDIT, voters, tables=tables,
                   outdir=outdir or indir, suffix=suffix)

    def _require_live(self, what):
        if self.mode != self.MODE_LIVE:
            raise RuntimeError(f"{what} requires from_hypotheses(); this object is in audit mode.")

    def _filter_main_kwargs(self, kw):
        if not kw:
            return {}
        try:
            sig = inspect.signature(self.factory().main)
        except Exception:
            return {}
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kw)
        return {k: v for k, v in kw.items() if k in params}

    # ---------- fusion and scoring against the consensus ----------

    def _fuse(self, subset):
        key = frozenset(subset)
        if key in self._fusion_cache:
            return self._fusion_cache[key]
        self._require_live("fusion")
        order = [n for n in self.voters if n in key]
        ens = self.factory()
        ens.combine_models_transcriptions(*[self.ms[n] for n in order])
        ens.main(**self.main_kwargs)
        si = ens.samples_info
        keep = [p for p in self.paths if p in si and has_pred(si[p]["normalized_prediction"])]
        refs = {p: si[p]["normalized_prediction"] for p in keep}
        meta = {p: si[p]["metadata"] for p in keep}
        self._fusion_cache[key] = (refs, meta, order)
        return self._fusion_cache[key]

    def _score(self, ref_map, name, key=None):
        """Score one system against a consensus transcript. The metric's `refs`
        argument is a fused hypothesis, never a human transcription."""
        ck = (key, name)
        if key is not None and ck in self._score_cache:
            return self._score_cache[ck]
        s = self.ms[name]
        out = {}
        for p in self.paths:
            r = ref_map.get(p)
            if not has_pred(r):
                continue
            out[p] = metrics.BasicSTTMetrics.evaluate(refs=r, hyps=s[p]["normalized_prediction"])
        if key is not None:
            self._score_cache[ck] = out
        return out

    def _loo_key(self, name, pool=None):
        pool = self.voters if pool is None else pool
        sub = [n for n in pool if n != name] if name in pool else list(pool)
        if len(sub) < self.min_pool_size:
            raise ValueError(f"LOO pool for {name} has {len(sub)} voters (< {self.min_pool_size}).")
        return frozenset(sub)

    def _loo_score(self, name, pool=None):
        key = self._loo_key(name, pool)
        return self._score(self._fuse(key)[0], name, key=key)

    # ---------- the reference-free ranking ----------
    @decorators.Decorators.calculate_execution_time
    def ranking(self, pool=None, store=True):
        if self.mode == self.MODE_AUDIT:
            d = self.tables.get("ranking")
            if d is None:
                raise RuntimeError("audit mode: ranking table not loaded.")
            return d, {"n_models": len(d), "source": "audit"}
        pool = self.voters if pool is None else pool
        rows = []
        for name in self.ms:
            l = self._loo_score(name, pool)
            ps = [p for p in self.paths if p in l]
            rows.append({"model": name,
                         "role": "voter" if name in pool else "ranked_only",
                         "n_samples": len(ps),
                         "wer_pseudo_loo": micro_wer(l, ps),
                         "cer_pseudo_loo": micro_cer(l, ps)})
        df = pd.DataFrame(rows).sort_values("wer_pseudo_loo").reset_index(drop=True)
        df["rank_pseudo"] = df.wer_pseudo_loo.rank().astype(int)
        stat = {"n_models": len(df), "pool": "+".join(pool)}
        if store:
            self.report["ranking"] = df
            self.report["ranking_stats"] = stat
        return df, stat

    # ---------- pool diversity ----------
    @decorators.Decorators.calculate_execution_time
    def cross_wer(self, store=True):
        """Pairwise disagreement between hypotheses. One system's output is the
        metric's reference for another's; no human transcript is involved."""
        if self.mode == self.MODE_AUDIT:
            cw = self.tables.get("cross_wer")
            if cw is None:
                raise RuntimeError("audit mode: cross_wer not loaded.")
            if store:
                self.report["cross_wer"] = cw
            return cw
        names = list(self.ms)
        cw = pd.DataFrame(index=names, columns=names, dtype=float)
        for a, b in itertools.combinations(names, 2):
            refs = [self.ms[a][p]["normalized_prediction"] for p in self.paths]
            hyps = [self.ms[b][p]["normalized_prediction"] for p in self.paths]
            v = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)["word_error_rate"]["wer (%)"]
            cw.loc[a, b] = v
            cw.loc[b, a] = v
        np.fill_diagonal(cw.values, 0.0)
        if store:
            self.report["cross_wer"] = cw
        return cw



    # ---------- rank distribution ----------
    def _subpool_filtration(self, subs):
        """Per-subset exclusion rate for every member, read from the fusion's own
        filter decisions. Reference-free: the filter never sees a transcription."""
        rows=[]
        for sub in subs:
            _,meta,order=self._fuse(sub)
            n=len(meta)
            if not n:
                continue
            cnt=defaultdict(int)
            for md in meta.values():
                for i in md["records_filtration"].get("filtered_indices",[]):
                    cnt[order[i]]+=1
            for m in order:
                rows.append({"pool":"+".join(sub),"k":len(sub),"model":m,
                             "n_samples":n,"filtered_out":cnt[m],
                             "filtered_out_rate":cnt[m]/n})
        return pd.DataFrame(rows)
    
    @decorators.Decorators.calculate_execution_time
    def model_filtration(self, subs, store=True):
        """f-bar: each model's mean exclusion rate over the subsets containing it.
        A model's filtration is a property of the pool it sits in, not of the model,
        so the full-pool figure is one observation rather than the quantity itself;
        averaging over the enumeration gives the model-level value. Every model
        appears in the same number of subsets, so the average is balanced."""
        per=self._subpool_filtration(subs)
        agg=(per.groupby("model")
                .agg(n_subpools=("pool","size"),
                     f_bar=("filtered_out_rate","mean"),
                     f_sd=("filtered_out_rate","std"),
                     f_min=("filtered_out_rate","min"),
                     f_max=("filtered_out_rate","max"))
                .reset_index().sort_values("f_bar").reset_index(drop=True))
        agg["survival"]=1.0-agg.f_bar
        if store:
            self.report["subpool_filtration"]=per
            self.report["model_filtration"]=agg
        return agg

    def _pool_weights(self, subs, fbar):
        """w(S) = sum over members of (1 - f-bar): the expected number of
        hypotheses that survive the filter and vote in S. A member the filter almost
        always removes contributes almost nothing, so a pool is not penalised for
        carrying one; a member that survives and disagrees counts in full. Pool size
        enters through the sum rather than a separate term, and no threshold is
        introduced. Weights are normalised over the enumeration."""
        w=np.array([sum(1.0-fbar[m] for m in sub) for sub in subs],float)
        return w/w.sum()
    
    def _subpool_counts(self, sub):
        """Per-model edit-distance and reference-length vectors over the utterances
        common to every model under this sub-pool's leave-one-out references."""
        names = list(self.ms)
        per = {n: self._loo_score(n, sub) for n in names}
        cp = [p for p in self.paths if all(p in per[n] for n in names)]
        E = np.array([[word_counts(per[n][p])[0] for p in cp] for n in names], float)
        L = np.array([[word_counts(per[n][p])[1] for p in cp] for n in names], float)
        return names, E, L, cp
    
    @decorators.Decorators.calculate_execution_time
    def rank_distribution(self, min_rank_pool=None, weighting="uniform",
                          store=True):
        """P(model at rank r), estimated by an exhaustive enumeration of voter
        subsets with an utterance bootstrap inside each.

        Two sources of uncertainty are combined. The outer loop varies the pool, so
        each subset supplies a different consensus and hence a different
        reference-induced displacement of every system's error rate; this is the
        quantity that can reorder a pair, and resampling utterances cannot vary it.
        The inner loop resamples utterances within each subset, which accounts for
        the corpus being finite.

        `weighting` decides how subsets are mixed. "uniform" treats each as one
        observation of what the method would report from that pool. "filtration"
        weights each subset by the expected number of hypotheses that survive its
        own filter, from model_filtration(); this is monotone in effective voters,
        which the ablation shows is the direction ranking fidelity improves in, and
        it introduces no threshold. Both are reported so that the weighting can be
        seen not to carry the conclusion.."""
        self._require_live("rank_distribution")
        pmin = (self.min_pool_size + 1) if min_rank_pool is None else min_rank_pool
        subs = [list(s) for k in range(pmin, len(self.voters) + 1)
                for s in itertools.combinations(self.voters, k)]
        names = list(self.ms)
        nm = len(names)
        full = self.ranking(store=False)[0].set_index("model")
        rng = np.random.default_rng(self.seed)
        if weighting == "filtration":
            fbar = self.model_filtration(subs).set_index("model")["f_bar"].to_dict()
            wts = self._pool_weights(subs, fbar)
        elif weighting == "uniform":
            wts = np.full(len(subs), 1.0 / len(subs))
        else:
            raise ValueError(f"unknown weighting: {weighting}")

        total = np.zeros((nm, nm), float)
        pair_wins = defaultdict(int)
        pair_tot = 0
        per_sub_rows, sub_ent = [], []
        t_counts = t_boot = 0.0

        for si_, sub in enumerate(subs):
            _t = time.perf_counter()
            try:
                _, E, L, cp = self._subpool_counts(sub)
            except ValueError:
                continue
            finally:
                t_counts += time.perf_counter() - _t
            npaths = len(cp)
            idx = np.arange(npaths)
            counts = np.zeros((nm, nm), int)
            _t = time.perf_counter()
            for _ in range(self.n_boot):
                s = rng.choice(idx, size=npaths, replace=True)
                c = np.bincount(s, minlength=npaths).astype(float)
                w = (E * c).sum(1) / (L * c).sum(1)
                order = np.argsort(np.argsort(w))
                counts[np.arange(nm), order] += 1
                for i, j in itertools.combinations(range(nm), 2):
                    if w[i] < w[j]:
                        pair_wins[(names[i], names[j])] += wts[si_]
                pair_tot += wts[si_]
            t_boot += time.perf_counter() - _t
            total += counts * wts[si_]
            p = counts / counts.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -(p * np.log(np.where(p > 0, p, 1))).sum(1) / np.log(nm)
            sub_ent.append({"pool": "+".join(sub), "k": len(sub),
                            "weight": float(wts[si_]),
                            "mean_entropy": float(h.mean())})
            for a, n in enumerate(names):
                for r in range(nm):
                    if counts[a, r]:
                        per_sub_rows.append({"pool": "+".join(sub), "k": len(sub),
                                             "model": n, "rank": r + 1,
                                             "count": int(counts[a, r])})

        P = total / total.sum(1, keepdims=True)
        dist = pd.DataFrame(P, index=names,
                            columns=[f"rank_{r+1}" for r in range(nm)])
        dist.index.name = "model"

        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(P * np.log(np.where(P > 0, P, 1))).sum(1) / np.log(nm)
        ranks = np.arange(1, nm + 1)
        rows = []
        for a, n in enumerate(names):
            rf = int(full.loc[n, "rank_pseudo"])
            cum = np.cumsum(P[a])
            mean_r = float((ranks * P[a]).sum())
            rows.append({"model": n,
                         "rank_pseudo_full": rf,
                         "rank_modal": int(ranks[P[a].argmax()]),
                         "p_modal": float(P[a].max()),
                         "p_at_full_rank": float(P[a, rf - 1]),
                         "entropy_norm": float(H[a]),
                         "rank_mean": mean_r,
                         "rank_sd": float(np.sqrt((((ranks - mean_r) ** 2) * P[a]).sum())),
                         "rank_ci_lo": int(ranks[min(np.searchsorted(cum, 0.025), nm - 1)]),
                         "rank_ci_hi": int(ranks[min(np.searchsorted(cum, 0.975), nm - 1)]),
                         "agrees_with_full": bool(int(ranks[P[a].argmax()]) == rf)})
        conf = pd.DataFrame(rows).sort_values("rank_pseudo_full").reset_index(drop=True)

        prow = []
        for a, b in itertools.combinations(names, 2):
            wa = pair_wins[(a, b)] / pair_tot
            lead = a if full.loc[a, "wer_pseudo_loo"] < full.loc[b, "wer_pseudo_loo"] else b
            prow.append({"pair": f"{a}|{b}",
                         "pred_winner_full": lead,
                         "gap_wer_pseudo": float(abs(full.loc[a, "wer_pseudo_loo"]
                                                     - full.loc[b, "wer_pseudo_loo"])),
                         "p_first_beats_second": float(wa),
                         "p_supports_full": float(wa if lead == a else 1 - wa)})
        pairs = pd.DataFrame(prow).sort_values("p_supports_full").reset_index(drop=True)

        ent = pd.DataFrame(sub_ent)
        by_k = ent.groupby("k").agg(pools=("pool", "size"),
                                    total_weight=("weight", "sum"),
                                    mean_entropy=("mean_entropy", "mean"),
                                    max_entropy=("mean_entropy", "max")).reset_index()

        summ = {"n_subpools": len(sub_ent), "n_boot_inner": self.n_boot,
                "n_fusions": len(self._fusion_cache),
                "n_scorings": len(self._score_cache),
                "seconds_fuse_and_score": round(t_counts, 1),
                "seconds_bootstrap": round(t_boot, 1),
                "weighting": weighting,
                "mean_entropy": float(H.mean()), "max_entropy": float(H.max()),
                "n_models_modal_differs_from_full": int((~conf.agrees_with_full).sum())}

        if store:
            self.report["rank_distribution"] = dist
            self.report["model_confidence"] = conf
            self.report["pair_probability"] = pairs
            self.report["by_pool_size"] = by_k
            self.report["subpool_ranks"] = pd.DataFrame(per_sub_rows)
            self.report["rank_distribution_summary"] = summ
        LOGGER.info("Rank distribution finished.")
        return dist, conf

    # ---------- driver ----------
    @decorators.Decorators.calculate_execution_time
    def run(self, ranks=True, weighting="uniform"):
        self.cross_wer()
        self.ranking()
        if self.mode == self.MODE_LIVE and ranks:
            self.rank_distribution(weighting=weighting)
        self._dump()
        LOGGER.info("Deployment protocol finished.")
        return self.report

    def _dump(self):
        scalars = {}
        for k, v in self.report.items():
            if isinstance(v, pd.DataFrame):
                _assert_clean(v, f"dump:{k}")
                v.to_csv(os.path.join(self.outdir, f"{k}{self.suffix}.csv"),
                         index=k in ("cross_wer", "rank_distribution"))
            else:
                scalars[k] = v
        with open(os.path.join(self.outdir, f"scalars{self.suffix}.json"), "w") as f:
            json.dump(scalars, f, indent=2, default=float)