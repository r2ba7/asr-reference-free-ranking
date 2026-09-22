"""
Evaluation protocol for reference-free ASR benchmarking via ensemble pseudo-groundtruth.

Single entry point: EvaluationProtocol(...).run()

Steps:
  0  decorrelation audit        pool uniqueness, measured not assumed
  1  self-agreement bias        full-pool vs LOO per-sample correlation; ranked-only = control
  2  system ranking             LOO pseudo-GT vs true GT, exact permutation p
  3  pipeline diagnostics       filter fire rate (per model), reference strategy mix, tie rate, determinism
  4  bootstrap resolution       adjacent + all-pair stability, minimum detectable WER gap
  5  baselines                  single-model / full-ensemble references. Without this, step 2 is uninterpretable.
  6  pool ablation              per-voter contribution on TWO objectives: reference quality AND ranking fidelity
  7  floor controls             nonzero-WER subset, tau-b, edit-distance-based correlation

Contract for `ensemble_factory`: zero-arg callable returning an object with
  .combine_models_transcriptions(*samples_dicts)
  .main()
  .samples_info -> {audio_path: {"normalized_prediction": str, "metadata": {...}}}

Contract for `model_samples`: {name: {audio_path: {"normalized_prediction": str,
  "normalized_transcription": str, "metrics": <BasicSTTMetrics.evaluate output>}}}
"""

import inspect
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics, logger

LOGGER = logger.Logger.get_logger(module_name=__name__)

# ---------- metric accessors (nested BasicSTTMetrics format) ----------
_PERM_CACHE={}

def _perm_matrix(rx):
    """All n! permutations of rx, cached. In step 6 the x-vector is rankdata(wer_true),
    which is identical across all 466 pool subsets, so this is built once."""
    key=(len(rx),tuple(rx))
    P=_PERM_CACHE.get(key)
    if P is None:
        P=np.array(list(itertools.permutations(rx)),dtype=float)
        _PERM_CACHE[key]=P
    return P

def wer_pct(m):
    return m["word_error_rate"]["wer (%)"]

def cer_pct(m):
    return m["character_error_rate"]["cer (%)"]

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

def has_ref(s):
    return isinstance(s, str) and s.strip() != ""

class EvaluationProtocol:

    def __init__(self, model_samples, voters, ensemble_factory,
                 outdir="protocol_out", n_boot=1000, seed=42,
                 min_pool_size=3, full_ensemble_samples=None,
                 main_kwargs=None, suffix=""):
        self.ms = model_samples
        self.voters = list(voters)
        self.ranked_only = [n for n in model_samples if n not in self.voters]
        self.factory = ensemble_factory
        self.main_kwargs = self._filter_main_kwargs(main_kwargs or {})
        self.outdir = outdir
        self.suffix = f"_{suffix.strip('_')}" if suffix else ""
        self.n_boot = n_boot
        self.seed = seed
        self.min_pool_size = min_pool_size
        os.makedirs(outdir, exist_ok=True)
        common = sorted(set.intersection(*[set(s.keys()) for s in model_samples.values()]))
        if not common:
            raise ValueError("No audio paths common to all models.")
        gt = {p: model_samples[self.voters[0]][p]["normalized_transcription"] for p in common}
        self.paths = [p for p in common if has_ref(gt[p])]
        self.skipped_empty_gt = [p for p in common if not has_ref(gt[p])]
        if not self.paths:
            raise ValueError("Every sample has an empty ground-truth reference.")
        self.true_ref = {p: gt[p] for p in self.paths}
        self._fusion_cache = {}
        if full_ensemble_samples is not None:
            key = frozenset(self.voters)
            pset = set(self.paths)
            keep = [p for p, v in full_ensemble_samples.items()
                    if p in pset and has_ref(v["normalized_prediction"])]
            self._fusion_cache[key] = (
                {p: full_ensemble_samples[p]["normalized_prediction"] for p in keep},
                {p: full_ensemble_samples[p]["metadata"] for p in keep},
                self.voters,
            )
        self.report = {"skipped_empty_gt": len(self.skipped_empty_gt),
                       "main_kwargs_applied": dict(self.main_kwargs),
                       "main_kwargs_dropped": getattr(self, "_dropped_main_kwargs", [])}
        self._score_cache = {}

    # ---------- fusion and scoring, cached ----------
    def _filter_main_kwargs(self, kw):
        """Keep only kwargs the ensemble's main() actually accepts.
        An older ensemble without substitute / normalize_final_letters still runs;
        dropped keys are recorded in the report."""
        if not kw:
            return {}
        try:
            sig = inspect.signature(self.factory().main)
        except Exception:
            return {}
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kw)
        keep = {k: v for k, v in kw.items() if k in params}
        self._dropped_main_kwargs = sorted(set(kw) - set(keep))
        return keep

    def _fuse(self, subset):
        key = frozenset(subset)
        if key in self._fusion_cache:
            return self._fusion_cache[key]
        order = [n for n in self.voters if n in key]
        ens = self.factory()
        ens.combine_models_transcriptions(*[self.ms[n] for n in order])
        ens.main(**self.main_kwargs)
        si = ens.samples_info
        keep = [p for p in self.paths if p in si and has_ref(si[p]["normalized_prediction"])]
        refs = {p: si[p]["normalized_prediction"] for p in keep}
        meta = {p: si[p]["metadata"] for p in keep}
        self._fusion_cache[key] = (refs, meta, order)
        return self._fusion_cache[key]

    def _score(self, ref_map, name, key=None):
        """key: hashable identifier of the reference (frozenset of the fused pool,
        or ('single', model) for single-model references). Enables caching."""
        ck = (key, name)
        if key is not None and ck in self._score_cache:
            return self._score_cache[ck]
        s = self.ms[name]
        out = {}
        for p in self.paths:
            r = ref_map.get(p)
            if not has_ref(r):
                continue
            out[p] = metrics.BasicSTTMetrics.evaluate(refs=r, hyps=s[p]["normalized_prediction"])
        if key is not None:
            self._score_cache[ck] = out
        return out

    def _loo_key(self, name, pool=None):
        """The voter subset whose fusion serves as `name`'s reference."""
        pool = self.voters if pool is None else pool
        sub = [n for n in pool if n != name] if name in pool else list(pool)
        if len(sub) < self.min_pool_size:
            raise ValueError(f"LOO pool for {name} has {len(sub)} voters (< {self.min_pool_size}).")
        return frozenset(sub)

    def _loo_ref(self, name, pool=None):
        return self._fuse(self._loo_key(name, pool))[0]

    def _loo_score(self, name, pool=None):
        """Per-sample metrics of `name` against its LOO pseudo-reference. Cached."""
        key = self._loo_key(name, pool)
        return self._score(self._fuse(key)[0], name, key=key)

    def _true_per_sample(self, name):
        return {p: self.ms[name][p]["metrics"] for p in self.paths}

    @staticmethod
    def exact_rank_p(x,y,max_exact=9,n_perm=200000,seed=0):
        """One-tailed exact permutation p-values for Spearman rho and Kendall tau_b
        from a single shared enumeration. Permuting x preserves its tie structure, so
        both statistics are monotone in their numerators: rx.ry for rho, the pairwise
        sign-product sum for tau_b. Above max_exact, seeded Monte-Carlo (see p_method)."""
        rx=stats.rankdata(x).astype(float)
        ry=stats.rankdata(y).astype(float)
        n=len(rx)
        rho=stats.spearmanr(rx,ry).correlation
        tau=stats.kendalltau(rx,ry,variant="b").correlation
        i,j=np.triu_indices(n,1)
        sy=np.sign(ry[j]-ry[i])
        exact=n<=max_exact
        if exact:
            P=_perm_matrix(rx)
        else:
            rng=np.random.default_rng(seed)
            P=np.stack([rng.permutation(rx) for _ in range(n_perm)])
        rho_num=P@ry
        tau_num=np.zeros(len(P))
        for k in range(len(i)):
            if sy[k]:
                tau_num+=np.sign(P[:,j[k]]-P[:,i[k]])*sy[k]
        obs_rho=float(rx@ry)
        obs_tau=float(np.sign(rx[j]-rx[i])@sy)
        eps=1e-9
        B=len(P)
        h_rho=int((rho_num>=obs_rho-eps).sum())
        h_tau=int((tau_num>=obs_tau-eps).sum())
        return {"spearman":float(rho),
                "kendall_b":float(tau),
                "spearman_exact_p_onetailed":float(h_rho/B if exact else (1+h_rho)/(1+B)),
                "kendall_exact_p_onetailed":float(h_tau/B if exact else (1+h_tau)/(1+B)),
                "p_method":"exact" if exact else f"monte_carlo_{n_perm}",
                "n_permutations":int(B),
                "min_attainable_p":float(1.0/B if exact else 1.0/(1+B))}

    @staticmethod
    def exact_spearman_p(x,y):
        r=EvaluationProtocol.exact_rank_p(x,y)
        return r["spearman"],r["spearman_exact_p_onetailed"]

    # ---------- step 0: decorrelation audit ----------

    def step0_decorrelation(self):
        names = list(self.ms)
        cw = pd.DataFrame(index=names, columns=names, dtype=float)   # cross-WER: hyp_i vs hyp_j
        for a, b in itertools.combinations(names, 2):
            ps = [p for p in self.paths if has_ref(self.ms[a][p]["normalized_prediction"])]
            if not ps:
                v = float("nan")
            else:
                refs = [self.ms[a][p]["normalized_prediction"] for p in ps]
                hyps = [self.ms[b][p]["normalized_prediction"] for p in ps]
                v = wer_pct(metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps))
            cw.loc[a, b] = v
            cw.loc[b, a] = v
        np.fill_diagonal(cw.values, 0.0)
        errs = {n: np.array([word_counts(self.ms[n][p]["metrics"])[0] for p in self.paths], float) for n in names}
        ec = pd.DataFrame(index=names, columns=names, dtype=float)
        for a, b in itertools.combinations(names, 2):
            v = stats.spearmanr(errs[a], errs[b]).correlation
            ec.loc[a, b] = v
            ec.loc[b, a] = v
        np.fill_diagonal(ec.values, 1.0)
        off = cw.values[~np.eye(len(names), dtype=bool)]
        flags = []
        for a, b in itertools.combinations(names, 2):
            if cw.loc[a, b] < off.mean() - off.std():
                flags.append({"pair": f"{a}|{b}", "cross_wer": cw.loc[a, b], "err_corr": ec.loc[a, b],
                              "note": "unusually similar hypotheses - suspect shared lab/data/teacher"})
        self.report["step0_cross_wer"] = cw
        self.report["step0_err_corr"] = ec
        self.report["step0_flags"] = pd.DataFrame(flags)
        LOGGER.info(f"Step 0 Finished.")
        return cw, ec

    # ---------- step 1: self-agreement bias ----------

    def step1_bias(self):
        full_key = frozenset(self.voters)
        full_ref = self._fuse(self.voters)[0]
        rows = []
        for name in self.ms:
            fsc = self._score(full_ref, name, key=full_key)
            lsc = self._loo_score(name)
            ps = [p for p in self.paths if p in fsc and p in lsc]
            t = np.array([wer_pct(self.ms[name][p]["metrics"]) for p in ps], float)
            f = np.array([wer_pct(fsc[p]) for p in ps], float)
            l = np.array([wer_pct(lsc[p]) for p in ps], float)
            rows.append({
                "model": name,
                "role": "voter" if name in self.voters else "ranked_only(control)",
                "n_samples": len(ps),
                "spearman_full": stats.spearmanr(t, f).correlation,
                "spearman_loo": stats.spearmanr(t, l).correlation,
                "kendall_full": stats.kendalltau(t, f, variant="b").correlation,
                "kendall_loo": stats.kendalltau(t, l, variant="b").correlation,
            })
        df = pd.DataFrame(rows)
        df["delta_spearman"] = df.spearman_loo - df.spearman_full
        df["delta_kendall"] = df.kendall_loo - df.kendall_full
        df = df.sort_values("spearman_full").reset_index(drop=True)
        self.report["step1_bias"] = df
        LOGGER.info(f"Step 1 Finished.")
        return df

    # ---------- step 2: system ranking ----------

    def step2_ranking(self, pool=None, store=True):
        pool = self.voters if pool is None else pool
        rows = []
        for name in self.ms:
            t = self._true_per_sample(name)
            l = self._loo_score(name, pool)
            ps = [p for p in self.paths if p in l]
            rows.append({
                "model": name,
                "role": "voter" if name in pool else "ranked_only",
                "n_samples": len(ps),
                "wer_true": micro_wer(t, ps),
                "cer_true": micro_cer(t, ps),
                "wer_pseudo_loo": micro_wer(l, ps),
                "cer_pseudo_loo": micro_cer(l, ps),
            })
        df = pd.DataFrame(rows).sort_values("wer_true").reset_index(drop=True)
        df["rank_true"] = df.wer_true.rank().astype(int)
        df["rank_pseudo"] = df.wer_pseudo_loo.rank().astype(int)
        df["wer_compression"] = df.wer_true - df.wer_pseudo_loo
        rk=self.exact_rank_p(df.wer_true.values,df.wer_pseudo_loo.values)
        pr=stats.pearsonr(df.wer_true.values,df.wer_pseudo_loo.values)
        stat={"n_models":len(df),**rk,
              "pearson":float(pr[0]),"pearson_p":float(pr[1]),
              "mean_wer_compression":float(df.wer_compression.mean())}
        if store:
            self.report["step2_ranking"] = df
            self.report["step2_stats"] = stat
            LOGGER.info(f"Step 2 Finished.")
        return df, stat

    # ---------- step 3: pipeline diagnostics ----------

    def step3_pipeline(self, check_determinism=True):
        rows = []
        pooled_filtered = defaultdict(int)
        pooled_present = defaultdict(int)
        for name in [None] + self.voters:
            sub = self.voters if name is None else [v for v in self.voters if v != name]
            _, meta, order = self._fuse(sub)
            n = len(meta)
            fired = ties = pos = 0
            strat = defaultdict(int)
            for p, md in meta.items():
                f = md["records_filtration"]
                if f.get("num_filtered", 0) > 0:
                    fired += 1
                for i in f.get("filtered_indices", []):
                    pooled_filtered[order[i]] += 1
                for m in order:
                    pooled_present[m] += 1
                strat[md["reference_selection"].get("strategy_metric", "unknown")] += 1
                for vd in md["voting"]["voting_details"]:
                    v = vd.get("token_votes") or {}
                    if v and max(v.values()) <= vd["models_voted"] / 2:
                        ties += 1
                    pos += 1
            rows.append({
                "run": "full_pool" if name is None else f"LOO_minus_{name}",
                "n_voters": len(sub), "n_samples": n,
                "filter_fire_rate": fired / n if n else 0.0,
                "tie_rate": ties / pos if pos else 0.0,
                **{f"refsel_{k}": v / n for k, v in strat.items()},
            })
        df = pd.DataFrame(rows)
        drop = pd.DataFrame([
            {"model": m, "times_filtered_out": pooled_filtered[m],
             "times_in_pool": pooled_present[m],
             "filtered_out_rate": pooled_filtered[m] / pooled_present[m] if pooled_present[m] else 0.0}
            for m in self.voters
        ]).sort_values("filtered_out_rate", ascending=False)
        self.report["step3_runs"] = df
        self.report["step3_model_dropout"] = drop
        if check_determinism:
            a = self._fuse(self.voters)[0]
            ens = self.factory()
            ens.combine_models_transcriptions(*[self.ms[n] for n in self.voters])
            ens.main(**self.main_kwargs)
            b = {p: v["normalized_prediction"] for p, v in ens.samples_info.items() if p in a}
            same = sum(1 for p in a if a[p] == b.get(p))
            self.report["step3_determinism"] = {"identical_outputs": same, "n": len(a),
                                                "deterministic": same == len(a),
                                                "main_kwargs": dict(self.main_kwargs)}
        LOGGER.info(f"Step 3 Finished.")
        return df, drop

    # ---------- step 4: bootstrap resolution ----------

    def step4_bootstrap(self, ref="true"):
        """Each reference gets its own generator, so a run is reproducible from the
        seed regardless of which references were bootstrapped before it."""
        rng = np.random.default_rng(self.seed + (0 if ref == "true" else 1))
        names = list(self.ms)
        if ref == "true":
            per = {n: self._true_per_sample(n) for n in names}
        else:
            per = {n: self._loo_score(n) for n in names}
        cp = [p for p in self.paths if all(p in per[n] for n in names)]
        E = {n: np.array([word_counts(per[n][p])[0] for p in cp], float) for n in names}
        # reference lengths are per-model: pseudo-references differ between models
        Lm = {n: np.array([word_counts(per[n][p])[1] for p in cp], float) for n in names}
        base = {n: 100.0 * E[n].sum() / Lm[n].sum() for n in names}
        order = sorted(names, key=base.get)
        pairs = list(itertools.combinations(order, 2))
        wins = {pair: 0 for pair in pairs}
        idx = np.arange(len(cp))
        for _ in range(self.n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            w = {n: E[n][s].sum() / Lm[n][s].sum() for n in names}
            for a, b in pairs:
                if w[a] < w[b]:
                    wins[(a, b)] += 1
        rows = []
        for (a, b), c in wins.items():
            rows.append({"pair": f"{a} < {b}", "adjacent": order.index(b) - order.index(a) == 1,
                         "gap_wer": base[b] - base[a], "preserved_frac": c / self.n_boot,
                         "resolved_95": c / self.n_boot >= 0.95})
        df = pd.DataFrame(rows).sort_values("gap_wer").reset_index(drop=True)
        res = df[df.resolved_95]
        mdg = float(res.gap_wer.min()) if len(res) else float("nan")
        unres_max = float(df[~df.resolved_95].gap_wer.max()) if (~df.resolved_95).any() else 0.0
        self.report[f"step4_pairs_{ref}"] = df
        self.report[f"step4_resolution_{ref}"] = {
            "reference": ref, "n_boot": self.n_boot, "n_samples": len(cp),
            "min_resolved_gap_wer": mdg, "max_unresolved_gap_wer": unres_max,
            "n_resolved": int(df.resolved_95.sum()), "n_pairs": len(df),
        }
        LOGGER.info(f"Step 4 Finished.")
        return df

    # ---------- step 5: baselines (is the ensemble necessary?) ----------

    def step5_baselines(self):
        """LOO ensemble vs full ensemble vs each single model as pseudo-reference.
        Single-model rows are computed on n_models-1 systems (a model cannot be its
        own reference); Spearman across different n is not strictly comparable, so
        matched ensemble rows on the same reduced set are reported alongside."""
        full_key = frozenset(self.voters)
        full_ref = self._fuse(self.voters)[0]

        def fidelity(names, ref_for, label):
            rows = []
            for name in names:
                rm, key = ref_for(name)
                sc = self._score(rm, name, key=key)
                ps = [p for p in self.paths if p in sc]
                rows.append({"model": name, "n_samples": len(ps),
                             "wer_true": micro_wer(self._true_per_sample(name), ps),
                             "wer_pseudo": micro_wer(sc, ps)})
            d=pd.DataFrame(rows)
            rk=self.exact_rank_p(d.wer_true.values,d.wer_pseudo.values)
            return {"reference":label,"n_models":len(d),**rk,
                    "mean_abs_wer_err":float((d.wer_true-d.wer_pseudo).abs().mean())}

        all_names = list(self.ms)
        out = [
            fidelity(all_names, lambda n: (self._fuse(self._loo_key(n))[0], self._loo_key(n)),
                     "LOO ensemble (proposed)"),
            fidelity(all_names, lambda n: (full_ref, full_key),
                     "full ensemble (biased)"),
        ]
        for m in self.voters:
            reduced = [n for n in all_names if n != m]
            sref = {p: self.ms[m][p]["normalized_prediction"] for p in self.paths}
            skey = ("single", m)
            out.append(fidelity(reduced, lambda n, sref=sref, skey=skey: (sref, skey),
                                f"single model: {m}"))
            out.append(fidelity(reduced, lambda n: (self._fuse(self._loo_key(n))[0], self._loo_key(n)),
                                f"LOO ensemble matched to n-1 (vs {m})"))
        df = pd.DataFrame(out)
        self.report["step5_baselines"] = df
        LOGGER.info(f"Step 5 Finished.")
        return df

    # ---------- step 6: pool ablation, two objectives ----------
    def _resolvable_pairs(self):
        """Unordered pairs the human references resolve at the 95% bootstrap bar.
        Read from step4_pairs_true; empty if step 4 has not been run."""
        d=self.report.get("step4_pairs_true")
        if d is None:
            return None
        out={}
        for _,r in d.iterrows():
            a,b=[t.strip() for t in r["pair"].split("<")]
            out[frozenset((a,b))]=bool(r["resolved_95"])
        return out

    @staticmethod
    def _pair_errors(d,res,pool):
        """Pairs whose pseudo-reference ordering contradicts the reference-based one.
        Returns (summary dict, list of per-pair rows). res=None leaves resolvability
        unknown, in which case the resolvable counts are nan and the flag is None."""
        t=d.set_index("model")["wer_true"].to_dict()
        p=d.set_index("model")["wer_pseudo_loo"].to_dict()
        wrong=0;wrong_res=0;worst=0.0;worst_pair="";rows=[]
        for a,b in itertools.combinations(t,2):
            if (p[a]<p[b])==(t[a]<t[b]):
                continue
            wrong+=1
            r=res.get(frozenset((a,b))) if res is not None else None
            g=abs(t[a]-t[b])
            tru=a if t[a]<t[b] else b
            pre=a if p[a]<p[b] else b
            rows.append({"pool":pool,"k":pool.count("+")+1,
                         "pair":f"{a}|{b}",
                         "true_winner":tru,"pred_winner":pre,
                         "wer_true_a":t[a],"wer_true_b":t[b],
                         "wer_pseudo_a":p[a],"wer_pseudo_b":p[b],
                         "gap_wer_true":g,"gap_wer_pseudo":abs(p[a]-p[b]),
                         "human_resolvable":r})
            if r:
                wrong_res+=1
                if g>worst:
                    worst=g;worst_pair=f"{a}|{b}"
        return ({"n_pairs_wrong":wrong,
                 "n_pairs_wrong_resolvable":float("nan") if res is None else wrong_res,
                 "worst_wrong_gap_wer":float("nan") if res is None else worst,
                 "worst_wrong_pair":worst_pair},rows)
    
    def step6_ablation(self, max_subsets=None):
        """
        For every voter subset (>= min_pool_size), report:
          obj_A ensemble_wer_vs_true   -> reference quality (what 'toxic model' removal optimizes)
          obj_B ranking_spearman       -> the actual thesis claim
        These can disagree. Report both; do not select on obj_A alone.
        obj_B is nan for pools whose LOO sub-pools fall below min_pool_size.
        """
        subs = [s for k in range(self.min_pool_size, len(self.voters) + 1)
                for s in itertools.combinations(self.voters, k)]
        if max_subsets:
            V=tuple(self.voters)
            req=[V]+[tuple(v for v in V if v!=m) for m in V]
            req=[s for s in req if len(s)>=self.min_pool_size]
            rest=[s for s in subs if s not in set(req)]
            subs=req+rest[:max(0,max_subsets-len(req))]
        res=self._resolvable_pairs()
        rows = [];errs=[]
        for sub in subs:
            sub = list(sub)
            refs = self._fuse(sub)[0]
            ens_m = {p: metrics.BasicSTTMetrics.evaluate(refs=self.true_ref[p], hyps=refs[p])
                     for p in self.paths if p in refs}
            objA = micro_wer(ens_m, list(ens_m))
            objA_cer = micro_cer(ens_m, list(ens_m))
            try:
                d,st=self.step2_ranking(pool=sub,store=False)
                objB,objB_p=st["spearman"],st["spearman_exact_p_onetailed"]
                objT,objT_p=st["kendall_b"],st["kendall_exact_p_onetailed"]
                summ,er=self._pair_errors(d,res,"+".join(sub))
                errs.extend(er)
            except ValueError:
                objB=objB_p=objT=objT_p=float("nan")
                summ={"n_pairs_wrong":float("nan"),
                      "n_pairs_wrong_resolvable":float("nan"),
                      "worst_wrong_gap_wer":float("nan"),"worst_wrong_pair":""}
            rows.append({"pool":"+".join(sub),"k":len(sub),
                         "ensemble_wer_vs_true":objA,"ensemble_cer_vs_true":objA_cer,
                         "ranking_spearman":objB,"ranking_exact_p":objB_p,
                         "ranking_kendall_b":objT,"ranking_kendall_exact_p":objT_p,
                         **summ})
        df = pd.DataFrame(rows)
        full = df[df.k == len(self.voters)].iloc[0]
        contrib = []
        for m in self.voters:
            row = df[df.pool == "+".join([v for v in self.voters if v != m])]
            if len(row):
                r = row.iloc[0]
                contrib.append({
                    "model_removed": m,
                    "delta_ensemble_wer": r.ensemble_wer_vs_true - full.ensemble_wer_vs_true,
                    "delta_ranking_spearman": r.ranking_spearman - full.ranking_spearman,
                    "helps_reference_quality": r.ensemble_wer_vs_true > full.ensemble_wer_vs_true,
                    "helps_ranking": bool(r.ranking_spearman >= full.ranking_spearman)
                                     if not np.isnan(r.ranking_spearman) else None,
                    "delta_pairs_wrong": r.n_pairs_wrong - full.n_pairs_wrong,
                    "delta_pairs_wrong_resolvable":
                        r.n_pairs_wrong_resolvable - full.n_pairs_wrong_resolvable,
                })
        self.report["step6_pools"] = df.sort_values("ensemble_wer_vs_true").reset_index(drop=True)
        self.report["step6_contribution"] = pd.DataFrame(contrib)
        ed = pd.DataFrame(errs)
        if len(ed):
            ed = ed.sort_values(["human_resolvable","gap_wer_true"],
                                ascending=[False,False]).reset_index(drop=True)
        self.report["step6_wrong_pairs"] = ed
        LOGGER.info(f"Step 6 Finished.")
        return df

    # ---------- step 7: floor-effect controls ----------

    def step7_floor(self):
        rows = []
        for name in self.ms:
            t = self._true_per_sample(name)
            l = self._loo_score(name)
            ps = [p for p in self.paths if p in l]
            tw = np.array([wer_pct(t[p]) for p in ps], float)
            lw = np.array([wer_pct(l[p]) for p in ps], float)
            te = np.array([word_counts(t[p])[0] for p in ps], float)
            le = np.array([word_counts(l[p])[0] for p in ps], float)
            nz = tw > 0
            rows.append({
                "model": name,
                "frac_zero_wer_samples": float((~nz).mean()),
                "kendall_b_all": stats.kendalltau(tw, lw, variant="b").correlation,
                "kendall_b_nonzero": stats.kendalltau(tw[nz], lw[nz], variant="b").correlation if nz.sum() > 2 else np.nan,
                "n_nonzero": int(nz.sum()),
                "kendall_b_editdist": stats.kendalltau(te, le, variant="b").correlation,
            })
        df = pd.DataFrame(rows).sort_values("frac_zero_wer_samples", ascending=False).reset_index(drop=True)
        self.report["step7_floor"] = df
        LOGGER.info(f"Step 7 Finished.")
        return df

    # ---------- step 8: permutation invariance ----------

    def step8_permutation(self,n_perm=5,include_reversed=True):
        """Fuse the full voter pool under random input orders and compare every
        transcript string by string with the canonical-order fusion. Order invariance
        holds by construction except for exact ties in reference selection, which are
        broken by input position; this counts how often that changes an output."""
        rng=np.random.default_rng(self.seed+8)
        canon=list(self.voters)
        base_refs,base_meta,_=self._fuse(canon)
        orders=[]
        if include_reversed:
            orders.append(canon[::-1])
        seen={tuple(canon)}|{tuple(o) for o in orders}
        while len(orders)<n_perm+int(include_reversed):
            o=list(rng.permutation(canon))
            if tuple(o) not in seen:
                seen.add(tuple(o));orders.append(o)
        rows=[];diffs=[]
        for k,order in enumerate(orders):
            ens=self.factory()
            ens.combine_models_transcriptions(*[self.ms[n] for n in order])
            ens.main(**self.main_kwargs)
            si=ens.samples_info
            same=0
            for p,ref in base_refs.items():
                out=si.get(p,{}).get("normalized_prediction")
                if out==ref:
                    same+=1
                    continue
                bm=base_meta[p]["reference_selection"];pm=si[p]["metadata"]["reference_selection"] if p in si else {}
                diffs.append({"perm":k,"order":"+".join(order),"path":p,
                              "canonical":ref,"permuted":out,
                              "refsel_canonical":bm.get("strategy_metric","unknown"),
                              "refsel_permuted":pm.get("strategy_metric","unknown")})
            rows.append({"perm":k,"order":"+".join(order),"n":len(base_refs),
                         "identical":same,"differing":len(base_refs)-same,
                         "identical_frac":same/len(base_refs) if base_refs else float("nan")})
            LOGGER.info(f"Step 8 permutation {k}: {same}/{len(base_refs)} identical.")
        df=pd.DataFrame(rows)
        self.report["step8_permutation"]=df
        self.report["step8_permutation_diffs"]=pd.DataFrame(diffs)
        self.report["step8_permutation_summary"]={
            "canonical_order":"+".join(canon),"n_orders":len(orders),
            "n_transcripts":len(base_refs),"total_differing":int(df.differing.sum()),
            "invariant":bool((df.differing==0).all()),"main_kwargs":dict(self.main_kwargs)}
        LOGGER.info(f"Step 8 Finished.")
        return df
    
    # ---------- driver ----------

    def run(self, steps=(0, 1, 2, 3, 4, 5, 6, 7), ablation_cap=None):
        if 0 in steps: self.step0_decorrelation()
        if 1 in steps: self.step1_bias()
        if 2 in steps: self.step2_ranking()
        if 3 in steps: self.step3_pipeline()
        if 4 in steps:
            self.step4_bootstrap(ref="true")
            self.step4_bootstrap(ref="pseudo")
        if 5 in steps: self.step5_baselines()
        if 6 in steps: self.step6_ablation(max_subsets=ablation_cap)
        if 7 in steps: self.step7_floor()
        if 8 in steps: self.step8_permutation()
        self._dump()
        self._print()
        return self.report

    def _dump(self):
        scalars = {}
        for k, v in self.report.items():
            if isinstance(v, pd.DataFrame):
                keep_index = k in ("step0_cross_wer", "step0_err_corr")
                v.to_csv(os.path.join(self.outdir, f"{k}{self.suffix}.csv"), index=keep_index)
            else:
                scalars[k] = v
        with open(os.path.join(self.outdir, f"scalars{self.suffix}.json"), "w") as f:
            json.dump(scalars, f, indent=2, default=float)

    def _print(self):
        for k, v in self.report.items():
            print(f"\n=== {k} ===")
            if isinstance(v, pd.DataFrame):
                print(v.round(4).to_string())
            else:
                print(json.dumps(v, indent=2, default=float))


def run_protocol(model_samples, voters, ensemble_factory,  steps=(0, 1, 2, 3, 4, 5, 6, 7),
                 ablation_cap=None, **kw):
    """Single entry point."""
    return EvaluationProtocol(model_samples, voters, ensemble_factory, **kw).run(steps=steps, ablation_cap=ablation_cap)