import jiwer
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr

class BasicSTTMetrics:
    @staticmethod
    def _normalize_inputs(refs, hyps):
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(hyps, str):
            hyps = [hyps]

        if not hyps:
            hyps = [""]

        return refs, hyps

    @staticmethod
    def wer_details(refs, hyps):
        measures = jiwer.compute_measures(refs, hyps)

        # Word Levenshtein Distance = S + D + I
        word_edit_distance = (
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )

        # N (Total words in reference) = H + S + D
        ref_len = (
            measures["hits"]
            + measures["substitutions"]
            + measures["deletions"]
        )
        
        # Total words in hypothesis = H + S + I
        hyp_len = (
            measures["hits"]
            + measures["substitutions"]
            + measures["insertions"]
        )

        wer = round(measures["wer"] * 100, 2)
        
        # --- ADDED: Levenshtein Similarity Ratio ---
        denominator = ref_len + hyp_len
        if denominator == 0:
            # Handle edge case where both ref and hyp are empty
            word_similarity = 1.0
        else:
            word_similarity = (denominator - word_edit_distance) / denominator
        # ---------------------------------------------

        return {
            "wer (%)": wer,
            "word_similarity_ratio (%)": round(word_similarity * 100, 2), # Added
            "word_edit_distance": word_edit_distance,
            "substitutions": measures["substitutions"],
            "deletions": measures["deletions"],
            "insertions": measures["insertions"],
            "hits": measures["hits"],
            "ref_length": ref_len,
            "hyp_length": hyp_len,
        }

    @staticmethod
    def cer_details(refs, hyps):
        def char_transform(texts):
            return [list(t.strip()) for t in texts]

        out = jiwer.process_words(
            refs, hyps,
            reference_transform=char_transform,
            hypothesis_transform=char_transform
        )

        # Character Levenshtein Distance = S + D + I
        char_edit_distance = (
            out.substitutions + out.deletions + out.insertions
        )

        # N (Total chars in reference, post-transform) = H + S + D
        ref_len = out.hits + out.substitutions + out.deletions
        
        # Total chars in hypothesis (post-transform) = H + S + I
        hyp_len = out.hits + out.substitutions + out.insertions

        cer = round(out.wer * 100, 2)
        
        # --- ADDED: Levenshtein Similarity Ratio ---
        denominator = ref_len + hyp_len
        if denominator == 0:
            # Handle edge case where both ref and hyp are empty
            char_similarity = 1.0
        else:
            char_similarity = (denominator - char_edit_distance) / denominator
        # ---------------------------------------------
        
        return {
            "cer (%)": cer,
            "char_similarity_ratio (%)": round(char_similarity * 100, 2), # Added
            "char_edit_distance": char_edit_distance,
            "substitutions": out.substitutions,
            "deletions": out.deletions,
            "insertions": out.insertions,
            "hits": out.hits,
            "ref_length": ref_len,
            "hyp_length": hyp_len,
        }
    
    @staticmethod
    def evaluate(refs, hyps):
        refs, hyps = BasicSTTMetrics._normalize_inputs(refs, hyps)
        return {
            "word_error_rate": BasicSTTMetrics.wer_details(refs, hyps),
            "character_error_rate": BasicSTTMetrics.cer_details(refs, hyps),
        }

# assumes `mt` (your metrics module) is importable in the calling scope,
# i.e. mt.BasicSTTMetrics.evaluate(refs=..., hyps=...) as used elsewhere.


def add_pseudo_reference_wer(df_filtered, ensemble_samples, transcript_key="fusion_transcript"):
    """
    Add per-sample pseudo-reference WER/CER: each model's normalized_prediction
    scored against the ensemble pseudo-groundtruth for the same audio_id.

    df_filtered : long df from collect_model_metrics + filter_audio_records
    ensemble_samples : list[dict] of ensemble trials (fusion_transcript per audio)
    Returns df_filtered with added columns: pseudo_wer, pseudo_cer.
    """
    ens = {}
    for trial in ensemble_samples:
        for audio_id, c in trial.items():
            ens[audio_id] = c.get(transcript_key, c.get("normalized_prediction", ""))

    pseudo_wer, pseudo_cer = [], []
    for _, row in df_filtered.iterrows():
        ref = ens.get(row["audio_id"])
        hyp = row["normalized_prediction"]
        if ref is None or hyp is None or not str(ref).strip():
            pseudo_wer.append(np.nan)
            pseudo_cer.append(np.nan)
            continue
        r = mt.BasicSTTMetrics.evaluate(refs=ref, hyps=hyp)
        pseudo_wer.append(r["word_error_rate"]["wer (%)"])
        pseudo_cer.append(r["character_error_rate"]["cer (%)"])

    df = df_filtered.copy()
    df["pseudo_wer"] = pseudo_wer
    df["pseudo_cer"] = pseudo_cer
    return df


def _corr(x, y):
    if len(x) < 3 or pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return None
    sp = spearmanr(x, y)
    kt = kendalltau(x, y)
    pr = pearsonr(x, y)
    return {
        "pearson": pr.statistic, "pearson_p": pr.pvalue,
        "spearman": sp.statistic, "spearman_p": sp.pvalue,
        "kendall": kt.statistic, "kendall_p": kt.pvalue,
    }


def per_sample_correlation(df_pseudo, exclude_models=("ensemble", "rover"),
                           metric_pair=("wer", "pseudo_wer"), min_pairs=30):
    """
    Option A. Per-sample WER tracking: correlate true WER vs pseudo-reference WER.

    Returns a df with one row per model (within-model, across samples) plus an
    'ALL' row (pooled over every sample-model pair, the direct NoRefER analogue).
    """
    true_col, pseudo_col = metric_pair
    sub = df_pseudo[~df_pseudo["model"].isin(exclude_models)].copy()
    rows = []

    for model, g in sub.groupby("model"):
        d = g[[true_col, pseudo_col]].dropna()
        if len(d) < min_pairs:
            continue
        c = _corr(d[true_col].values, d[pseudo_col].values)
        if c:
            rows.append({"model": model, "n": len(d), **c})

    d_all = sub[[true_col, pseudo_col]].dropna()
    c_all = _corr(d_all[true_col].values, d_all[pseudo_col].values)
    if c_all:
        rows.append({"model": "ALL (pooled)", "n": len(d_all), **c_all})

    return pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)


def per_sample_ranking(df_pseudo, exclude_models=("ensemble", "rover"),
                       metric_cols=("wer", "pseudo_wer"),
                       tiebreak_cols=("cer", "pseudo_cer"),
                       min_models=3):
    """
    Per-sample ranking agreement, WER primary with CER as tiebreak only.

    For each sample, systems are ordered by true (WER, then CER) and by
    pseudo (WER, then CER); CER is consulted solely to break exact WER ties,
    so the axis remains a WER ranking, not a WER+CER blend. The two orderings
    are correlated within the sample, then aggregated across samples.

    Returns a one-row summary df with mean/median/std of Spearman and Kendall,
    usable/dropped sample counts, and the residual WER tie fraction.
    """
    true_w, pseudo_w = metric_cols
    true_c, pseudo_c = tiebreak_cols
    sub = df_pseudo[~df_pseudo["model"].isin(exclude_models)].copy()

    sp_vals, kt_vals, tie_fracs = [], [], []
    n_dropped = 0

    for audio_id, g in sub.groupby("audio_id"):
        d = g[[true_w, pseudo_w, true_c, pseudo_c]].dropna()
        if len(d) < min_models:
            n_dropped += 1
            continue

        # residual WER-only tie fraction, before CER breaks them (diagnostic)
        _, counts = np.unique(d[true_w].values, return_counts=True)
        tie_fracs.append(1.0 - len(counts) / len(d))

        # composite ranks: WER primary, CER tiebreak. lexsort keys are
        # applied last-key-first, so primary key goes last.
        true_rank = np.lexsort((d[true_c].values, d[true_w].values)).argsort()
        pseudo_rank = np.lexsort((d[pseudo_c].values, d[pseudo_w].values)).argsort()

        if pd.Series(true_rank).nunique() < 2 or pd.Series(pseudo_rank).nunique() < 2:
            n_dropped += 1
            continue

        sp_vals.append(spearmanr(true_rank, pseudo_rank).statistic)
        kt_vals.append(kendalltau(true_rank, pseudo_rank).statistic)

    summary = {
        "n_samples_used": len(sp_vals),
        "n_samples_dropped": n_dropped,
        "mean_wer_ties": float(np.mean(tie_fracs)) if tie_fracs else np.nan,
        "spearman_mean": float(np.nanmean(sp_vals)) if sp_vals else np.nan,
        "spearman_median": float(np.nanmedian(sp_vals)) if sp_vals else np.nan,
        "spearman_std": float(np.nanstd(sp_vals)) if sp_vals else np.nan,
        "kendall_mean": float(np.nanmean(kt_vals)) if kt_vals else np.nan,
        "kendall_median": float(np.nanmedian(kt_vals)) if kt_vals else np.nan,
        "kendall_std": float(np.nanstd(kt_vals)) if kt_vals else np.nan,
    }
    return pd.DataFrame([summary])


def ranking_correlation(df_absolute, df_relative, exclude_models=("ensemble", "rover")):
    """
    Option B (headline). Correlate the reference ranking against the
    pseudo-reference ranking over models, on full-set aggregate WER.

    df_absolute : output of evaluate_absolute_metrics
    df_relative : output of evaluate_relative_metrics
    Returns (merged_df, coefficients_dict).
    """
    a = df_absolute[["model", "WER (%)"]].rename(columns={"WER (%)": "wer_abs"})
    r = df_relative[["model", "WER (%)"]].rename(columns={"WER (%)": "wer_rel"})
    m = a.merge(r, on="model", how="inner")
    m = m[~m["model"].isin(exclude_models)].reset_index(drop=True)

    c = _corr(m["wer_abs"].values, m["wer_rel"].values)
    return m, c