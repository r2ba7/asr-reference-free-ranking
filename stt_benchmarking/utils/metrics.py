import jiwer
import Levenshtein

class S2TMetrics:

    @staticmethod
    def normalized_levenshtein(refs, hyps):
        sims = []
        for r, h in zip(refs, hyps):
            max_len = max(len(r), len(h), 1)
            dist = Levenshtein.distance(r, h)
            sim = 1 - dist / max_len
            sims.append(sim)
        return round(sum(sims) / len(sims), 3)
    
    @staticmethod
    def wer(refs, hyps):
        wer_score = jiwer.wer(refs, hyps) * 100
        return round(min(wer_score, 100.0), 3)

    @staticmethod
    def cer(refs, hyps):
        cer_score = jiwer.cer(refs, hyps) * 100
        return round(min(cer_score, 100.0), 3)

    @staticmethod
    def avg_length_diff(refs, hyps):
        diffs = [abs(len(h.split()) - len(r.split())) for r, h in zip(refs, hyps)]
        return round(sum(diffs) / len(diffs), 3)

    @staticmethod
    def length_ratios(refs, hyps):
        # Gives length ratio per sample (not rounded or percentage, used for analysis)
        return [round(len(h.split()) / max(1, len(r.split())), 3) for r, h in zip(refs, hyps)]

    @staticmethod
    def evaluate(refs, hyps):
        return {
            "wer (%)": S2TMetrics.wer(refs, hyps),
            "cer (%)": S2TMetrics.cer(refs, hyps),
            "levenshtein_sim": S2TMetrics.normalized_levenshtein(refs, hyps),
            "avg_length_diff": S2TMetrics.avg_length_diff(refs, hyps),
        }