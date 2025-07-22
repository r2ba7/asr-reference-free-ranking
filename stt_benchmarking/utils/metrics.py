import jiwer
import Levenshtein

class S2TMetrics:
    @staticmethod
    def _normalize_inputs(refs, hyps):
        """
        Normalize inputs to ensure refs and hyps are lists of strings.
        Convert single strings to single-element lists.
        """
        if not refs or not hyps:
            raise ValueError("Input lists cannot be empty")
        
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(hyps, str):
            hyps = [hyps]

        if len(refs) != len(hyps):
            raise ValueError("Input lists must have equal length")

        return refs, hyps
    
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
    def evaluate(refs, hyps):
        refs, hyps = S2TMetrics._normalize_inputs(refs, hyps)
        return {
            "wer (%)": S2TMetrics.wer(refs, hyps),
            "cer (%)": S2TMetrics.cer(refs, hyps),
            "levenshtein_sim": S2TMetrics.normalized_levenshtein(refs, hyps),
        }