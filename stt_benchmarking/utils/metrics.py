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
    def word_accuracy(refs, hyps):
        wer_score = jiwer.wer(refs, hyps)
        wer_score = min(wer_score, 1)
        word_accuracy_metric = 1 - wer_score
        word_accuracy_metric = round(word_accuracy_metric, 3)
        return word_accuracy_metric

    @staticmethod
    def character_accuracy(refs, hyps):
        cer_score = jiwer.cer(refs, hyps)
        cer_score = min(cer_score, 1)
        character_accuracy_metric = 1 - cer_score
        character_accuracy_metric = round(character_accuracy_metric, 3)
        return character_accuracy_metric
    
    @staticmethod
    def average_metrics(refs, hyps):
        """
        Compute the average of all defined similarity metrics.
        """
        metrics = S2TMetrics.evaluate(refs, hyps)
        avg = round(sum(metrics.values()) / len(metrics), 3)
        metrics["average"] = avg
        return metrics
    
    @staticmethod
    def evaluate(refs, hyps):
        refs, hyps = S2TMetrics._normalize_inputs(refs, hyps)

        metrics = {
            "word_accuracy": S2TMetrics.word_accuracy(refs, hyps),
            "char_accuracy": S2TMetrics.character_accuracy(refs, hyps),
            "levenshtein_sim": S2TMetrics.normalized_levenshtein(refs, hyps),
        }

        # Add average of all similarity metrics
        metrics["average_metric"] = round(sum(metrics.values()) / len(metrics), 3)
        return metrics