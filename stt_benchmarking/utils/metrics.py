import jiwer

class BasicSTTMetrics:
    @staticmethod
    def _normalize_inputs(refs, hyps):
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(hyps, str):
            hyps = [hyps]

        # guarantee at least empty string instead of None
        if not hyps:
            hyps = [""]

        if len(refs) != len(hyps):
            n = min(len(refs), len(hyps))
            refs, hyps = refs[:n], hyps[:n]

        return refs, hyps

    @staticmethod
    def wer_details(refs, hyps):
        measures = jiwer.compute_measures(refs, hyps)

        ref_words = " ".join(refs).split()
        hyp_words = " ".join(hyps).split()
        ref_len = len(ref_words)
        hyp_len = len(hyp_words)
        wer = round(measures["wer"] * 100, 2)
        return {
            "wer (%)": wer,
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

        ref_len = sum(len(r) for r in refs)
        hyp_len = sum(len(h) for h in hyps)

        cer = round(out.wer * 100, 2)
        return {
            "cer (%)": cer,
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