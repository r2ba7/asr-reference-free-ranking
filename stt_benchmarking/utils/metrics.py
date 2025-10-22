import jiwer

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