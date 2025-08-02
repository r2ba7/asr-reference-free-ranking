import jiwer
from bert_score import score

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
    def bert_score_similarity(refs, hyps):
        """
        Compute BERTScore F1 similarity for Arabic text.
        Uses multilingual BERT model that supports Arabic.
        """
        # Use multilingual BERT that supports Arabic
        # Alternative: "aubmindlab/bert-base-arabertv2" for better Arabic support
        P, R, F1 = score(hyps, refs, 
                         model_type="bert-base-multilingual-cased", 
                         lang="ar",
                         verbose=False)
        
        # Return average F1 score
        avg_f1 = F1.mean().item()
        return round(avg_f1, 3)
    
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
    def evaluate(refs, hyps):
        refs, hyps = S2TMetrics._normalize_inputs(refs, hyps)

        metrics = {
            "word_accuracy": S2TMetrics.word_accuracy(refs, hyps),
            "char_accuracy": S2TMetrics.character_accuracy(refs, hyps),
            # "bert_score": S2TMetrics.bert_score_similarity(refs, hyps),
        }

        # Add average of all similarity metrics
        metrics["average_score"] = round(sum(metrics.values()) / len(metrics), 3)
        return metrics
    
class FilteredS2TMetrics:
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
    def compute_wer_accuracy(ref, hyp):
        """
        Compute word-level accuracy from WER.
        """
        sample_wer = jiwer.wer(ref, hyp)
        sample_wer = min(sample_wer, 1)
        return round(1 - sample_wer, 3), sample_wer

    @staticmethod
    def compute_cer_accuracy(ref, hyp):
        """
        Compute character-level accuracy from CER.
        """
        sample_cer = jiwer.cer(ref, hyp)
        sample_cer = min(sample_cer, 1)
        return round(1 - sample_cer, 3)

    @staticmethod
    def compute_bert_score(refs, hyps):
        """
        Compute average BERTScore F1 for a batch of Arabic sentences.
        """
        if not refs or not hyps:
            return 0.0
        P, R, F1 = score(hyps, refs,
                         model_type="bert-base-multilingual-cased",
                         lang="ar", verbose=False)
        avg_f1 = F1.mean().item()
        return round(avg_f1, 3)

    @staticmethod
    def evaluate(refs, hyps, wer_threshold=0.9, verbose=False, single_sample=False):
        """
        Compute average accuracy metrics for samples passing WER threshold.
        If single_sample=True, returns metrics for one sample or None if it doesn't meet criteria.
        """
        refs, hyps = FilteredS2TMetrics._normalize_inputs(refs, hyps)
        
        # Handle single sample case
        if single_sample:
            if len(refs) != 1 or len(hyps) != 1:
                raise ValueError("single_sample=True requires exactly one reference and hypothesis")
            
            ref, hyp = refs[0], hyps[0]
            word_acc, sample_wer = FilteredS2TMetrics.compute_wer_accuracy(ref, hyp)
            if sample_wer > wer_threshold:
                if verbose:
                    print(f"Excluded sample - WER: {sample_wer:.3f} | Ref: {ref} | Hyp: {hyp}")
                return {
                    "word_accuracy": None,
                    "char_accuracy": None,
                    "bert_score": None,
                    "average_score": None,
                }
            
            char_acc = FilteredS2TMetrics.compute_cer_accuracy(ref, hyp)
            # bert_score = FilteredS2TMetrics.compute_bert_score([ref], [hyp])
            average_score = round((word_acc + char_acc) / 3, 3)
            
            return {
                "word_accuracy": round(word_acc, 3),
                "char_accuracy": round(char_acc, 3),
                # "bert_score": bert_score,
                "average_score": average_score,
            }
        
        # Handle batch processing (original logic)
        valid_refs = []
        valid_hyps = []
        word_accuracies = []
        char_accuracies = []
        non_valid = 0
        for ref, hyp in zip(refs, hyps):
            word_acc, sample_wer = FilteredS2TMetrics.compute_wer_accuracy(ref, hyp)
            if sample_wer > wer_threshold:
                if verbose:
                    print(f"Excluded sample - WER: {sample_wer:.3f} | Ref: {ref} | Hyp: {hyp}")
                non_valid += 1
                continue

            char_acc = FilteredS2TMetrics.compute_cer_accuracy(ref, hyp)
            valid_refs.append(ref)
            valid_hyps.append(hyp)
            word_accuracies.append(word_acc)
            char_accuracies.append(char_acc)

        if not valid_refs:
            return {
                "word_accuracy": 0.0,
                "char_accuracy": 0.0,
                # "bert_score": 0.0,
                "average_score": 0.0,
                "num_valid_samples": 0
            }

        # bert_score_avg = FilteredS2TMetrics.compute_bert_score(valid_refs, valid_hyps)
        word_avg = round(sum(word_accuracies) / len(word_accuracies), 3)
        char_avg = round(sum(char_accuracies) / len(char_accuracies), 3)
        average_score = round((word_avg + char_avg) / 3, 3)

        return {
            "word_accuracy": word_avg,
            "char_accuracy": char_avg,
            # "bert_score": bert_score_avg,
            "average_score": average_score,
            "num_valid_samples": f"{len(valid_refs)}/{len(refs)}",
            "num_excluded_samples": non_valid
        }
