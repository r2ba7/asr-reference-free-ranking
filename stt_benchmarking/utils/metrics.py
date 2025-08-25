import jiwer

class StandardSTTMetrics:
    @staticmethod
    def _normalize_inputs(refs, hyps):
        """
        Normalize inputs to ensure refs and hyps are lists of strings.
        Convert single strings to single-element lists.
        """
        if not refs or not hyps:
            return [], []
        
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(hyps, str):
            hyps = [hyps]

        if len(refs) != len(hyps):
            return refs[:min(len(refs), len(hyps))], hyps[:min(len(refs), len(hyps))]

        return refs, hyps
    
    @staticmethod
    def word_accuracy(refs, hyps):
        if not refs or not hyps:
            return 0.0
            
        wer_score = jiwer.wer(refs, hyps)
        wer_score = min(wer_score, 1)
        word_accuracy_metric = 1 - wer_score
        word_accuracy_metric = round(word_accuracy_metric, 3)
        return word_accuracy_metric

    @staticmethod
    def character_accuracy(refs, hyps):
        if not refs or not hyps:
            return 0.0
            
        cer_score = jiwer.cer(refs, hyps)
        cer_score = min(cer_score, 1)
        character_accuracy_metric = 1 - cer_score
        character_accuracy_metric = round(character_accuracy_metric, 3)
        return character_accuracy_metric
    
    @staticmethod
    def compute_wer_accuracy(ref, hyp):
        """
        Compute word-level accuracy from WER.
        """
        if not ref or not hyp:
            return 0.0, 1.0
            
        sample_wer = jiwer.wer(ref, hyp)
        sample_wer = min(sample_wer, 1)
        return round(1 - sample_wer, 3)

    @staticmethod
    def compute_cer_accuracy(ref, hyp):
        """
        Compute character-level accuracy from CER.
        """
        if not ref or not hyp:
            return 0.0
            
        sample_cer = jiwer.cer(ref, hyp)
        sample_cer = min(sample_cer, 1)
        return round(1 - sample_cer, 3)
    
    @staticmethod
    def evaluate(refs, hyps, verbose=False, single_sample=False):
        """
        Compute average accuracy metrics for all samples.
        If single_sample=True, returns metrics for one sample.
        Zero accuracy samples are counted, no filtering applied.
        """
        refs, hyps = StandardSTTMetrics._normalize_inputs(refs, hyps)
        if not refs or not hyps:
            return {
                "word_accuracy": 0.0,
                "char_accuracy": 0.0,
                "average_score": 0.0,
                "zero_accuracy_count": 0
            }
        
        if single_sample:
            if len(refs) != 1 or len(hyps) != 1:
                # If not exactly one sample, take the first one or return zeros
                if refs and hyps:
                    ref, hyp = refs[0], hyps[0]
                else:
                    return {
                        "word_accuracy": 0.0,
                        "char_accuracy": 0.0,
                        # "bert_score": 0.0,
                        "average_score": 0.0,
                        "zero_accuracy_count": 1
                    }
            else:
                ref, hyp = refs[0], hyps[0]
            
            word_acc = StandardSTTMetrics.compute_wer_accuracy(ref, hyp)
            char_acc = StandardSTTMetrics.compute_cer_accuracy(ref, hyp)
            average_score = round((word_acc + char_acc) / 2, 3)
            zero_count = 1 if average_score == 0.0 else 0
            
            return {
                "word_accuracy": round(word_acc, 3),
                "char_accuracy": round(char_acc, 3),
                "average_score": average_score,
                "zero_accuracy_count": zero_count
            }
        
        # Handle batch processing
        word_accuracies = []
        char_accuracies = []
        zero_accuracy_count = 0
        
        for ref, hyp in zip(refs, hyps):
            word_acc, _ = StandardSTTMetrics.compute_wer_accuracy(ref, hyp)
            char_acc = StandardSTTMetrics.compute_cer_accuracy(ref, hyp)
            
            word_accuracies.append(word_acc)
            char_accuracies.append(char_acc)
            
            # Count samples with zero accuracy
            sample_avg = (word_acc + char_acc) / 2
            if sample_avg == 0.0:
                zero_accuracy_count += 1
                if verbose:
                    print(f"Zero accuracy sample - Word: {word_acc}, Char: {char_acc} | Ref: {ref} | Hyp: {hyp}")

        # bert_score_avg = StandardSTTMetrics.compute_bert_score(refs, hyps)
        word_avg = round(sum(word_accuracies) / len(word_accuracies), 3) if word_accuracies else 0.0
        char_avg = round(sum(char_accuracies) / len(char_accuracies), 3) if char_accuracies else 0.0
        average_score = round((word_avg + char_avg) / 2, 3)
        
        return {
            "word_accuracy": word_avg,
            "char_accuracy": char_avg,
            "average_score": average_score,
            "total_samples": len(refs),
            "zero_accuracy_count": zero_accuracy_count
        }