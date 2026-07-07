from difflib import SequenceMatcher
from collections import defaultdict, Counter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
from typing import Dict, Any, List, Tuple

import pandas as pd
from Levenshtein import distance
import numpy as np
from tqdm import tqdm
from editdistance import eval as edit_distance

from stt_benchmarking.utils import text_processing, helpers, metrics
from . import LOGGER, NORMALIZER_OBJ


class TranscriptFilter:
    def __init__(self, mode="mean"):
        self.mode = mode
        self.__verify_mode()
    
    def __verify_mode(self):
        if self.mode not in ["mean", "iqr"]:
            raise ValueError("mode should be either mean or iqr.")
        
    def _calculate_similarity_matrix(self, transcriptions: List[str]) -> pd.DataFrame:
        """Creates an N*N pairwise similarity matrix for all transcriptions."""
        def _word_sequence_ratio(words1: List[str], words2: List[str]) -> float:
            """Calculates the similarity ratio (0-100) based on matching *words*."""
            if not words1 and not words2: return 100.0
            if not words1 or not words2: return 0.0
            matcher = difflib.SequenceMatcher(None, words1, words2, autojunk=False)
            return matcher.ratio() * 100.0
        
        num_systems = len(transcriptions)
        tokenized_trans = [t.split() for t in transcriptions]
        df_similarity = pd.DataFrame(index=range(num_systems), columns=range(num_systems), dtype=float)
        for i in range(num_systems):
            for j in range(i, num_systems):
                if i == j:
                    similarity = 100.0
                else:
                    similarity = _word_sequence_ratio(tokenized_trans[i], tokenized_trans[j])
                
                df_similarity.loc[i, j] = similarity
                df_similarity.loc[j, i] = similarity
        return df_similarity
    
    def main(self, transcriptions: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Filters transcripts by pairwise agreement with dynamic threshold detection.
        
        Finds natural gap in agreement scores to separate high-agreement from low-agreement.
        """
        num_transcriptions = len(transcriptions)
        
        if num_transcriptions < 3:
            return transcriptions, {
                "status": "Skipped", 
                "reason": "Need ≥3 transcripts",
                "kept_indices": list(range(num_transcriptions)), 
                "filtered_indices": []
            }
        
        # Build similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(transcriptions).values
        agreement_scores = []
        for i in range(num_transcriptions):
            others_similarity = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            agreement_scores.append(np.mean(others_similarity))
        
        agreement_scores = np.array(agreement_scores)
        # Check for uniform scores
        if len(np.unique(agreement_scores)) == 1:
            return transcriptions, {
                "status": "Skipped",
                "reason": "All transcripts have identical agreement",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
        
        # Sort scores to find gaps
        sorted_indices = np.argsort(agreement_scores)
        sorted_scores = agreement_scores[sorted_indices]
        gaps = np.diff(sorted_scores)
        if self.mode == "mean":
            significant_gaps = np.where(gaps > np.mean(gaps) + np.std(gaps))[0]
        else:
            q1, q3 = np.percentile(gaps, [25, 75])
            iqr = q3 - q1
            upper_fence = q3 + (1.5 * iqr)
            significant_gaps = np.where(gaps > upper_fence)[0]

        if len(significant_gaps) == 0:
            return transcriptions, {
                "status": "Skipped",
                "reason": "Cannot compute gaps",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
        
        first_gap_idx = significant_gaps[0]
        dynamic_threshold = sorted_scores[first_gap_idx + 1]
        kept_indices = np.where(agreement_scores >= dynamic_threshold)[0]
        filtered_indices = np.where(agreement_scores < dynamic_threshold)[0]
        if len(kept_indices) == 0:
            best_idx = np.argmax(agreement_scores)
            kept_indices = np.array([best_idx])
            filtered_indices = np.setdiff1d(np.arange(num_transcriptions), kept_indices)
        
        filtered_transcriptions = [transcriptions[i] for i in kept_indices]
        metadata = {
            "status": "Success",
            "agreement_scores": agreement_scores.tolist(),
            "significant_gaps": significant_gaps.tolist(),
            "dynamic_threshold": float(dynamic_threshold),
            
            "kept_indices": kept_indices.tolist(),
            "filtered_indices": filtered_indices.tolist(),
            "num_models": len(transcriptions),
            "num_kept": len(kept_indices),
            "num_filtered": len(filtered_indices)
        }
        
        return filtered_transcriptions, metadata

class ROVER:
    """
    ROVER (Recognizer Output Voting Error Reduction) with optional greedy progressive fusion.

    Two modes:
      - Standard (greedy=False): build global WTN across all systems (full multi-sequence alignment)
      - Greedy (greedy=True): progressively fuse systems one by one (faster, local greedy alignment)
    """

    def __init__(self, model_weights=None, greedy=False):
        """
        Args:
            model_weights (list | None): Precomputed weights for each model. If None -> equal weights.
            greedy (bool): If True, performs progressive greedy fusion instead of full WTN.
        """
        self.model_weights = model_weights
        self.greedy = greedy
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None

    # -------------------------------------------------------------------------
    # Core combination
    # -------------------------------------------------------------------------
    def combine_models_transcriptions(self, *samples_dicts, missing_value=""):
        """
        Merge predictions from multiple models by audio_id.
        Returns dict: {audio_path: [t1, t2, ...]}
        """
        if not samples_dicts:
            return {}
        all_audio_paths = sorted({p for d in samples_dicts for p in d})
        combined = {}
        for path in all_audio_paths:
            combined[path] = [
                (d[path]["normalized_prediction"] if path in d else missing_value)
                for d in samples_dicts
            ]
        self._input_to_fusion = combined
        return combined

    # -------------------------------------------------------------------------
    # Word Transition Network (WTN)
    # -------------------------------------------------------------------------
    def build_word_transition_network(self, transcriptions):
        """
        Multi-sequence alignment producing a Word Transition Network (WTN).
        Greedy or full alignment depending on self.greedy.
        """
        def pairwise_align(words1, words2):
            if not words1 and not words2:
                return [], []
            if not words1:
                return [None] * len(words2), words2
            if not words2:
                return words1, [None] * len(words1)

            matcher = SequenceMatcher(None, words1, words2)
            a1, a2 = [], []
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == 'equal':
                    for k in range(i2 - i1):
                        a1.append(words1[i1 + k])
                        a2.append(words2[j1 + k])
                elif op == 'replace':
                    m = max(i2 - i1, j2 - j1)
                    for k in range(m):
                        w1 = words1[i1 + k] if i1 + k < i2 else None
                        w2 = words2[j1 + k] if j1 + k < j2 else None
                        a1.append(w1)
                        a2.append(w2)
                elif op == 'delete':
                    for k in range(i2 - i1):
                        a1.append(words1[i1 + k])
                        a2.append(None)
                elif op == 'insert':
                    for k in range(j2 - j1):
                        a1.append(None)
                        a2.append(words2[j1 + k])
            return a1, a2

        valid = [(i, t.strip()) for i, t in enumerate(transcriptions) if t and t.strip()]
        if not valid:
            return {"alignment_matrix": [], "num_slots": 0, "num_models": 0, "model_indices": []}
        if len(valid) == 1:
            idx, t = valid[0]
            words = t.split()
            return {"alignment_matrix": [[w] for w in words], "num_slots": len(words),
                    "num_models": 1, "model_indices": [idx]}

        # Progressive greedy alignment (faster, approximate)
        if self.greedy:
            idx0, first = valid[0]
            matrix = [[w] for w in first.split()]
            indices = [idx0]

            for idx, text in valid[1:]:
                new_words = text.split()
                consensus = [next((w for w in slot if w is not None), None) for slot in matrix]
                consensus_compact = [w for w in consensus if w is not None]
                a1, a2 = pairwise_align(consensus_compact, new_words)

                new_matrix = []
                cons_i = 0
                for cword, nword in zip(a1, a2):
                    if cword is not None:
                        new_matrix.append(matrix[cons_i] + [nword])
                        cons_i += 1
                    else:
                        new_matrix.append([None] * len(indices) + [nword])
                matrix = new_matrix
                indices.append(idx)
        else:
            # Full multi-sequence alignment (accurate but O(n^2))
            matrix = None
            indices = []
            for idx, text in valid:
                words = text.split()
                num_existing = len(indices)
                matrix = self._merge_into_alignment(matrix, words, num_existing, pairwise_align)
                indices.append(idx)

        return {
            "alignment_matrix": matrix,
            "num_slots": len(matrix) if matrix else 0,
            "num_models": len(valid),
            "model_indices": indices
        }

    @staticmethod
    def _merge_into_alignment(matrix, new_words, num_existing, pairwise_align_fn):
        if not matrix:
            return [[w] for w in new_words]
        consensus = [next((w for w in slot if w is not None), None) for slot in matrix]
        compact = [w for w in consensus if w is not None]
        a1, a2 = pairwise_align_fn(compact, new_words)
        new_matrix = []
        ci = 0
        for cw, nw in zip(a1, a2):
            if cw is not None:
                new_matrix.append(matrix[ci] + [nw])
                ci += 1
            else:
                new_matrix.append([None] * num_existing + [nw])
        return new_matrix

    # -------------------------------------------------------------------------
    # Voting
    # -------------------------------------------------------------------------
    def vote_on_wtn(self, wtn, weights=None):
        """
        Majority vote + edit-distance tie-break.
        """
        def vote_slot(slot, slot_weights):
            cands = [(i, w) for i, w in enumerate(slot) if w is not None]
            if not cands:
                return None, {}, 0.0
            if len(cands) == 1:
                w = cands[0][1]
                return w, {w: 1.0}, 1.0

            votes = defaultdict(float)
            for i, w in cands:
                votes[w] += slot_weights[i]
            max_vote = max(votes.values())
            tied = [w for w, v in votes.items() if v == max_vote]
            if len(tied) > 1:
                allw = [w for _, w in cands]
                scores = {tw: sum(edit_distance(tw, ow) for ow in allw if ow != tw) for tw in tied}
                mind = min(scores.values())
                best = [w for w, d in scores.items() if d == mind]
                chosen = min(best, key=lambda w: (len(w), w))
            else:
                chosen = tied[0]
            total = sum(votes.values())
            conf = votes[chosen] / total if total > 0 else 0.0
            return chosen, dict(votes), conf

        matrix = wtn["alignment_matrix"]
        if not matrix:
            return {"fusion_transcript": "", "fusion_tokens": [], "voting_details": [],
                    "confidence_score": 0.0}

        num_models = wtn["num_models"]
        model_indices = wtn["model_indices"]

        if weights is None:
            slot_weights = [1.0] * num_models
        else:
            slot_weights = [weights[i] for i in model_indices]
            s = sum(slot_weights)
            slot_weights = [w / s for w in slot_weights] if s else [1.0] * num_models

        fusion_tokens, details = [], []
        for slot_i, slot in enumerate(matrix):
            word, dist, conf = vote_slot(slot, slot_weights)
            fusion_tokens.append(word)
            details.append({
                "slot": slot_i,
                "chosen_word": word,
                "vote_distribution": dist,
                "confidence": conf,
                "candidates": [w for w in slot if w is not None]
            })

        transcript = " ".join([w for w in fusion_tokens if w is not None])
        avg_conf = float(np.mean([d["confidence"] for d in details])) if details else 0.0
        return {
            "fusion_transcript": transcript,
            "fusion_tokens": fusion_tokens,
            "voting_details": details,
            "confidence_score": avg_conf,
            "num_slots": len(fusion_tokens),
            "metadata": {
                "algorithm": "ROVER-greedy" if self.greedy else "ROVER",
                "num_models": num_models,
                "weighted": weights is not None
            }
        }
    
    def main(self, filteration_mode="mean", model_weights=None):
        """
        Main ROVER fusion pipeline.
        """
        def fuse_sample(audio_path, transcriptions):
            transcriptions_copy = list(transcriptions)
            fusion_start = time.time()
            filtered_transcriptions, filtration_metadata = TranscriptFilter(mode=filteration_mode).main(transcriptions_copy)
            wtn = self.build_word_transition_network(filtered_transcriptions)
            result = self.vote_on_wtn(wtn, model_weights)
            result["fusion_time"] = time.time() - fusion_start
            result["wtn"] = {
                "num_slots": wtn["num_slots"],
                "num_models": wtn["num_models"],
                "model_indices": wtn["model_indices"],
            }
                
            return {audio_path: result}
        
        def process_item(item):
            audio_path, transcriptions = item
            return fuse_sample(audio_path, transcriptions)
        
        if not self.input_to_fusion:
            raise ValueError("Run ROVEREnsemble.combine_models_transcriptions first.")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item 
                      for item in self.input_to_fusion.items()}
            fusion_results = {}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="ROVER Fusion..."):
                result = future.result()
                fusion_results.update(result)
        
        self._fusion_results = dict(sorted(fusion_results.items()))
    
    def eval(self, audios_chunk):
        """
        Evaluate fusion results against ground truth.
        """
        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        for audio_path in list(self._fusion_results.keys()):
            if audio_path in self._fusion_results:
                try:
                    ref = refs_lookup[audio_path]
                    hyp = self._fusion_results[audio_path]["fusion_transcript"]
                    norm_hyp = NORMALIZER_OBJ(hyp)
                    self._fusion_results[audio_path]["normalized_prediction"] = norm_hyp
                    self._fusion_results[audio_path]["normalized_transcription"] = ref   
                    if ref is not None:
                        sample_metrics = metrics.BasicSTTMetrics.evaluate(
                            refs=ref, hyps=norm_hyp)
                        self._fusion_results[audio_path]["metrics"] = sample_metrics
                    else:
                        self._fusion_results[audio_path]["metrics"] = helpers._empty_metrics()
                
                except Exception as e:
                    LOGGER.error(f"Error processing {audio_path}: {e}")
                    self._fusion_results[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._fusion_results[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }
        # Overall metrics
        refs = [v["normalized_transcription"] for v in self._fusion_results.values()]
        hyps = [v["normalized_prediction"] for v in self._fusion_results.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)
    
    def summary_of_evaluation(self):
        """
        Display evaluation summary.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        
        LOGGER.info("ROVER Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
    
    def reset(self):
        """
        Reset the ROVER instance.
        """
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
        LOGGER.info("ROVER ensemble instance has been reset.")
    
    @property
    def input_to_fusion(self):
        if not self._input_to_fusion:
            raise ValueError("Run ROVEREnsemble.combine_models_transcriptions first.")
        return self._input_to_fusion
    
    @input_to_fusion.setter
    def input_to_fusion(self, value):
        if not isinstance(value, dict):
            raise ValueError("Input to fusion must be a dictionary.")
        self._input_to_fusion = value
    
    @property
    def fusion_results(self):
        if not self._fusion_results:
            raise ValueError("Run ROVEREnsemble.fusion first.")
        return self._fusion_results
    
    @property
    def overall_metrics(self):
        return self._overall_metrics