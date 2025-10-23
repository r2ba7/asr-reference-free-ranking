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

from stt_benchmarking.utils import text_processing, helpers, metrics
from . import LOGGER



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
            matcher = difflib.SequenceMatcher(None, words1, words2)
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
            # "largest_gap": float(gaps[largest_gap_idx]),
            "kept_indices": kept_indices.tolist(),
            "filtered_indices": filtered_indices.tolist(),
            "num_models": len(transcriptions),
            "num_kept": len(kept_indices),
            "num_filtered": len(filtered_indices)
        }
        
        return filtered_transcriptions, metadata

class ROVEREnsemble:
    """
    ROVER (Recognizer Output Voting Error Reduction) implementation.
    
    Classic ROVER algorithm:
    1. Align all hypotheses using word-level alignment (no timing info needed)
    2. Create Word Transition Network (WTN) with aligned positions
    3. Majority voting at each position with frequency-based selection
    4. Character-level edit distance for tie-breaking
    """
    
    def __init__(self, model_weights=None):
        """
        Args:
            model_weights (list): Optional pre-computed weights for each model
                                 If None, uses equal weights (standard ROVER)
        """
        self.model_weights = model_weights
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
    
    def combine_models_transcriptions(self, *samples_dicts, missing_value=""):
        """
        Align multiple samples_info dicts into an audio-centric structure.
        
        Args:
            *samples_dicts: Each is a dict like {audio_path: {normalized_prediction: str}}
            missing_value (str): Placeholder when a dict has no prediction for a sample.
        
        Returns:
            dict: {audio_path: [transcript_from_model1, transcript_from_model2, ...]}
        """
        if not samples_dicts:
            return {}
        
        all_audio_paths = sorted({path for d in samples_dicts for path in d.keys()})
        combined = {}
        for audio_path in all_audio_paths:
            combined[audio_path] = []
            for d in samples_dicts:
                if audio_path in d:
                    transcript = d[audio_path].get("normalized_prediction", missing_value)
                else:
                    transcript = missing_value
                combined[audio_path].append(transcript)
        
        self._input_to_fusion = combined
    
    def build_word_transition_network(self, transcriptions):
        """
        Build Word Transition Network (WTN) using multi-sequence alignment.
        
        "Time slot" in ROVER = aligned word position (not actual time):
        - Align word sequences to find corresponding positions
        - Each column in alignment matrix = one "time slot" (word position)
        - Use SequenceMatcher to find matching/mismatching word positions
        
        Args:
            transcriptions (list): List of transcription strings
            
        Returns:
            dict: WTN structure with aligned tokens and metadata
        """
        def pairwise_align(words1, words2):
            """
            Align two word sequences.
            Returns two lists of same length with None for gaps.
            """
            if not words1 and not words2:
                return [], []
            if not words1:
                return [None] * len(words2), words2
            if not words2:
                return words1, [None] * len(words1)
            
            matcher = SequenceMatcher(None, words1, words2)
            aligned1, aligned2 = [], []
            
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == 'equal':
                    # Words match - add them aligned
                    for idx in range(i2 - i1):
                        aligned1.append(words1[i1 + idx])
                        aligned2.append(words2[j1 + idx])
                        
                elif op == 'replace':
                    # Words differ at same position
                    max_len = max(i2 - i1, j2 - j1)
                    for idx in range(max_len):
                        w1 = words1[i1 + idx] if i1 + idx < i2 else None
                        w2 = words2[j1 + idx] if j1 + idx < j2 else None
                        aligned1.append(w1)
                        aligned2.append(w2)
                        
                elif op == 'delete':
                    # Words only in sequence 1
                    for idx in range(i2 - i1):
                        aligned1.append(words1[i1 + idx])
                        aligned2.append(None)
                        
                elif op == 'insert':
                    # Words only in sequence 2
                    for idx in range(j2 - j1):
                        aligned1.append(None)
                        aligned2.append(words2[j1 + idx])
            
            return aligned1, aligned2
        
        def merge_into_alignment(alignment_matrix, new_words, num_existing_models):
            """
            Merge new word sequence into existing alignment matrix.
            
            alignment_matrix: list of slots, each slot = list of words (one per model)
            new_words: list of words from new model
            """
            if not alignment_matrix:
                # First sequence - create matrix
                return [[word] for word in new_words]
            
            # Extract consensus from current alignment
            consensus_words = []
            for slot in alignment_matrix:
                # Pick first non-None word as representative
                non_null = [w for w in slot if w is not None]
                consensus_words.append(non_null[0] if non_null else None)
            
            # Remove Nones for alignment
            consensus_compact = [w for w in consensus_words if w is not None]
            
            # Align new sequence to consensus
            aligned_consensus, aligned_new = pairwise_align(consensus_compact, new_words)
            
            # Build new alignment matrix
            new_matrix = []
            consensus_idx = 0
            
            for cons_word, new_word in zip(aligned_consensus, aligned_new):
                if cons_word is not None:
                    # Extend existing slot
                    new_matrix.append(alignment_matrix[consensus_idx] + [new_word])
                    consensus_idx += 1
                else:
                    # Create new slot for insertion
                    # Pad with None for all existing models
                    new_matrix.append([None] * num_existing_models + [new_word])
            
            return new_matrix
        
        # Filter valid transcriptions
        valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) 
                               if t and t.strip()]
        
        if not valid_transcriptions:
            return {
                "alignment_matrix": [],
                "num_slots": 0,
                "num_models": 0,
                "model_indices": []
            }
        
        if len(valid_transcriptions) == 1:
            idx, trans = valid_transcriptions[0]
            words = trans.split()
            return {
                "alignment_matrix": [[w] for w in words],
                "num_slots": len(words),
                "num_models": 1,
                "model_indices": [idx]
            }
        
        # Progressive alignment: align sequences one by one
        alignment_matrix = None
        model_indices = []
        
        for idx, trans in valid_transcriptions:
            words = trans.split()
            num_existing = len(model_indices)
            alignment_matrix = merge_into_alignment(alignment_matrix, words, num_existing)
            model_indices.append(idx)
        
        return {
            "alignment_matrix": alignment_matrix,
            "num_slots": len(alignment_matrix) if alignment_matrix else 0,
            "num_models": len(valid_transcriptions),
            "model_indices": model_indices
        }
    
    def vote_on_wtn(self, wtn, weights=None):
        """
        Standard ROVER voting: majority vote + edit distance tie-breaking.
        
        For each slot (word position):
        1. Count occurrences of each word (weighted if weights provided)
        2. Select most frequent word
        3. If tie: use character-level edit distance to pick most "central" word
        
        Edit distance tie-breaking:
        - For tied words, compute sum of edit distances to ALL other candidates
        - Pick word with minimum total distance (most similar to others)
        
        Args:
            wtn (dict): Word Transition Network structure
            weights (list): Optional weights for voting (None = equal weights)
            
        Returns:
            dict: Voting results
        """
        def vote_slot(slot, slot_weights):
            """
            Vote on single slot using frequency + edit distance.
            """
            # Extract non-null candidates with their model indices
            candidates = [(i, word) for i, word in enumerate(slot) if word is not None]
            
            if not candidates:
                return None, {}, 0.0
            
            if len(candidates) == 1:
                word = candidates[0][1]
                return word, {word: 1.0}, 1.0
            
            # Weighted voting (count occurrences weighted by model confidence)
            vote_counts = defaultdict(float)
            for model_idx, word in candidates:
                vote_counts[word] += slot_weights[model_idx]
            
            # Find max vote
            max_votes = max(vote_counts.values())
            tied_words = [word for word, votes in vote_counts.items() 
                         if votes == max_votes]
            
            # Tie-breaking with edit distance
            if len(tied_words) > 1:
                # Compute total edit distance to all other candidates
                all_candidate_words = [word for _, word in candidates]
                
                distance_scores = {}
                for tied_word in tied_words:
                    total_distance = sum(
                        distance(tied_word, other) 
                        for other in all_candidate_words 
                        if other != tied_word
                    )
                    distance_scores[tied_word] = total_distance
                
                # Pick word with minimum total distance (most central)
                min_distance = min(distance_scores.values())
                best_words = [w for w, d in distance_scores.items() 
                            if d == min_distance]
                
                # Final tie-break: shortest word, then lexicographic
                chosen = min(best_words, key=lambda w: (len(w), w))
            else:
                chosen = tied_words[0]
            
            # Confidence = vote proportion
            total_votes = sum(vote_counts.values())
            confidence = vote_counts[chosen] / total_votes if total_votes > 0 else 0.0
            
            return chosen, dict(vote_counts), confidence
        
        alignment_matrix = wtn["alignment_matrix"]
        num_models = wtn["num_models"]
        model_indices = wtn["model_indices"]
        
        # Setup voting weights
        if weights is None:
            # Standard ROVER: equal weights
            slot_weights = [1.0] * num_models
        else:
            # Use provided model weights (e.g., based on WER)
            slot_weights = [weights[i] for i in model_indices]
            # Normalize
            total = sum(slot_weights)
            slot_weights = [w / total for w in slot_weights]
        
        # Vote on each slot (word position)
        fusion_tokens = []
        voting_details = []
        
        for slot_idx, slot in enumerate(alignment_matrix):
            chosen_word, vote_distribution, confidence = vote_slot(slot, slot_weights)
            
            fusion_tokens.append(chosen_word)
            voting_details.append({
                "slot": slot_idx,
                "chosen_word": chosen_word,
                "vote_distribution": vote_distribution,
                "confidence": confidence,
                "all_candidates": [w for w in slot if w is not None]
            })
        
        # Build final transcript
        fusion_transcript = " ".join([w for w in fusion_tokens if w is not None])
        
        # Overall confidence
        avg_confidence = (np.mean([d["confidence"] for d in voting_details]) 
                         if voting_details else 0.0)
        
        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "voting_details": voting_details,
            "confidence_score": avg_confidence,
            "num_slots": len(fusion_tokens),
            "metadata": {
                "algorithm": "ROVER",
                "num_models": num_models,
                "weighted": weights is not None
            }
        }
    
    def fusion(self, filteration_mode="mean"):
        """
        Main ROVER fusion pipeline.
        """
        def fuse_sample(audio_path, transcriptions):
            transcriptions_copy = list(transcriptions)
            fusion_start = time.time()
            filtered_transcriptions, filtration_metadata = TranscriptFilter(mode=filteration_mode).main(transcriptions_copy)
            # Step 1: Build Word Transition Network
            wtn = self.build_word_transition_network(filtered_transcriptions)
            
            # Step 2: Vote on WTN
            result = self.vote_on_wtn(wtn, self.model_weights)
            result["fusion_time"] = time.time() - fusion_start
            
            # Add alignment info for inspection
            result["wtn"] = {
                "num_slots": wtn["num_slots"],
                "num_models": wtn["num_models"],
                "model_indices": wtn["model_indices"]
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
                    norm_hyp = text_processing.StandardArabicTextProcessor.main(
                        hyp, substitute=True)
                    
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