from difflib import SequenceMatcher
from collections import defaultdict
from itertools import islice
from collections import Counter, OrderedDict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import time
import hashlib
import math

import numpy as np
from tqdm import tqdm
from Levenshtein import distance
import kenlm

from stt_benchmarking.utils import text_processing, helpers, metrics
from stt_benchmarking.models.llms import reinforcer
from . import LOGGER

class HybridEnsemble:
    def __init__(self, use_llm, mode="conservative", perfection_rule_length=5, dynamic_match_percentage=0.8):
        self.use_llm = use_llm
        if self.use_llm:
            self._initialize_llm()

        self.perfection_rule_length = perfection_rule_length
        self.dynamic_match_percentage = dynamic_match_percentage
        self.priority_order = self._voting_priority(mode)
        self._input_to_fusion = {}
        self._samples_info = {}
        self._processed_results = []
        self._overall_metrics = None
        LOGGER.info(f"Using the following parameters, perfection_rule_length: {self.perfection_rule_length}, dynamic_match_percentage: {self.dynamic_match_percentage}, priority_order: {self.priority_order}")

    def _voting_priority(self, mode):
        if mode.lower() == "conservative":
            priority_order = ["<KEEP>", "<REPLACE>", "<DELETE>", "<SKIP>", "<INSERT>"]
        elif mode.lower() == "aggressive":
            priority_order = ["<REPLACE>", "<KEEP>", "<INSERT>", "<DELETE>", "<SKIP>"]
        # reference
        elif mode.lower() == "reference":
            priority_order = ["<KEEP>", "<SKIP>", "<REPLACE>", "<INSERT>", "<DELETE>"]
        else:
            raise ValueError("mode should be one of the following: [conservative, aggressive, reference]")
        return priority_order

    def _initialize_llm(self):
        self.REINFORCER = reinforcer.FusionReinforcer()
    
    @staticmethod
    def compute_weights(accuracies):
        """
        Compute model weights based on inverse error scores.
        
        Args:
            errors (list): List of error scores for each model
            
        Returns:
            np.array: Normalized weights (higher weight for lower error)
        """
        accuracies = np.array(accuracies)
        weights = np.maximum(accuracies, 1e-10)
        weights = weights / weights.sum()
        return weights
        
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

        # Collect all unique audio paths
        all_audio_paths = sorted({path for d in samples_dicts for path in d.keys()})
        # Build combined dict
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
    
    @staticmethod
    def find_first_anchors(reference_sentence, compared_sentence):
        ref_words = reference_sentence.split()
        comp_words = compared_sentence.split()
        for i, ref_word in enumerate(ref_words):
            for j, comp_word in enumerate(comp_words):
                if ref_word == comp_word:
                    return (i, j)
        return None
    
    

    def get_reference_from_transcriptions(self, transcriptions):
        from Levenshtein import distance as levenshtein_distance
        
        def build_validation_matrix(valid_transcriptions):
            """Build pairwise validation matrix and identify outliers"""
            n = len(valid_transcriptions)
            if n <= 1:
                return set(range(n)), {}
            
            validation_matrix = {}
            for i in range(n):
                for j in range(i + 1, n):
                    trans_i = valid_transcriptions[i][1]
                    trans_j = valid_transcriptions[j][1]
                    is_valid, _ = validate_anchor_quality_levenshtein(trans_i, trans_j)
                    validation_matrix[(i, j)] = is_valid
            
            # Count agreements per transcription
            agreement_counts = [0] * n
            for i in range(n):
                for j in range(n):
                    if i < j:
                        key = (i, j)
                    elif i > j:
                        key = (j, i)
                    else:
                        continue
                    if validation_matrix.get(key, False):
                        agreement_counts[i] += 1
                        agreement_counts[j] += 1
            
            # Filter: keep transcriptions agreeing with >=50% of others
            min_agreements = max(1, int((n - 1) * 0.5))
            valid_indices = {i for i, count in enumerate(agreement_counts) if count >= min_agreements}
            
            # If everyone filtered out, keep top 2 by agreement count
            if not valid_indices:
                sorted_indices = sorted(range(n), key=lambda i: agreement_counts[i], reverse=True)
                valid_indices = set(sorted_indices[:2])
            
            return valid_indices, validation_matrix
        
        def validate_anchor_quality_levenshtein(reference, transcription):
            """Character-level Levenshtein validation"""
            anchors = self.find_first_anchors(reference, transcription)
            if not anchors:
                return False, None
            
            ref_pos, trans_pos = anchors
            ref_words = reference.split()
            trans_words = transcription.split()
            ref_remaining = ref_words[ref_pos:]
            trans_remaining = trans_words[trans_pos:]
            
            shorter_word_count = min(len(ref_remaining), len(trans_remaining))
            
            ref_char_seq = ' '.join(ref_remaining)
            trans_char_seq = ' '.join(trans_remaining)
            ref_normalized = ' '.join(ref_char_seq.split())
            trans_normalized = ' '.join(trans_char_seq.split())
            
            edit_dist = levenshtein_distance(ref_normalized, trans_normalized)
            max_length = max(len(ref_normalized), len(trans_normalized))
            
            if max_length == 0:
                return True, {"edit_distance": 0, "similarity": 1.0}
            
            similarity = 1.0 - (edit_dist / max_length)
            
            if shorter_word_count <= self.perfection_rule_length:
                is_valid = similarity >= 0.95
                threshold = 0.95
            else:
                adjusted_threshold = max(0.90, 0.90 + (shorter_word_count - self.perfection_rule_length) * 0.01)
                is_valid = similarity >= adjusted_threshold
                threshold = adjusted_threshold
            
            return is_valid, {
                "edit_distance": edit_dist,
                "similarity": similarity,
                "threshold": threshold
            }
        
        def get_longest_reference_filtered(transcriptions):
            """Longest reference with outlier filtering"""
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None
            
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], {
                    'strategy_metric': 'single_transcription',
                    'score': 1.0,
                    'reference_index': valid_transcriptions[0][0]
                }
            
            # Build validation matrix and filter outliers
            valid_indices, validation_matrix = build_validation_matrix(valid_transcriptions)
            
            # From filtered set, pick longest
            filtered = [valid_transcriptions[i] for i in valid_indices]
            longest_original_index, longest_reference = max(filtered, key=lambda x: len(x[1]))
            
            # Calculate success metrics on filtered set
            n = len(filtered)
            successful = sum(1 for i in valid_indices for j in valid_indices 
                            if i < j and validation_matrix.get((i, j), False))
            total_possible = (n * (n - 1)) // 2
            
            success_ratio = successful / total_possible if total_possible > 0 else 1.0
            success = success_ratio >= 0.5
            
            metadata = {
                'strategy_metric': 'filtered_validation_ratio',
                'score': success_ratio,
                'successful_alignments': successful,
                'total_comparisons': total_possible,
                'cluster_size': len(valid_indices),
                'outliers_removed': len(valid_transcriptions) - len(valid_indices),
                'reference_index': longest_original_index
            }
            
            return success, longest_reference, metadata
        
        def get_common_words_reference_filtered(transcriptions):
            """Similarity-based selection WITH outlier filtering"""
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None
            
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], {
                    'strategy_metric': 'single_transcription',
                    'score': 1.0,
                    'reference_index': valid_transcriptions[0][0]
                }
            
            # Filter outliers first
            valid_indices, _ = build_validation_matrix(valid_transcriptions)
            filtered = [(idx, valid_transcriptions[idx][1]) for idx in valid_indices]
            
            # Calculate similarity only within filtered cluster
            best_total_similarity = -1
            best_reference, best_original_index = None, None
            
            for local_i, (_, trans_i) in enumerate(filtered):
                total_similarity = 0.0
                words_i = trans_i.split()
                
                for local_j, (_, trans_j) in enumerate(filtered):
                    if local_i != local_j:
                        words_j = trans_j.split()
                        matcher = SequenceMatcher(None, words_i, words_j)
                        total_similarity += matcher.ratio()
                
                if total_similarity > best_total_similarity:
                    best_total_similarity = total_similarity
                    best_reference = trans_i
                    best_original_index = valid_transcriptions[list(valid_indices)[local_i]][0]
            
            num_comparisons = len(filtered) - 1
            avg_similarity = best_total_similarity / num_comparisons if num_comparisons > 0 else 1.0
            
            success = avg_similarity >= 0.5
            metadata = {
                'strategy_metric': 'filtered_average_similarity',
                'score': avg_similarity,
                'cluster_size': len(valid_indices),
                'outliers_removed': len(valid_transcriptions) - len(valid_indices),
                'reference_index': best_original_index
            }
            
            return success, best_reference, metadata
        
        def get_longest_reference_fallback(transcriptions):
            """Unchanged fallback for edge cases"""
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], valid_transcriptions[0][0]
            
            best_avg_similarity = -1
            best_reference, best_index = None, None
            
            for original_i, transcription_i in valid_transcriptions:
                words_i = transcription_i.split()
                total_similarity = 0.0
                comparisons = 0
                
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        words_j = transcription_j.split()
                        matcher = SequenceMatcher(None, words_i, words_j)
                        total_similarity += matcher.ratio()
                        comparisons += 1
                
                avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
                
                if avg_similarity > best_avg_similarity or \
                (avg_similarity == best_avg_similarity and len(transcription_i) > len(best_reference or "")):
                    best_avg_similarity = avg_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            success = best_avg_similarity >= 0.2
            metadata = {
                'strategy_metric': 'fallback_similarity',
                'score': best_avg_similarity,
                'reference_index': best_index
            }
            return success, best_reference, metadata
        
        # Cascade with filtering
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        success, reference, metadata = get_longest_reference_filtered(transcriptions)
        if success:
            return "longest_filtered", reference, metadata
        
        success, reference, metadata = get_common_words_reference_filtered(transcriptions)
        if success:
            return "common_words_filtered", reference, metadata
        
        success, reference, metadata = get_longest_reference_fallback(transcriptions)
        return "fallback", reference, metadata

    def align_transcriptions_to_reference(self, reference, reference_type, reference_index, transcriptions):
        """
        Simple alignment function that extracts operations and candidate values.
        
        Args:
            reference (str): Reference transcription
            transcription (str): Candidate transcription to align
            
        Returns:
            tuple: (operations, candidate_values)
        """
        # Wont change
        def align_with_longest_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns a list of transcriptions against a single authoritative reference.

            This function uses the provided 'reference' sentence as a structural backbone.
            It iterates through each transcription in the list and compares it to the
            reference using Python's SequenceMatcher. For each position in the
            reference, it determines an operation ('<KEEP>', '<REPLACE>', '<DELETE>')
            and the corresponding token from the transcription.

            The key characteristic of this strategy is that the output structure is
            always dictated by the length of the reference. Insertions in the
            candidate transcriptions are ignored in the final alignment to preserve this
            structure, ensuring all output lists have the same length.

            Args:
                reference (str): The authoritative transcription to align against.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence within the
                    transcriptions list.
                reference_type (str): A string descriptor for the strategy used
                    (e.g., "longest").

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription. Each dictionary contains:
                            - 'model_index' (int): The original index of the model.
                            - 'reference_type' (str): The strategy name.
                            - 'is_reference' (bool): A flag indicating if this was
                            the reference transcription.
                            - 'operations' (list[str]): A list of operations
                            relative to the reference.
                            - 'tokens' (list[str or None]): A list of words (or None)
                            aligned to the reference positions.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # --- PHASE 1: SURVEY AND PAD THE REFERENCE ---
            # Find all unique insertion points from other transcriptions
            insertion_map = {}
            for model_idx, transcription in enumerate(transcriptions):
                if model_idx == reference_index or not transcription: 
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, _, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])

            # Build the new, flexible "padded" reference blueprint
            padded_ref_words = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    # Add placeholders for unique words other models inserted
                    padded_ref_words.extend([None] * len(set(insertion_map[i])))
                padded_ref_words.append(word)
            
            # Handle insertions that occur after the last word of the reference
            if len(ref_words) in insertion_map:
                padded_ref_words.extend([None] * len(set(insertion_map[len(ref_words)])))
                
            padded_ref_length = len(padded_ref_words)

            # --- PHASE 2: ALIGN ALL TRANSCRIPTIONS TO THE PADDED REFERENCE ---
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                # Align the original reference model to the new padded structure
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words[:]
                
                # Align all other models to the new padded structure
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            # 'insert' is now handled implicitly by the padded structure,
                            # so a 'pass' is safe here as a fallback.
                            elif op == 'insert':
                                pass
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata

        def align_with_common_words_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns transcriptions using a flexible, consensus-based reference.

            This strategy is designed for cases where no single transcription is a clear
            structural authority. It works in two main phases:

            1.  **Reference Padding:** It first surveys all other transcriptions to find
                words they contain that are missing from the reference ('insertions').
                It then creates a new "padded" reference by inserting `None`
                placeholders at the appropriate positions to create slots for these
                potential new words.

            2.  **Final Alignment:** It then aligns all transcriptions against this new,
                longer, padded reference. This allows words that would have been
                insertions to be properly mapped to a slot, enabling a vote on whether
                they should be included in the final output.

            The purpose is to create a flexible alignment structure that accommodates
            structural differences between models, rather than strictly enforcing the
            structure of one reference.

            Args:
                reference (str): The consensus-based transcription to use as a starting point.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence.
                reference_type (str): A string descriptor for the strategy.

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription, all aligned to the padded reference length.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # Pad reference by finding all unique insertion points from other transcriptions
            padded_ref_words = ref_words.copy()
            insertion_map = {}  # Maps ref position -> list of words inserted before it
            for model_index, transcription in enumerate(transcriptions):
                if model_index == reference_index or transcription is None:
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        # Words in transcription that don't exist in reference
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])
            
            # Build padded reference with None placeholders for insertions
            final_ref = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    final_ref.extend([None] * len(set(insertion_map[i])))  # Unique insertions only
                final_ref.append(word)
            if len(ref_words) in insertion_map:  # Trailing insertions
                final_ref.extend([None] * len(set(insertion_map[len(ref_words)])))
            
            padded_ref_words = final_ref
            padded_ref_length = len(padded_ref_words)
            
            # Align each transcription to padded reference
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words.copy()
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            elif op == 'insert':
                                pass  # Insertions don't map to reference positions
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata
        
        def align_with_longest_fallback_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Fallback alignment when anchor validation fails.
            
            Flow:
            1. Compute pairwise similarity matrix for all valid transcriptions
            2. Calculate reliability weight for each model (average similarity to others)
            3. Perform standard word-level alignment (same as longest strategy)
            4. Attach weight metadata to each alignment result for downstream use
            
            Difference from longest strategy:
            - Longest: High anchor confidence, reference is structural authority
            - Fallback: Low anchor confidence, reference is "best guess", weights signal reliability
            
            Output: Same structure as longest strategy + 'weight' field per model
            Purpose: Enable weighted voting in consensus step when reference quality is uncertain
            """
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t and t.strip()]
            
            if len(valid_transcriptions) <= 1:
                # Degenerate case: use simple alignment without weights
                return align_with_longest_strategy(reference, transcriptions, reference_index, reference_type)
            
            # Calculate pairwise similarities
            model_weights = {}
            for i, trans_i in valid_transcriptions:
                words_i = trans_i.split()
                total_sim = 0.0
                comparisons = 0
                for j, trans_j in valid_transcriptions:
                    if i != j:
                        words_j = trans_j.split()
                        total_sim += SequenceMatcher(None, words_i, words_j).ratio()
                        comparisons += 1
                model_weights[i] = total_sim / comparisons if comparisons > 0 else 0.0
            
            # Normalize weights to [0, 1]
            max_weight = max(model_weights.values()) if model_weights else 1.0
            if max_weight > 0:
                model_weights = {k: v / max_weight for k, v in model_weights.items()}
            
            # Perform standard alignment
            ref_words = reference.split()
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index),
                    'weight': model_weights.get(model_index, 0.0)  # Reliability score
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * len(ref_words)
                    alignment_result['tokens'] = ref_words.copy()
                else:
                    if not transcription or not transcription.strip():
                        alignment_result['operations'] = ["<DELETE>"] * len(ref_words)
                        alignment_result['tokens'] = [None] * len(ref_words)
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * len(ref_words)
                        candidate_values = [None] * len(ref_words)
                        
                        matcher = SequenceMatcher(None, ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    candidate_values[i] = ref_words[i]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        candidate_values[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = candidate_values
                
                alignment_results.append(alignment_result)
            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': ref_words.copy(), 
                'model_weights': model_weights # This is the key metadata for this strategy
            }
            return alignment_results, metadata
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest_filtered":
            alignment_results, alignment_metadata = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words_filtered":
            alignment_results, alignment_metadata = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "fallback":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results, alignment_metadata
        
    def token_voting_scheme(self, alignment_results):
        """
        Implement a token-only majority voting scheme.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """
        
        def collect_position_tokens(alignment_results, position):
            """Collect all tokens for a specific position, including None."""
            tokens = []
            for result in alignment_results:
                tokens.append(result['tokens'][position])
            return tokens

        def vote_for_token(position_tokens):
            """
            Determine the final token by majority vote, with complex tie-breaking.
            
            Args:
                position_tokens: A list of all token candidates at this position
                                 (e.g., ['wordA', 'wordB', 'wordA', None])
                                 
            Returns:
                (final_token, token_counts_dict)
            """
            def token_similarity_tiebreaker(tied_tokens, all_candidate_tokens):
                """Select token with highest character overlap across all candidates."""
                def stable_hash(s):
                    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
                
                if len(tied_tokens) == 1:
                    return tied_tokens[0]
                
                scores = {}
                for candidate in tied_tokens:
                    if candidate is None: 
                        scores[candidate] = -1 
                        continue
                        
                    candidate_chars = set(candidate)
                    total_overlap = sum(
                        len(candidate_chars & set(other)) 
                        for other in all_candidate_tokens if other is not None and other != candidate
                    )
                    scores[candidate] = total_overlap
                
                max_score = max(scores.values())
                best_tokens = [t for t, s in scores.items() if s == max_score]
                
                return min(best_tokens, key=lambda t: (len(t) if t is not None else float('inf'), stable_hash(t) if t is not None else float('inf')))

            def vote_by_edit_distance(tied_tokens, all_tokens):
                """
                Tie-breaker using Levenshtein distance.
                Selects the token with the minimum total edit distance to all other tokens.
                """
                scores = {}
                for candidate in tied_tokens:
                    if candidate is None:
                        scores[candidate] = float('inf') 
                        continue

                    total_distance = sum(
                        distance(candidate, other) 
                        for other in all_tokens if other and other != candidate
                    )
                    scores[candidate] = total_distance
                
                min_distance = min(scores.values())
                best_tokens = [t for t, d in scores.items() if d == min_distance]
                
                if len(best_tokens) > 1:
                    return token_similarity_tiebreaker(best_tokens, all_tokens)
                return best_tokens[0]

            token_counts = Counter(position_tokens)
            if not token_counts: return None, {} 

            max_count = token_counts.most_common(1)[0][1]
            tied_tokens = [t for t, c in token_counts.items() if c == max_count]
            final_token = None
            if len(tied_tokens) == 1:
                final_token = tied_tokens[0]
            else:
                word_tokens = [t for t in tied_tokens if t is not None]
                if len(word_tokens) == 0:
                    final_token = None
                elif len(word_tokens) == 1:
                    final_token = word_tokens[0]
                else:
                    all_non_none_tokens = [t for t in position_tokens if t is not None]
                    final_token = vote_by_edit_distance(word_tokens, all_non_none_tokens)
            
            return final_token, dict(token_counts)

        def create_voting_detail(position, final_token, token_counts, total_votes):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': token_counts,
                'final_token': final_token,
                'models_voted': total_votes,
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0:
                return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0:
                return 0
                
            for i in range(total_positions):
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            return sum(position_confidences) / total_positions

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        sequence_length = len(alignment_results[0]['tokens'])
        fusion_tokens = []
        voting_details = []
        total_models = len(alignment_results)

        for position in range(sequence_length):
            position_tokens = collect_position_tokens(alignment_results, position)
            final_token, token_counts = vote_for_token(position_tokens)
            
            fusion_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, final_token, token_counts, total_models)
            voting_details.append(voting_detail)
        
        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': total_models,
            'sequence_length': sequence_length,
            'selection_method': 'token_majority_vote',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }
        
    def llm_reinforcer(self, fusion_tokens, candidates_tokens, max_tokens, chunk_size, overlap):
        def postprocess_reinforced_output(response: reinforcer.GeneratedResponse) -> tuple:
            chunks = response.reinforced_results or []
            modifications = sum(1 for chunk in chunks if chunk.get("is_modified"))
            transcript_pieces = []
            for chunk in chunks:
                if chunk.get("is_modified") and chunk.get("sentence"):
                    transcript_pieces.append(chunk["sentence"])
                else:
                    original_sentence = chunk.get("metadata", {}).get("fusion_sentence", "")
                    transcript_pieces.append(original_sentence)

            unwanted_values = [None, "None", "Null", "null", ""]
            filtered_pieces = [piece for piece in transcript_pieces if piece not in unwanted_values]
            llm_transcript = " ".join(filtered_pieces).strip()
            llm_metadata = {
                "chunks": chunks,
                "num_chunks": len(chunks),
                "modifications": modifications,
                "modification_ratio": modifications / len(chunks) if chunks else 0.0,
                "is_chunked": response.is_chunked,
            }
            
            return llm_transcript, llm_metadata

        if not self.use_llm:
            # Return an empty transcript and a default metadata object for consistency
            llm_metadata = {
                "llm_time": 0.0,
                "status": "LLM reinforcement was disabled.",
                "num_chunks": 0, "modifications": 0, "modification_ratio": 0.0
            }
            return "", llm_metadata
        
        llm_start = time.time()
        response = self.REINFORCER.main(
            fusion_tokens=fusion_tokens,
            candidate_tokens=candidates_tokens,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        llm_time = time.time() - llm_start
        llm_transcript, llm_metadata = postprocess_reinforced_output(response)
        llm_metadata["llm_time"] = llm_time
        return llm_transcript, llm_metadata

    def rover_scheme(self, alignment_results):
        """
        Implement the ROVER (Recognizer Output Voting Error Reduction) voting scheme.
        This method first builds a Word Transition Network (WTN) and then finds the
        best-scoring path through it to generate the fused transcript.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """

        def build_word_transition_network(alignment_results):
            """Create a Word Transition Network from aligned hypotheses."""
            if not alignment_results:
                return []
            
            sequence_length = len(alignment_results[0]['tokens'])
            wtn = []
            for i in range(sequence_length):
                position_tokens = [result['tokens'][i] for result in alignment_results]
                wtn.append(Counter(position_tokens))
            return wtn

        def find_best_path(wtn):
            """Find the best path through the WTN using dynamic programming (Viterbi-like)."""
            if not wtn:
                return []

            scores = [{}]
            backpointers = [{}]

            # Initialize first column
            for word, count in wtn[0].items():
                scores[0][word] = count
                backpointers[0][word] = None

            # Iterate through the rest of the network
            for i in range(1, len(wtn)):
                scores.append({})
                backpointers.append({})
                for word, count in wtn[i].items():
                    max_score = -1
                    best_prev_word = None
                    
                    # Find the previous word that maximizes the path score
                    for prev_word, prev_score in scores[i-1].items():
                        current_score = prev_score + count
                        if current_score > max_score:
                            max_score = current_score
                            best_prev_word = prev_word
                    
                    scores[i][word] = max_score
                    backpointers[i][word] = best_prev_word

            # Trace back from the best final word
            best_path = []
            last_word = max(scores[-1], key=scores[-1].get)
            
            for i in range(len(wtn) - 1, -1, -1):
                best_path.insert(0, last_word)
                last_word = backpointers[i][last_word]
                
            return best_path

        def create_voting_detail(position, final_token, wtn_level):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': dict(wtn_level),
                'final_token': final_token,
                'models_voted': sum(wtn_level.values()),
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0: return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0: return 0
                
            for i in range(total_positions):
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            return sum(position_confidences) / total_positions

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        wtn = build_word_transition_network(alignment_results)
        fusion_tokens = find_best_path(wtn)

        voting_details = []
        for i, token in enumerate(fusion_tokens):
            voting_details.append(create_voting_detail(i, token, wtn[i]))

        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'sequence_length': len(fusion_tokens),
            'selection_method': 'rover',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }

    def two_stage_voting_scheme(self, alignment_results):
        """
        Implements a 2-stage token voting scheme to filter noise.
        Stage 1: Votes on 'None' (deletion) vs. 'Word' (any token).
        Stage 2: If 'Word' wins, runs a majority vote with tie-breaking 
                 on the non-None tokens.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """

        # --- Start of Nested Helper Functions ---

        def collect_position_tokens(alignment_results, position):
            """Collect all tokens for a specific position, including None."""
            tokens = []
            for result in alignment_results:
                tokens.append(result['tokens'][position])
            return tokens

        def vote_2_stage(position_tokens):
            """
            Runs the 2-stage (Null-vs-Word) voting logic.
            
            Returns:
                (final_token, token_counts_dict_for_metadata)
            """

            # --- Tie-breaker functions (from your token_voting_scheme) ---
            def token_similarity_tiebreaker(tied_tokens, all_candidate_tokens):
                """Select token with highest character overlap across all candidates."""
                def stable_hash(s):
                    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
                
                if len(tied_tokens) == 1:
                    return tied_tokens[0]
                
                scores = {}
                for candidate in tied_tokens:
                    # This function only receives words, so no None check needed
                    candidate_chars = set(candidate)
                    total_overlap = sum(
                        len(candidate_chars & set(other)) 
                        for other in all_candidate_tokens if other is not None and other != candidate
                    )
                    scores[candidate] = total_overlap
                
                max_score = max(scores.values())
                best_tokens = [t for t, s in scores.items() if s == max_score]
                
                return min(best_tokens, key=lambda t: (len(t), stable_hash(t)))

            def vote_by_edit_distance(tied_tokens, all_tokens):
                """
                Tie-breaker using Levenshtein distance.
                Selects the token with the minimum total edit distance to all other tokens.
                """
                scores = {}
                for candidate in tied_tokens:
                    # This function only receives words, so no None check needed
                    total_distance = sum(
                        distance(candidate, other) 
                        for other in all_tokens if other and other != candidate
                    )
                    scores[candidate] = total_distance
                
                min_distance = min(scores.values())
                best_tokens = [t for t, d in scores.items() if d == min_distance]
                
                if len(best_tokens) > 1:
                    return token_similarity_tiebreaker(best_tokens, all_tokens)
                return best_tokens[0]

            # --- End of Tie-breaker functions ---

            if not position_tokens:
                return None, {}
            
            all_token_counts = Counter(position_tokens)
            total_votes = len(position_tokens)
            null_votes = all_token_counts.get(None, 0)
            word_votes = total_votes - null_votes

            # --- Stage 1: Null vs. Word Vote ---
            if null_votes > word_votes:
                # 'None' wins, return deletion
                return None, dict(all_token_counts)

            # --- Stage 2: Token Vote (Word wins or ties) ---
            word_token_list = [t for t in position_tokens if t is not None]
            
            if not word_token_list:
                # Only 'None' was present, but it didn't win (e.g., 0 votes)
                return None, dict(all_token_counts)
                
            word_token_counts = Counter(word_token_list)
            max_count = word_token_counts.most_common(1)[0][1]
            tied_word_tokens = [t for t, c in word_token_counts.items() if c == max_count]
            
            final_token = None
            if len(tied_word_tokens) == 1:
                final_token = tied_word_tokens[0]
            else:
                # Pass only the tied words to the tie-breaker
                final_token = vote_by_edit_distance(tied_word_tokens, word_token_list)
            
            # Return the original counts for metadata, but the chosen word
            return final_token, dict(all_token_counts)


        def create_voting_detail(position, final_token, token_counts, total_votes):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': token_counts, # Full counts (None + words)
                'final_token': final_token,
                'models_voted': total_votes,
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0: return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0: return 0
                
            for i in range(total_positions):
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            if not position_confidences: return 0
            return sum(position_confidences) / len(position_confidences)

        # --- Main Function Logic ---

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        sequence_length = len(alignment_results[0]['tokens'])
        fusion_tokens = []
        voting_details = []
        total_models = len(alignment_results)

        for position in range(sequence_length):
            position_tokens = collect_position_tokens(alignment_results, position)
            final_token, token_counts = vote_2_stage(position_tokens)
            
            fusion_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, final_token, token_counts, total_models)
            voting_details.append(voting_detail)
        
        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': total_models,
            'sequence_length': sequence_length,
            'selection_method': '2_stage_token_vote',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }

    def hybrid_rover_scheme(self, alignment_results):
        """
        Implements a Hybrid ROVER scheme.
        It uses a 2-stage (Null-vs-Word) vote to prune the 
        Word Transition Network (WTN) at each position *before*
        running the Viterbi path-finding algorithm.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """

        def build_hybrid_wtn(alignment_results):
            """
            Create a WTN using 2-stage (Null-vs-Word) voting
            to prune nodes at each position.
            """
            if not alignment_results:
                return []
            
            sequence_length = len(alignment_results[0]['tokens'])
            wtn = []
            for i in range(sequence_length):
                position_tokens = [result['tokens'][i] for result in alignment_results]
                all_token_counts = Counter(position_tokens)
                total_votes = len(position_tokens)
                if total_votes == 0:
                    wtn.append(Counter())
                    continue

                null_votes = all_token_counts.get(None, 0)
                word_votes = total_votes - null_votes
                
                final_counts = Counter()
                
                # --- Stage 1: Null vs. Word Vote ---
                if null_votes > word_votes:
                    # 'None' wins, only add None to the network node
                    final_counts[None] = null_votes
                else:
                    # 'Word' wins or ties, add all words (and *only* words)
                    for token, count in all_token_counts.items():
                        if token is not None:
                            final_counts[token] = count
                    
                    # Fallback: if 'Word' won but list was empty
                    if not final_counts and null_votes > 0:
                         final_counts[None] = null_votes
                         
                wtn.append(final_counts)
            return wtn

        def find_best_path(wtn):
            """Find the best path through the WTN using dynamic programming (Viterbi-like)."""
            if not wtn:
                return []

            scores = [{}]
            backpointers = [{}]

            # Initialize first column
            if not wtn[0]: # Handle empty first node
                 scores[0][None] = 0 # Start with a 'None' path
                 backpointers[0][None] = None
            else:
                for word, count in wtn[0].items():
                    scores[0][word] = count
                    backpointers[0][word] = None

            # Iterate through the rest of the network
            for i in range(1, len(wtn)):
                scores.append({})
                backpointers.append({})
                
                if not wtn[i]: # If this WTN node is empty, skip (e.g., word won but no words)
                    scores[i] = scores[i-1] # Carry over previous scores
                    backpointers[i] = {word: word for word in scores[i-1]}
                    continue
                    
                for word, count in wtn[i].items():
                    max_score = -1
                    best_prev_word = None
                    
                    if not scores[i-1]: # If previous state was empty
                        max_score = count
                        best_prev_word = None
                    else:
                        for prev_word, prev_score in scores[i-1].items():
                            current_score = prev_score + count
                            if current_score > max_score:
                                max_score = current_score
                                best_prev_word = prev_word
                    
                    scores[i][word] = max_score
                    backpointers[i][word] = best_prev_word

            # Trace back from the best final word
            best_path = []
            if not scores[-1]: # Handle if all paths died
                return [None] * len(wtn) # Return a path of Nones
                
            last_word = max(scores[-1], key=scores[-1].get)
            
            for i in range(len(wtn) - 1, -1, -1):
                best_path.insert(0, last_word)
                last_word = backpointers[i].get(last_word)
                if last_word is None and i > 0:
                    # Path broke, find best word in previous-1
                    prev_last_word = max(scores[i-1], key=scores[i-1].get)
                    last_word = backpointers[i-1].get(prev_last_word)
                
            return best_path

        def create_voting_detail(position, final_token, wtn_level):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': dict(wtn_level),
                'final_token': final_token,
                'models_voted': sum(wtn_level.values()),
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0: return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0: return 0
                
            for i in range(total_positions):
                if i >= len(alignment_results[0]['tokens']): break # Safety check
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            if not position_confidences: return 0
            return sum(position_confidences) / len(position_confidences)

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        wtn = build_hybrid_wtn(alignment_results)
        fusion_tokens = find_best_path(wtn)

        voting_details = []
        for i, token in enumerate(fusion_tokens):
            voting_details.append(create_voting_detail(i, token, wtn[i] if i < len(wtn) else Counter()))

        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'sequence_length': len(fusion_tokens),
            'selection_method': 'hybrid_rover_2_stage',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }

    def get_model_weights(self, alignment_results, exponent):
        """
        Calculate a global quality score (weight) for each model based on
        its 'operations' list against the reference.
        """
        weights = {}
        for result in alignment_results:
            model_index = result['model_index']
            ops = result.get('operations') # Use .get for safety
            
            if not ops:
                weights[model_index] = 0.1 # Assign a low default weight
                continue
                
            keep_count = ops.count('<KEEP>')
            total_ops = len(ops)
            
            if total_ops == 0:
                weights[model_index] = 0.1
                continue

            keep_rate = keep_count / total_ops 
            weight = keep_rate ** exponent
            
            if result.get('is_reference', False):
                weight = max(weight, 1.0) # Reference always gets at least 1.0
                
            weights[model_index] = weight
        return weights

    def confidence_weighted_rover_scheme(self, alignment_results, weight_exponent=2):
        """
        Implements ROVER using both:
        1. Global model quality (from metadata)
        2. Local token confidence (from ASR)
        
        Args:
            alignment_results: List of alignment dicts, NOW MUST INCLUDE 'confidences' list.
            weight_exponent: Power to raise the 'keep_rate' to.
            
        Returns:
            dict: Final voting result with tokens and metadata
        """

        def build_word_transition_network(alignment_results, model_weights):
            """
            Create a WTN where each token vote is multiplied
            by its model's global_weight * local_confidence.
            """
            if not alignment_results:
                return []
            
            sequence_length = len(alignment_results[0]['tokens'])
            
            # Create default confidence lists for models that don't provide them
            default_confs = [1.0] * sequence_length 

            wtn = []
            for i in range(sequence_length):
                position_counts = Counter()
                for result in alignment_results:
                    token = result['tokens'][i]
                    
                    # Get global weight
                    global_weight = model_weights[result['model_index']]
                    
                    # Get local confidence
                    confidences = result.get('confidences', default_confs)
                    local_confidence = confidences[i] if i < len(confidences) else 1.0

                    # Combine weights
                    final_vote_power = global_weight * local_confidence
                    
                    position_counts[token] += final_vote_power 
                    
                wtn.append(position_counts)
            return wtn

        def find_best_path(wtn):
            """Find the best path through the WTN using dynamic programming (Viterbi-like)."""
            if not wtn:
                return []

            scores = [{}]
            backpointers = [{}]

            # Initialize first column
            if not wtn[0]:
                 scores[0][None] = 0
                 backpointers[0][None] = None
            else:
                for word, count in wtn[0].items():
                    scores[0][word] = count
                    backpointers[0][word] = None

            # Iterate through the rest of the network
            for i in range(1, len(wtn)):
                scores.append({})
                backpointers.append({})
                
                if not wtn[i]:
                    scores[i] = scores[i-1]
                    backpointers[i] = {word: word for word in scores[i-1]}
                    continue
                    
                for word, count in wtn[i].items():
                    max_score = -float('inf') # Use -inf for weighted scores
                    best_prev_word = None
                    
                    if not scores[i-1]:
                        max_score = count
                        best_prev_word = None
                    else:
                        for prev_word, prev_score in scores[i-1].items():
                            current_score = prev_score + count
                            if current_score > max_score:
                                max_score = current_score
                                best_prev_word = prev_word
                    
                    scores[i][word] = max_score
                    backpointers[i][word] = best_prev_word

            # Trace back from the best final word
            best_path = []
            if not scores[-1]:
                return [None] * len(wtn)
                
            last_word = max(scores[-1], key=scores[-1].get)
            
            for i in range(len(wtn) - 1, -1, -1):
                best_path.insert(0, last_word)
                last_word = backpointers[i].get(last_word)
                if last_word is None and i > 0:
                    # Handle path break
                    prev_last_word = max(scores[i-1], key=scores[i-1].get)
                    last_word = backpointers[i-1].get(prev_last_word)
                
            return best_path

        def create_voting_detail(position, final_token, wtn_level):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': dict(wtn_level),
                'final_token': final_token,
                'models_voted_weighted': sum(wtn_level.values()),
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0: return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0: return 0
                
            for i in range(total_positions):
                if i >= len(alignment_results[0]['tokens']): break 
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            if not position_confidences: return 0
            return sum(position_confidences) / len(position_confidences)

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
        
        # 1. Calculate global weights
        model_weights = self.get_model_weights(alignment_results, weight_exponent)
        
        # 2. Build WTN using global weights and local confidences
        wtn = build_word_transition_network(alignment_results, model_weights)
        
        # 3. Find best path
        fusion_tokens = find_best_path(wtn)

        voting_details = []
        for i, token in enumerate(fusion_tokens):
            voting_details.append(create_voting_detail(i, token, wtn[i] if i < len(wtn) else Counter()))

        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'sequence_length': len(fusion_tokens),
            'selection_method': 'confidence_weighted_rover',
            'weight_exponent': weight_exponent,
            'model_weights': model_weights,
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }
    
    def main(self, **kwargs):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            voting_start = time.time()
            reference_type, reference, reference_metadata = self.get_reference_from_transcriptions(transcriptions)
            alignment_results, alignment_metadata = self.align_transcriptions_to_reference(reference=reference, reference_type=reference_type, 
                                                                                           reference_index=reference_metadata["reference_index"], 
                                                                                           transcriptions=transcriptions)
            voting_output = self.rover_scheme(alignment_results)
            llm_transcript, llm_metadata = self.llm_reinforcer(
                fusion_tokens=voting_output["fusion_tokens"],
                candidates_tokens=voting_output["candidates_tokens"],
                max_tokens=kwargs.get("max_tokens", 25),
                chunk_size=kwargs.get("chunk_size", 15),
                overlap=kwargs.get("overlap", 3),
            )
            data = {
                # --- Primary Outputs ---
                "fusion_transcript": voting_output["fusion_transcript"],
                "llm_transcript": llm_transcript,
                
                # --- Diagnostic/Internal Data ---
                "fusion_tokens": voting_output["fusion_tokens"],
                "candidates_tokens": voting_output["candidates_tokens"],
                "voting_time": time.time() - voting_start,
                
                # --- Aggregated Metadata ---
                "metadata": {
                    "reference_selection": reference_metadata,
                    "alignment": alignment_metadata,
                    "voting": voting_output["voting_metadata"],
                    "llm_reinforcement": llm_metadata  # <-- NEW metadata section
                }
            }
            
            return {audio_path: data}

        def process_item(item):
            key, value = item
            transcriptions_lists = [t for t in value]
            voting_result = fuse_sample_transcriptions(key, transcriptions_lists)
            return voting_result
        
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item for item in self.input_to_fusion.items()}
            samples_info = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing Inputs..."):
                result = future.result()  # this is {audio_path: {...}}
                samples_info.update(result)

        self._samples_info = dict(sorted(samples_info.items()))

    def eval(self, audios_chunk):
        def _eval_common(data):
            return data["fusion_transcript"]

        def _eval_llm(data):
            return data["llm_transcript"]

        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_audio_paths = list(self._samples_info.keys())
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    ref = refs_lookup.get(audio_path)
                    hyp = _eval_llm(self._samples_info[audio_path]) if self.use_llm else _eval_common(self._samples_info[audio_path])
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(hyp, substitute=True)
                    self._samples_info[audio_path]["normalized_transcription"] = ref
                    if ref is not None:
                        norm_hyp = self._samples_info[audio_path]["normalized_prediction"]
                        sample_metrics = metrics.BasicSTTMetrics.evaluate(refs=ref, hyps=norm_hyp)
                        self._samples_info[audio_path]["metrics"] = sample_metrics
                    else:
                        self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()

                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        # Compute overall metrics
        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        """
        Reset the EnsembleInferenceRefactored instance.
        Clears input_to_fusion and fusion_results, but keeps the model loaded.
        """
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
        LOGGER.info("Ensemble instance has been reset.")

    @property
    def input_to_fusion(self):
        """
        Get the input to fusion dictionary.
        """
        if not self._input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        return self._input_to_fusion
    
    @input_to_fusion.setter
    def input_to_fusion(self, value):
        """
        Set the input to fusion dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Input to fusion must be a dictionary.")
        self._input_to_fusion = value

    @property
    def samples_info(self):
        """
        Get the fusion results.
        """
        if not self._samples_info:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics
    

class LLMEnsemble:
    def __init__(self, perfection_rule_length=5, dynamic_match_percentage=0.8):
        self._initialize_llm()

        self.perfection_rule_length = perfection_rule_length
        self.dynamic_match_percentage = dynamic_match_percentage
        self._input_to_fusion = {}
        self._samples_info = {}
        self._processed_results = []
        self._overall_metrics = None
        LOGGER.info(f"Using the following parameters, perfection_rule_length: {self.perfection_rule_length}, dynamic_match_percentage: {self.dynamic_match_percentage}")

    def _initialize_llm(self):
        self.REINFORCER = reinforcer.TokenReinforcer()
        
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

        # Collect all unique audio paths
        all_audio_paths = sorted({path for d in samples_dicts for path in d.keys()})
        # Build combined dict
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
    
    @staticmethod
    def find_first_anchors(reference_sentence, compared_sentence):
        ref_words = reference_sentence.split()
        comp_words = compared_sentence.split()
        for i, ref_word in enumerate(ref_words):
            for j, comp_word in enumerate(comp_words):
                if ref_word == comp_word:
                    return (i, j)
        return None
    
    def validate_anchor_quality(self, reference, transcription):
        """
        Validate that anchors represent meaningful common structure using SequenceMatcher
        
        Returns:
            tuple: (is_valid, anchor_info)
        """
        anchors = self.find_first_anchors(reference, transcription)
        if not anchors: return False, None
            
        ref_pos, trans_pos = anchors
        ref_words = reference.split()
        trans_words = transcription.split()
        ref_remaining = ref_words[ref_pos:]
        trans_remaining = trans_words[trans_pos:]
        # Use SequenceMatcher on the remaining sequences for better validation
        matcher = SequenceMatcher(None, ref_remaining, trans_remaining)
        # Get the longest matching block starting from position 0 (right after anchor)
        longest_match = matcher.find_longest_match(0, len(ref_remaining), 0, len(trans_remaining))
        additional_matches = longest_match.size
        shorter_sequence_length = min(len(ref_remaining), len(trans_remaining))
        is_valid = False
        min_required = 0
        if shorter_sequence_length <= self.perfection_rule_length:
            # Check if the number of matching words is exactly equal to the length of the shorter sequence.
            min_required = shorter_sequence_length
            if additional_matches == shorter_sequence_length:
                is_valid = True
        else:
            min_required = int(shorter_sequence_length * self.dynamic_match_percentage)
            # Check if we found at least our minimum required number of matches.
            if additional_matches >= min_required:
                is_valid = True

        return is_valid, {
            "positions": anchors, 
            "additional_matches": additional_matches,
            "min_required": min_required
        }

    def get_reference_from_transcriptions(self, transcriptions):
        def get_longest_reference(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            successful_alignments = 0
            total_comparisons = len(valid_transcriptions) - 1
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, _ = self.validate_anchor_quality(longest_reference, transcription)
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            metadata = {
                'strategy_metric': 'validation_success_ratio',
                'score': success_ratio,
                'successful_alignments': successful_alignments,
                'total_comparisons': total_comparisons,
                'reference_index': longest_original_index
            }
            return success, longest_reference, longest_original_index, metadata
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                words1, words2 = sentence1.split(), sentence2.split()
                matcher = SequenceMatcher(None, words1, words2)
                return matcher.ratio()
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None

            best_total_similarity = -1
            best_reference, best_index = None, None
            for original_i, transcription_i in valid_transcriptions:
                total_similarity = 0.0
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        similarity = calculate_similarity(transcription_i, transcription_j)
                        total_similarity += similarity
                
                if total_similarity > best_total_similarity:
                    best_total_similarity = total_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            if best_reference and best_total_similarity >= 0:
                num_comparisons = len(valid_transcriptions) - 1 if len(valid_transcriptions) > 1 else 1
                avg_similarity = best_total_similarity / num_comparisons if num_comparisons > 0 else 0.0
                metadata = {
                    'strategy_metric': 'average_similarity_to_best',
                    'score': avg_similarity,
                    'total_similarity_score': best_total_similarity,
                    'reference_index': best_index
                }
                return True, best_reference, best_index, metadata
            
            return False, None, None, None

        def get_longest_reference_fallback(transcriptions):
            """
            Fallback when anchor validation fails but we still need a reference.
            Uses average pairwise similarity to find most representative transcription.
            """
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            # Single transcription case
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], valid_transcriptions[0][0]
            
            # Calculate average word-level edit distance for each transcription
            best_avg_similarity = -1
            best_reference = None
            best_index = None
            
            for original_i, transcription_i in valid_transcriptions:
                words_i = transcription_i.split()
                total_similarity = 0.0
                comparisons = 0
                
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        words_j = transcription_j.split()
                        # Use SequenceMatcher for word-level similarity
                        matcher = SequenceMatcher(None, words_i, words_j)
                        total_similarity += matcher.ratio()
                        comparisons += 1
                
                avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
                
                # Tiebreaker: prefer longer transcription when similarity is equal
                if avg_similarity > best_avg_similarity or \
                (avg_similarity == best_avg_similarity and len(transcription_i) > len(best_reference or "")):
                    best_avg_similarity = avg_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            # Success if we found reasonable consensus (>30% similarity)
            success = best_avg_similarity >= 0.2
            metadata = {
                'strategy_metric': 'average_pairwise_similarity',
                'score': best_avg_similarity,
                'reference_index': best_index
            }
            return success, best_reference, best_index, metadata
        
        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, longest_index, reference_metadata = get_longest_reference(transcriptions)
        if longest_success: return "longest", longest_reference, longest_index, reference_metadata

        common_words_success, common_words_reference, common_words_index, reference_metadata = get_common_words_reference(transcriptions)
        if common_words_success: return "common_words", common_words_reference, common_words_index, reference_metadata

        fallback_success, fallback_reference, fallback_index, reference_metadata = get_longest_reference_fallback(transcriptions)
        if fallback_success: return "longest_fallback", fallback_reference, fallback_index, reference_metadata
        else: return "failed", fallback_reference, fallback_index , reference_metadata

    def align_transcriptions_to_reference(self, reference, reference_type, reference_index, transcriptions):
        """
        Simple alignment function that extracts operations and candidate values.
        
        Args:
            reference (str): Reference transcription
            transcription (str): Candidate transcription to align
            
        Returns:
            tuple: (operations, candidate_values)
        """
        # Wont change
        def align_with_longest_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns a list of transcriptions against a single authoritative reference.

            This function uses the provided 'reference' sentence as a structural backbone.
            It iterates through each transcription in the list and compares it to the
            reference using Python's SequenceMatcher. For each position in the
            reference, it determines an operation ('<KEEP>', '<REPLACE>', '<DELETE>')
            and the corresponding token from the transcription.

            The key characteristic of this strategy is that the output structure is
            always dictated by the length of the reference. Insertions in the
            candidate transcriptions are ignored in the final alignment to preserve this
            structure, ensuring all output lists have the same length.

            Args:
                reference (str): The authoritative transcription to align against.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence within the
                    transcriptions list.
                reference_type (str): A string descriptor for the strategy used
                    (e.g., "longest").

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription. Each dictionary contains:
                            - 'model_index' (int): The original index of the model.
                            - 'reference_type' (str): The strategy name.
                            - 'is_reference' (bool): A flag indicating if this was
                            the reference transcription.
                            - 'operations' (list[str]): A list of operations
                            relative to the reference.
                            - 'tokens' (list[str or None]): A list of words (or None)
                            aligned to the reference positions.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # --- PHASE 1: SURVEY AND PAD THE REFERENCE ---
            # Find all unique insertion points from other transcriptions
            insertion_map = {}
            for model_idx, transcription in enumerate(transcriptions):
                if model_idx == reference_index or not transcription: 
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, _, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])

            # Build the new, flexible "padded" reference blueprint
            padded_ref_words = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    # Add placeholders for unique words other models inserted
                    padded_ref_words.extend([None] * len(set(insertion_map[i])))
                padded_ref_words.append(word)
            
            # Handle insertions that occur after the last word of the reference
            if len(ref_words) in insertion_map:
                padded_ref_words.extend([None] * len(set(insertion_map[len(ref_words)])))
                
            padded_ref_length = len(padded_ref_words)

            # --- PHASE 2: ALIGN ALL TRANSCRIPTIONS TO THE PADDED REFERENCE ---
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                # Align the original reference model to the new padded structure
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words[:]
                
                # Align all other models to the new padded structure
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            # 'insert' is now handled implicitly by the padded structure,
                            # so a 'pass' is safe here as a fallback.
                            elif op == 'insert':
                                pass
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata

        def align_with_common_words_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns transcriptions using a flexible, consensus-based reference.

            This strategy is designed for cases where no single transcription is a clear
            structural authority. It works in two main phases:

            1.  **Reference Padding:** It first surveys all other transcriptions to find
                words they contain that are missing from the reference ('insertions').
                It then creates a new "padded" reference by inserting `None`
                placeholders at the appropriate positions to create slots for these
                potential new words.

            2.  **Final Alignment:** It then aligns all transcriptions against this new,
                longer, padded reference. This allows words that would have been
                insertions to be properly mapped to a slot, enabling a vote on whether
                they should be included in the final output.

            The purpose is to create a flexible alignment structure that accommodates
            structural differences between models, rather than strictly enforcing the
            structure of one reference.

            Args:
                reference (str): The consensus-based transcription to use as a starting point.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence.
                reference_type (str): A string descriptor for the strategy.

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription, all aligned to the padded reference length.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # Pad reference by finding all unique insertion points from other transcriptions
            padded_ref_words = ref_words.copy()
            insertion_map = {}  # Maps ref position -> list of words inserted before it
            for model_index, transcription in enumerate(transcriptions):
                if model_index == reference_index or transcription is None:
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        # Words in transcription that don't exist in reference
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])
            
            # Build padded reference with None placeholders for insertions
            final_ref = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    final_ref.extend([None] * len(set(insertion_map[i])))  # Unique insertions only
                final_ref.append(word)
            if len(ref_words) in insertion_map:  # Trailing insertions
                final_ref.extend([None] * len(set(insertion_map[len(ref_words)])))
            
            padded_ref_words = final_ref
            padded_ref_length = len(padded_ref_words)
            
            # Align each transcription to padded reference
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words.copy()
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            elif op == 'insert':
                                pass  # Insertions don't map to reference positions
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata
        
        def align_with_longest_fallback_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Fallback alignment when anchor validation fails.
            
            Flow:
            1. Compute pairwise similarity matrix for all valid transcriptions
            2. Calculate reliability weight for each model (average similarity to others)
            3. Perform standard word-level alignment (same as longest strategy)
            4. Attach weight metadata to each alignment result for downstream use
            
            Difference from longest strategy:
            - Longest: High anchor confidence, reference is structural authority
            - Fallback: Low anchor confidence, reference is "best guess", weights signal reliability
            
            Output: Same structure as longest strategy + 'weight' field per model
            Purpose: Enable weighted voting in consensus step when reference quality is uncertain
            """
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t and t.strip()]
            
            if len(valid_transcriptions) <= 1:
                # Degenerate case: use simple alignment without weights
                return align_with_longest_strategy(reference, transcriptions, reference_index, reference_type)
            
            # Calculate pairwise similarities
            model_weights = {}
            for i, trans_i in valid_transcriptions:
                words_i = trans_i.split()
                total_sim = 0.0
                comparisons = 0
                for j, trans_j in valid_transcriptions:
                    if i != j:
                        words_j = trans_j.split()
                        total_sim += SequenceMatcher(None, words_i, words_j).ratio()
                        comparisons += 1
                model_weights[i] = total_sim / comparisons if comparisons > 0 else 0.0
            
            # Normalize weights to [0, 1]
            max_weight = max(model_weights.values()) if model_weights else 1.0
            if max_weight > 0:
                model_weights = {k: v / max_weight for k, v in model_weights.items()}
            
            # Perform standard alignment
            ref_words = reference.split()
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index),
                    'weight': model_weights.get(model_index, 0.0)  # Reliability score
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * len(ref_words)
                    alignment_result['tokens'] = ref_words.copy()
                else:
                    if not transcription or not transcription.strip():
                        alignment_result['operations'] = ["<DELETE>"] * len(ref_words)
                        alignment_result['tokens'] = [None] * len(ref_words)
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * len(ref_words)
                        candidate_values = [None] * len(ref_words)
                        
                        matcher = SequenceMatcher(None, ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    candidate_values[i] = ref_words[i]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        candidate_values[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = candidate_values
                
                alignment_results.append(alignment_result)
            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': ref_words.copy(), 
                'model_weights': model_weights # This is the key metadata for this strategy
            }
            return alignment_results, metadata
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results, alignment_metadata = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words":
            alignment_results, alignment_metadata = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "longest_fallback":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "failed":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results, alignment_metadata
    
    def llm_voting_scheme(self, alignment_results):
        """
        Use LLM-only voting for token selection.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """
        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)
        
        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "",
                "fusion_tokens": [],
                "candidates_tokens": [],
                "voting_metadata": {"status": "No alignment results"}
            }
        
        result = self.REINFORCER.main(alignment_results)
        fusion_tokens = result["selected_tokens"]
        selection_metadata = result["selection_metadata"]
        fusion_transcript = construct_final_transcription(fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'total_models': len(alignment_results),
            'operations_length': len(fusion_tokens),
            'selection_method': 'llm_only',
            'selection_details': selection_metadata
        }
        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }

    def main(self, **kwargs):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            voting_start = time.time()
            reference_type, reference, reference_index, reference_metadata = self.get_reference_from_transcriptions(transcriptions)
            alignment_results, alignment_metadata = self.align_transcriptions_to_reference(reference=reference, reference_type=reference_type, 
                                                                                           reference_index=reference_index, transcriptions=transcriptions)
            voting_output = self.llm_voting_scheme(alignment_results)
            data = {
                # --- Primary Outputs ---
                "fusion_transcript": voting_output["fusion_transcript"],
                
                # --- Diagnostic/Internal Data ---
                "fusion_tokens": voting_output["fusion_tokens"],
                "candidates_tokens": voting_output["candidates_tokens"],
                "voting_time": time.time() - voting_start,
                
                # --- Aggregated Metadata ---
                "metadata": {
                    "reference_selection": reference_metadata,
                    "alignment": alignment_metadata,
                    "voting": voting_output["voting_metadata"],
                }
            }
            
            return {audio_path: data}

        def process_item(item):
            key, value = item
            transcriptions_lists = [t for t in value]
            voting_result = fuse_sample_transcriptions(key, transcriptions_lists)
            return voting_result
        
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item for item in self.input_to_fusion.items()}
            samples_info = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing Inputs..."):
                result = future.result()  # this is {audio_path: {...}}
                samples_info.update(result)

        self._samples_info = dict(sorted(samples_info.items()))

    def eval(self, audios_chunk):
        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_audio_paths = list(self._samples_info.keys())
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    ref = refs_lookup.get(audio_path)
                    hyp = self._samples_info[audio_path]["fusion_transcript"]
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(hyp, substitute=True)
                    self._samples_info[audio_path]["normalized_transcription"] = ref
                    if ref is not None:
                        norm_hyp = self._samples_info[audio_path]["normalized_prediction"]
                        sample_metrics = metrics.BasicSTTMetrics.evaluate(refs=ref, hyps=norm_hyp)
                        self._samples_info[audio_path]["metrics"] = sample_metrics
                    else:
                        self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()

                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        # Compute overall metrics
        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        """
        Reset the EnsembleInferenceRefactored instance.
        Clears input_to_fusion and fusion_results, but keeps the model loaded.
        """
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
        LOGGER.info("Ensemble instance has been reset.")

    @property
    def input_to_fusion(self):
        """
        Get the input to fusion dictionary.
        """
        if not self._input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        return self._input_to_fusion
    
    @input_to_fusion.setter
    def input_to_fusion(self, value):
        """
        Set the input to fusion dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Input to fusion must be a dictionary.")
        self._input_to_fusion = value

    @property
    def samples_info(self):
        """
        Get the fusion results.
        """
        if not self._samples_info:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics