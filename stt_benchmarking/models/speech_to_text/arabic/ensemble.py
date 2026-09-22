from difflib import SequenceMatcher
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import time
import hashlib
import difflib
import os
import math

import numpy as np
from tqdm import tqdm
from Levenshtein import distance
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


from stt_benchmarking.utils import text_processing, helpers, metrics
from stt_benchmarking.models.llms import reinforcer
from . import LOGGER

def _sym_ratio(w1,w2):
    """Order-free Ratcliff/Obershelp ratio: the pair is put in canonical order first."""
    a,b=(w1,w2) if w1<=w2 else (w2,w1)
    return SequenceMatcher(None,a,b,autojunk=False).ratio()

def _stable_key(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(),16)

class TranscriptFilter:
    def __init__(self, mode="mean"):
        self.mode = mode
        self.__verify_mode()
    
    def __verify_mode(self):
        if self.mode not in ["mean", "iqr", "mad"]:
            raise ValueError("mode should be mean, iqr, or mad.")
        
    def _calculate_similarity_matrix(self, transcriptions: List[str]) -> pd.DataFrame:
        """Creates an N*N pairwise similarity matrix for all transcriptions."""
        def _word_sequence_ratio(words1: List[str], words2: List[str]) -> float:
            """Calculates the similarity ratio (0-100) based on matching *words*."""
            if not words1 and not words2: return 100.0
            if not words1 or not words2: return 0.0
            return _sym_ratio(words1, words2) * 100.0
        
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
        
        if num_transcriptions < 4:
            return transcriptions, {
                "status": "Skipped", 
                "reason": "Need ≥4 transcripts",
                "kept_indices": list(range(num_transcriptions)), 
                "filtered_indices": []
            }
        
        # Build similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(transcriptions).values
        agreement_scores = []
        for i in range(num_transcriptions):
            others_similarity = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            agreement_scores.append(math.fsum(others_similarity) / len(others_similarity))
        
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
        elif self.mode == "iqr":
            q1, q3 = np.percentile(gaps, [25, 75])
            iqr = q3 - q1
            upper_fence = q3 + (1.5 * iqr)
            significant_gaps = np.where(gaps > upper_fence)[0]
        else:
            median = np.median(gaps)
            mad = np.median(np.abs(gaps - median))
            scale = 1.4826 * mad
            significant_gaps = np.where(gaps > median + 3.0 * scale)[0]

        if len(significant_gaps) == 0:
            return transcriptions, {
                "status": "Skipped",
                "reason": "No significant gap detected; distribution treated as unimodal",
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

class ReferenceSelection:
    def __init__(self, lengthmode="mean", matchmode="mean"):
        self.lengthmode = lengthmode
        self.matchmode = matchmode
        self.__verify_mode()
    
    def __verify_mode(self):
        if self.matchmode and self.lengthmode not in ["mean", "iqr"]:
            raise ValueError("mode should be either mean or iqr.")
        
    def main(self, transcriptions):
        def get_longest_reference(transcriptions):
            def validate_anchor_quality(reference, transcription, length_stats, match_stats):
                """
                Validate that anchors represent meaningful common structure using SequenceMatcher
                
                Returns:
                    tuple: (is_valid, anchor_info)
                """
                def find_first_anchors(reference_sentence, compared_sentence):
                    ref_words = reference_sentence.split()
                    comp_words = compared_sentence.split()
                    for i, ref_word in enumerate(ref_words):
                        for j, comp_word in enumerate(comp_words):
                            if ref_word == comp_word:
                                return (i, j)
                    return None
                
                if not reference or not transcription: return False, None
                anchors = find_first_anchors(reference, transcription)
                if not anchors: return False, None
                    
                ref_pos, trans_pos = anchors
                ref_words = reference.split()
                trans_words = transcription.split()
                ref_remaining = ref_words[ref_pos:]
                trans_remaining = trans_words[trans_pos:]
                matcher = SequenceMatcher(None, ref_remaining, trans_remaining, autojunk=False)
                longest_match = matcher.find_longest_match(0, len(ref_remaining), 0, len(trans_remaining))
                additional_matches = longest_match.size
                shorter_sequence_length = min(len(ref_remaining), len(trans_remaining))
                min_required = 0
                if shorter_sequence_length == 0: return False, None
                if length_stats is None or match_stats is None:
                    required_ratio = 0.75
                    category = "unknown"
                else:
                    lengthmode = length_stats['mode']
                    matchmode = match_stats['mode']
                    if lengthmode == "iqr":
                        q1_len = length_stats['q1']
                        q3_len = length_stats['q3']
                        if shorter_sequence_length < q1_len:
                            category = "short"
                            required_ratio = match_stats['q3'] if matchmode == "iqr" else match_stats['upper']
                        elif shorter_sequence_length > q3_len:
                            category = "long"
                            required_ratio = match_stats['q1'] if matchmode == "iqr" else match_stats['lower']
                        else:
                            category = "mid"
                            required_ratio = match_stats['median'] if matchmode == "iqr" else match_stats['mean']
                                        
                    else:
                        lower_len = length_stats['lower']
                        upper_len = length_stats['upper']
                        if shorter_sequence_length < lower_len:
                            category = "short"
                            required_ratio = match_stats['upper'] if matchmode == "mean" else match_stats['q3']
                        elif shorter_sequence_length > upper_len:
                            category = "long"
                            required_ratio = match_stats['lower'] if matchmode == "mean" else match_stats['q1']
                        else:
                            category = "mid"
                            required_ratio = match_stats['mean'] if matchmode == "mean" else match_stats['median']

                min_required = int(np.ceil(shorter_sequence_length * required_ratio))
                is_valid = additional_matches >= min_required
                return is_valid, {
                    "positions": anchors,
                    "additional_matches": additional_matches,
                    "min_required": min_required,
                    "required_ratio": required_ratio,
                    "shorter_sequence_length": shorter_sequence_length,
                    "length_category": category
                }
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            all_lengths = [len(t.split()) for _, t in valid_transcriptions]
            if len(all_lengths) < 2:
                length_stats = None
            else:
                if self.lengthmode == "iqr":
                    median_len = np.median(all_lengths)
                    q1 = np.percentile(all_lengths, 25)
                    q3 = np.percentile(all_lengths, 75)
                    iqr = q3 - q1
                    length_stats = {
                        'mode': 'iqr',
                        'median': median_len,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'min': min(all_lengths),
                        'max': max(all_lengths)
                    }
                else:
                    mean_len = np.mean(all_lengths)
                    std_len = np.std(all_lengths)
                    length_stats = {
                        'mode': 'mean',
                        'mean': mean_len,
                        'std': std_len,
                        'lower': mean_len - std_len,
                        'upper': mean_len + std_len,
                        'min': min(all_lengths),
                        'max': max(all_lengths)
                    }

            observed_ratios = []
            for i in range(len(valid_transcriptions)):
                for j in range(i + 1, len(valid_transcriptions)):
                    _, t1 = valid_transcriptions[i]
                    _, t2 = valid_transcriptions[j]
                    observed_ratios.append(_sym_ratio(t1.split(), t2.split()))

            observed_ratios.sort()
            if len(observed_ratios) == 0:   
                match_stats = None
            else:
                if self.matchmode == "iqr":
                    match_stats = {
                        'mode': 'iqr',
                        'q1': np.percentile(observed_ratios, 25),
                        'median': np.median(observed_ratios),
                        'q3': np.percentile(observed_ratios, 75),
                    }
                else:
                    match_stats = {
                        'mode': 'mean',
                        'mean': np.mean(observed_ratios),
                        'std': np.std(observed_ratios),
                        'lower': np.mean(observed_ratios) - np.std(observed_ratios),
                        'upper': np.mean(observed_ratios) + np.std(observed_ratios),
                    }

            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: (len(x[1]), -_stable_key(x[1])))
            total_comparisons = len(valid_transcriptions) - 1
            anchor_metadata_list = []
            successful_alignments = 0
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, anchor_metadata = validate_anchor_quality(longest_reference, transcription, length_stats, match_stats)
                    anchor_metadata_list.append({
                        'transcription_index': original_index,
                        'is_valid': is_valid,
                        'details': anchor_metadata
                    })
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            metadata = {
                'strategy_metric': 'longest_match_validation',
                'score': success_ratio,
                'successful_alignments': successful_alignments,
                'total_comparisons': total_comparisons,
                'reference_index': longest_original_index,
                'length_stats': length_stats,
                'match_stats': match_stats,
                'anchor_metadata': anchor_metadata_list
            }
            
            return success, longest_reference, metadata
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                if not sentence1 or not sentence2: return 0.0
                return _sym_ratio(sentence1.split(), sentence2.split())
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None

            totals = []
            for original_i, transcription_i in valid_transcriptions:
                total_similarity = math.fsum(calculate_similarity(transcription_i, transcription_j)
                                             for original_j, transcription_j in valid_transcriptions
                                             if original_j != original_i)
                totals.append((total_similarity, original_i, transcription_i))
            best_total_similarity, best_index, best_reference = max(
                totals, key=lambda x: (x[0], -_stable_key(x[2])))
            
            num_comparisons = len(valid_transcriptions) - 1 if len(valid_transcriptions) > 1 else 1
            avg_similarity = best_total_similarity / num_comparisons if num_comparisons > 0 else 0.0
            metadata = {
                'strategy_metric': 'most_common_words',
                'score': avg_similarity,
                'total_similarity_score': best_total_similarity,
                'reference_index': best_index
            }
            return True, best_reference, metadata

        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, reference_metadata = get_longest_reference(transcriptions)
        if longest_success: return "longest", longest_reference, reference_metadata
        _, common_words_reference, reference_metadata = get_common_words_reference(transcriptions)
        return "common_words", common_words_reference, reference_metadata

class Alignment:
    def main(self, reference, reference_type, reference_index, transcriptions):
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
                matcher = SequenceMatcher(None, ref_words, trans_words, autojunk=False)
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
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words, autojunk=False)
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
                matcher = SequenceMatcher(None, ref_words, trans_words, autojunk=False)
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
                        
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words, autojunk=False)
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
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results, alignment_metadata = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        else:
            alignment_results, alignment_metadata = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results, alignment_metadata

class TokenLevelVoting:
    def main(self, alignment_results):
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
            voting_metadata = {"fusion_tokens": [], "candidates_tokens": [], 'confidence_score': 0,
                               "total_models": 0, "sequence_length": 0, "selection_method": None, "voting_details": []}
            return "fusion_transcript", voting_metadata
            
                
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
            'fusion_tokens': fusion_tokens,
            'candidates_tokens': candidates_tokens,
            'confidence_score': confidence_score,
            'total_models': total_models,
            'sequence_length': sequence_length,
            'selection_method': 'token_plurality_vote',
            'voting_details': voting_details
        }

        return fusion_transcript, voting_metadata

class VotingReinforcer:
    def __init__(self, use_llm=False, max_tokens=25, chunk_size=25, overlap=3):
        self.use_llm = use_llm
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.REINFORCER = reinforcer.FusionReinforcer()

    def main(self, fusion_tokens, candidates_tokens):
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
            max_tokens=self.max_tokens,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        llm_time = time.time() - llm_start
        llm_transcript, llm_metadata = postprocess_reinforced_output(response)
        llm_metadata["llm_time"] = llm_time
        return llm_transcript, llm_metadata    

class HybridEnsemble:
    def __init__(self):
        self._input_to_fusion = {}
        self._samples_info = {}
        self._processed_results = []
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

    def main(self, filter_params={"mode": "mean"}, reference_params={"lengthmode": "iqr", "matchmode": "mean"}, 
             reinforcer_params={"use_llm": False, "max_tokens":25, "chunk_size":15, "overlap":3}, substitute=False, normalize_final_letters=True):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            transcriptions_copy = list(transcriptions)
            fusion_start = time.time()
            filtered_transcriptions, filtration_metadata = TranscriptFilter(**filter_params).main(transcriptions_copy)
            reference_type, reference, reference_metadata = ReferenceSelection(**reference_params).main(filtered_transcriptions)
            alignment_results, alignment_metadata = Alignment().main(reference=reference, reference_type=reference_type, 
                                                                     reference_index=reference_metadata["reference_index"], 
                                                                     transcriptions=filtered_transcriptions)
            fusion_transcript, voting_metadata = TokenLevelVoting().main(alignment_results)
            llm_transcript, llm_metadata = VotingReinforcer(**reinforcer_params).main(fusion_tokens=voting_metadata["fusion_tokens"],
                                                                                      candidates_tokens=voting_metadata["candidates_tokens"])
            to_be_normalized = fusion_transcript if not llm_transcript or not llm_transcript.strip() else llm_transcript
            normalized_prediction = text_processing.StandardArabicTextProcessor.main(to_be_normalized, substitute=substitute, normalize_final_letters=normalize_final_letters)
            data = {
                # --- Primary Outputs ---
                "fusion_transcript": fusion_transcript,
                "llm_transcript": llm_transcript,
                "normalized_prediction": normalized_prediction,
                # --- Diagnostic/Internal Data ---
                "fusion_time": time.time() - fusion_start,
                # --- Aggregated Metadata ---
                "metadata": {
                    "records_filtration": filtration_metadata,
                    "reference_selection": reference_metadata,
                    "alignment": alignment_metadata,
                    "voting": voting_metadata,
                    "llm_reinforcement": llm_metadata  # <-- NEW metadata section
                }
            }
            
            return {audio_path: data}

        def process_item(item):
            key, value = item
            voting_result = fuse_sample_transcriptions(key, value)
            return voting_result
        
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        if not self._input_to_fusion:
            raise ValueError("Run EnsembleInference.combine_models_transcriptions first.")

        wall_start=time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures={executor.submit(process_item,item):item for item in self.input_to_fusion.items()}
            samples_info={}
            for future in tqdm(as_completed(futures),total=len(futures),desc="Fusing Inputs..."):
                try:
                    samples_info.update(future.result())
                except Exception as e:
                    raise RuntimeError(f"Error during fusion for a sample: {e}") from e
        self._fusion_wall_time=time.time()-wall_start
        self._fusion_workers=8
                
        self._samples_info = dict(sorted(samples_info.items()))

    def eval(self, audios_chunk):
            
        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_audio_paths = list(self._samples_info.keys())
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    ref = refs_lookup.get(audio_path)
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
            raise ValueError("Run EnsembleInference.combine_models_transcriptions first.")
        
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
            raise ValueError("Run HybridEnsemble.main first.")
        
        return self._samples_info
    
    @samples_info.setter
    def samples_info(self, value):
        """
        Set the input to fusion dictionary.
        """
        self._samples_info = value

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def fusion_wall_time(self):
        return self._fusion_wall_time