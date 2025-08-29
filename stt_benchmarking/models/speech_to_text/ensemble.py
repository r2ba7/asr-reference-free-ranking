from difflib import SequenceMatcher
from collections import defaultdict
from itertools import islice
from collections import Counter, OrderedDict
import random

import numpy as np

from stt_benchmarking.utils import text_processing
from stt_benchmarking.utils import metrics
from .. import LOGGER

class EnsembleInference:
    def __init__(self):
        self.input_to_fusion = {}
    
    def align_model_records(self, *models, missing_value=None):
        """
        Align samples from multiple models based on common keys.
        Sorts by audio_path for consistent ordering.
        
        Args:
            *models: Model objects with samples_info attribute
            missing_value: Value to use when a model doesn't have a specific key (default: None)
        
        Returns:
            dict: {accuracy: [normalized_predictions_list]} where all lists have same length
            Also sets self.sorted_audio_paths for reference
        """
        if not models:
            return {}
        
        # Get all unique sample keys across all models
        all_keys = set()
        for model in models:
            if hasattr(model, 'samples_info') and model.samples_info:
                all_keys.update(model.samples_info.keys())
        
        # Sort keys (audio_paths) alphabetically for consistent ordering
        self.sorted_audio_paths = sorted(all_keys)
        
        for model in models:
            # Get model accuracy
            accuracy = None
            if hasattr(model, 'overall_metrics') and model.overall_metrics:
                accuracy = model.overall_metrics.get('average_score')
            
            if accuracy is None:
                continue  # Skip models without accuracy
            
            # Align samples - extract raw_prediction only, sorted by audio_path
            aligned_samples = []
            for audio_path in self.sorted_audio_paths:
                if (hasattr(model, 'samples_info') and 
                    model.samples_info and 
                    audio_path in model.samples_info):
                    sample = model.samples_info[audio_path].get('normalized_prediction', missing_value)
                else:
                    sample = missing_value
                
                aligned_samples.append(sample)
            
            self.input_to_fusion[accuracy] = aligned_samples
    
    @staticmethod
    def compute_weights(accuracies):
        """
        Compute model weights based on accuracy scores.
        
        Args:
            accuracies (list): List of accuracy scores for each model
            
        Returns:
            np.array: Normalized weights (higher weight for higher accuracy)
        """
        accuracies = np.array(accuracies)
        weights = np.maximum(accuracies, 1e-10)
        weights = weights / weights.sum()
        return weights
    
    @staticmethod
    def extract_additional_info(reference, longest_sentence):
        """
        Find tokens in longest sentence that aren't in reference.
        
        Args:
            reference (list): Reference sentence tokens
            longest_sentence (list): Longest sentence tokens
            
        Returns:
            list: Additional tokens not present in reference
        """
        ref_set = set(token.lower() for token in reference)
        additional_tokens = []
        for token in longest_sentence:
            if token.lower() not in ref_set:
                additional_tokens.append(token)
        
        return additional_tokens
    
    @staticmethod
    def smart_insert_tokens(base_ref, additional_tokens, all_versions):
        """
        Insert additional tokens at positions where they commonly appear in other sentences.
        
        Args:
            base_ref (list): Base reference tokens (will be modified)
            additional_tokens (list): Tokens to insert
            all_versions (list): All sentence versions for position analysis
            
        Returns:
            list: Enhanced reference with inserted tokens
        """
        enhanced_ref = base_ref.copy()
        
        # For each additional token, find where it typically appears
        for token in additional_tokens:
            # Find relative positions where this token appears in other sentences
            positions = []
            for version in all_versions:
                if token.lower() in [t.lower() for t in version]:
                    # Find first occurrence
                    for i, t in enumerate(version):
                        if t.lower() == token.lower():
                            relative_pos = i / len(version) if len(version) > 0 else 0
                            positions.append(relative_pos)
                            break
            
            if positions:
                # Insert at average relative position
                avg_relative_pos = sum(positions) / len(positions)
                insert_pos = min(int(avg_relative_pos * len(enhanced_ref)), len(enhanced_ref))
                enhanced_ref.insert(insert_pos, token)
            else:
                # Fallback: append at end
                enhanced_ref.append(token)
        
        return enhanced_ref
    
    @staticmethod
    def create_enhanced_reference(consensus_ref, versions):
        """
        Enhance consensus reference with missing information from longer sentences.
        
        Args:
            consensus_ref (list): Consensus reference tokens
            versions (list): All sentence versions
            
        Returns:
            list: Enhanced reference with additional information
        """
        if not versions:
            return consensus_ref
        
        longest = max(versions, key=len)
        
        if len(longest) <= len(consensus_ref):
            return consensus_ref  # No enhancement needed
        
        # Find additional information
        additional_info = EnsembleInference.extract_additional_info(consensus_ref, longest)
        
        if not additional_info:
            return consensus_ref
        
        # Use smart insertion to place tokens at logical positions
        enhanced_ref = EnsembleInference.smart_insert_tokens(consensus_ref, additional_info, versions)
        return enhanced_ref

    def get_consensus_reference(self, versions):
        """
        Find the sentence that shares the most vocabulary with all other sentences.
        
        Args:
            versions (list): List of tokenized sentences
            
        Returns:
            list: Tokens of the most representative sentence
        """
        if not versions:
            return []
        
        if len(versions) == 1:
            return versions[0]
        
        max_common_score = -1
        best_reference = None
        
        for i, candidate in enumerate(versions):
            common_score = 0
            candidate_set = set(token.lower() for token in candidate)
            
            # Count how many words this candidate shares with others
            for j, other in enumerate(versions):
                if i != j:
                    other_set = set(token.lower() for token in other)
                    common_score += len(candidate_set.intersection(other_set))
            
            if common_score > max_common_score:
                max_common_score = common_score
                best_reference = candidate
        
        return best_reference if best_reference is not None else versions[0]
    
    def get_filtered_longest(self, versions):
        """
        Get the longest sentence that has common words with at least one other sentence.
        Filters out complete garbage sentences with no overlap.
        
        Args:
            versions (list): List of tokenized sentences
            
        Returns:
            list: Tokens of the longest meaningful sentence
        """
        if not versions:
            return []
        
        if len(versions) == 1:
            return versions[0]
        
        # Filter out sentences that have NO common words with any other sentence
        valid_candidates = []
        
        for i, candidate in enumerate(versions):
            candidate_set = set(token.lower() for token in candidate)
            has_common_words = False
            
            for j, other in enumerate(versions):
                if i != j:
                    other_set = set(token.lower() for token in other)
                    if len(candidate_set.intersection(other_set)) > 0:
                        has_common_words = True
                        break
            
            if has_common_words:
                valid_candidates.append(candidate)
        
        # Among valid candidates, pick the longest
        if valid_candidates:
            return max(valid_candidates, key=len)
        else:
            # Ultimate fallback: just pick longest (even if garbage)
            return max(versions, key=len)
    
    def get_optimal_reference(self, versions):
        """
        Select optimal reference using consensus approach with enhancement from longest sentences.
        
        Args:
            versions (list): List of tokenized sentences
            
        Returns:
            list: Tokens of the optimal enhanced reference sentence
        """
        if not versions:
            return []
        
        # Try consensus first
        consensus_ref = self.get_consensus_reference(versions)
        
        # Enhance with missing information from longer sentences
        enhanced_ref = self.create_enhanced_reference(consensus_ref, versions)
        
        # Calculate average length for comparison
        avg_length = sum(len(v) for v in versions) / len(versions)
        
        # If enhanced reference is still significantly shorter than average, 
        # use filtered longest as fallback
        if enhanced_ref and len(enhanced_ref) >= avg_length * 0.8:
            return enhanced_ref
        else:
            filtered_longest = self.get_filtered_longest(versions)
            return filtered_longest

    def find_first_common_word(self, reference_tokens, candidate_tokens):
        """
        Find the first common word between reference and candidate (case-insensitive).
        
        Args:
            reference_tokens (list): Reference sentence tokens
            candidate_tokens (list): Candidate sentence tokens
            
        Returns:
            tuple: (ref_idx, cand_idx) or (None, None) if no common word found
        """
        for ref_idx, ref_token in enumerate(reference_tokens):
            for cand_idx, cand_token in enumerate(candidate_tokens):
                if ref_token.lower() == cand_token.lower():
                    return ref_idx, cand_idx
        return None, None

    def align_sentences(self, reference_tokens, candidate_tokens):
        """
        Align candidate to reference using first common word as anchor.
        If no common words, fall back to position-based alignment.
        Returns operation tokens and candidate values for each reference position.
        
        Args:
            reference_tokens (list): Reference sentence tokens (longest sentence)
            candidate_tokens (list): Candidate sentence tokens to align
            
        Returns:
            tuple: (operations, candidate_values) where both are same length as reference
        """
        # print(reference_tokens, candidate_tokens)
        # print("----")
        if not candidate_tokens:
            return (["<DELETE>"] * len(reference_tokens), 
                    [None] * len(reference_tokens))
        
        if not reference_tokens:
            return ([], [])
        
        operations = ["<DELETE>"] * len(reference_tokens)
        candidate_values = [None] * len(reference_tokens)
        
        # Find first common word as anchor
        ref_anchor, cand_anchor = self.find_first_common_word(reference_tokens, candidate_tokens)
        if ref_anchor is None:
            # No common words - use position-based alignment (first-to-first)
            min_length = min(len(reference_tokens), len(candidate_tokens))
            
            for i in range(min_length):
                if reference_tokens[i].lower() == candidate_tokens[i].lower():
                    operations[i] = "<KEEP>"
                    candidate_values[i] = reference_tokens[i]
                else:
                    operations[i] = "<SUBSTITUTE>"
                    candidate_values[i] = candidate_tokens[i]
            
            # Remaining reference tokens (if any) stay as <DELETE>
            return operations, candidate_values
        
        # Use anchor-based alignment
        
        # 1. ALIGN PREFIX (before anchor) - NEW LOGIC
        if ref_anchor > 0 or cand_anchor > 0:
            ref_prefix = reference_tokens[:ref_anchor]
            cand_prefix = candidate_tokens[:cand_anchor]
            
            prefix_matcher = SequenceMatcher(None, ref_prefix, cand_prefix)
            
            for tag, i1, i2, j1, j2 in prefix_matcher.get_opcodes():
                if tag == 'equal':
                    # Tokens match exactly
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        operations[i] = "<KEEP>"
                        candidate_values[i] = reference_tokens[i]
                        
                elif tag == 'replace':
                    ref_span = i2 - i1
                    cand_span = j2 - j1
                    
                    if ref_span == cand_span:
                        # 1-to-1 substitution
                        for i, j in zip(range(i1, i2), range(j1, j2)):
                            operations[i] = "<SUBSTITUTE>"
                            candidate_values[i] = candidate_tokens[j]
                    elif ref_span > cand_span:
                        # Multiple reference tokens -> fewer candidate tokens
                        if cand_span > 0:
                            operations[i1] = "<MERGE>"
                            candidate_values[i1] = candidate_tokens[j1:j2]
                            # Remaining positions stay as <DELETE>
                        # else: positions stay as <DELETE>
                    else:
                        # Fewer reference -> more candidate tokens
                        for i, j in zip(range(i1, i2), range(j1, j1 + ref_span)):
                            operations[i] = "<SUBSTITUTE>"
                            candidate_values[i] = candidate_tokens[j]
                            
                elif tag == 'delete':
                    # Reference tokens are deleted (already initialized as <DELETE>)
                    pass
                    
                elif tag == 'insert':
                    # Candidate has extra tokens - these are ignored in this alignment approach
                    # since we're aligning candidate TO reference structure
                    pass
        
        # 2. ALIGN SUFFIX (from anchor onwards) - EXISTING LOGIC
        ref_suffix = reference_tokens[ref_anchor:]
        cand_suffix = candidate_tokens[cand_anchor:]
        
        suffix_matcher = SequenceMatcher(None, ref_suffix, cand_suffix)
        
        for tag, i1, i2, j1, j2 in suffix_matcher.get_opcodes():
            # Adjust indices to account for the anchor offset
            ref_start = ref_anchor + i1
            ref_end = ref_anchor + i2
            cand_start = cand_anchor + j1
            cand_end = cand_anchor + j2
            
            if tag == 'equal':
                # Tokens match exactly
                for i, j in zip(range(ref_start, ref_end), range(cand_start, cand_end)):
                    operations[i] = "<KEEP>"
                    candidate_values[i] = reference_tokens[i]
                    
            elif tag == 'replace':
                ref_span = ref_end - ref_start
                cand_span = cand_end - cand_start
                
                if ref_span == cand_span:
                    # 1-to-1 substitution
                    for i, j in zip(range(ref_start, ref_end), range(cand_start, cand_end)):
                        operations[i] = "<SUBSTITUTE>"
                        candidate_values[i] = candidate_tokens[j]
                elif ref_span > cand_span:
                    # Multiple reference tokens -> fewer candidate tokens
                    if cand_span > 0:
                        operations[ref_start] = "<MERGE>"
                        candidate_values[ref_start] = candidate_tokens[cand_start:cand_end]
                        # Remaining positions stay as <DELETE>
                    # else: positions stay as <DELETE>
                else:
                    # Fewer reference -> more candidate (shouldn't happen with longest ref)
                    for i, j in zip(range(ref_start, ref_end), range(cand_start, cand_start + ref_span)):
                        operations[i] = "<SUBSTITUTE>"
                        candidate_values[i] = candidate_tokens[j]
                        
            elif tag == 'delete':
                # Reference tokens are deleted (already initialized as <DELETE>)
                pass
                
            elif tag == 'insert':
                # Candidate has extra tokens - ignored in this alignment approach
                pass
        
        return operations, candidate_values
    
    def align_all_sentences(self, sentence_lists):
        """
        Align all sentence lists using the longest sentence as reference.
        
        Args:
            sentence_lists (list): List of sentence lists from different models
            
        Returns:
            list: List of alignment results for each sentence position
        """
        # Handle missing sentences by padding with empty strings
        num_sentences = max(len(sents) for sents in sentence_lists)
        for sents in sentence_lists:
            while len(sents) < num_sentences:
                sents.append("")
        
        aligned_results = []
        for i in range(num_sentences):
            versions = []
            for sent_list in sentence_lists:
                try:
                    sent = sent_list[i].strip()
                    tokens = sent.split() if sent else []
                    versions.append(tokens)

                except Exception as e:
                    LOGGER.error(f"Error processing sentence {i+1}, failed: {e}")
                    versions.append([])
                # print(tokens)
            
            # print("Versions:", versions)
            # Select optimal reference using consensus + filtered longest
            # Reference is correct
            reference = self.get_optimal_reference(versions)
            # print("Reference:", reference)
            # Align all versions to the reference
            sentence_alignments = []
            for tokens in versions:
                operations, candidate_values = self.align_sentences(reference, tokens)
                sentence_alignments.append({
                    'operations': operations,
                    'candidate_values': candidate_values
                })
            
            aligned_results.append({
                'reference': reference,
                'alignments': sentence_alignments
            })
            # break
        return aligned_results

    def fuse_aligned_sentences(self, aligned_results, weights):
        """
        Fuse sentences after alignment using weighted voting.
        
        Args:
            aligned_results (list): Results from align_all_sentences()
            weights (np.array): Model weights based on errors
            
        Returns:
            list: List of fused sentences
        """
        fused_sentences = []
        
        for result in aligned_results:
            reference = result['reference']
            # print(reference)
            alignments = result['alignments']
            # print(alignments)
            if not reference:
                fused_sentences.append("")
                continue
            # print('---')
            # Initialize voting matrices
            alignment_matrix = [defaultdict(float) for _ in range(len(reference))]
            operation_matrix = [defaultdict(float) for _ in range(len(reference))]
            
            # Collect votes from each model's alignment
            for j, alignment in enumerate(alignments):
                operations = alignment['operations']
                candidate_values = alignment['candidate_values']
                vote_weight = weights[j]
                
                for idx, (op, value) in enumerate(zip(operations, candidate_values)):
                    if op == "<KEEP>":
                        operation_matrix[idx]["equal"] += vote_weight
                        alignment_matrix[idx][reference[idx]] += vote_weight
                    elif op == "<SUBSTITUTE>":
                        operation_matrix[idx]["replace"] += vote_weight
                        if value:
                            alignment_matrix[idx][value] += vote_weight
                    elif op == "<MERGE>":
                        operation_matrix[idx]["replace"] += vote_weight
                        if value:
                            merged_text = ' '.join(value) if isinstance(value, list) else str(value)
                            alignment_matrix[idx][merged_text] += vote_weight
                    elif op == "<DELETE>":
                        operation_matrix[idx]["delete"] += vote_weight
                        alignment_matrix[idx][""] += vote_weight
            
            # Fuse tokens based on votes
            fused = []
            for idx, (op_votes, word_votes) in enumerate(zip(operation_matrix, alignment_matrix)):
                if op_votes:
                    best_op = max(op_votes.items(), key=lambda x: x[1])[0]
                    total_weight = sum(op_votes.values())
                    
                    if best_op == "delete" and op_votes["delete"] > 0.6 * total_weight:
                        continue  # Skip token if deletion strongly supported
                    elif best_op in ["equal", "replace"]:
                        if word_votes:
                            # Check if weights are close (within 10%)
                            valid_words = {k: v for k, v in word_votes.items() if k != ""}
                            if valid_words:
                                max_weight = max(valid_words.values())
                                close_weights = [k for k, v in valid_words.items() if v >= 0.9 * max_weight]
                                if len(close_weights) > 1:
                                    # Fallback to reference token for ties
                                    fused.append(reference[idx])
                                else:
                                    # Use highest-weighted word
                                    best_word = max(valid_words.items(), key=lambda x: x[1])[0]
                                    fused.append(best_word)
                else:
                    # No votes, keep reference
                    fused.append(reference[idx])
            
            fused_sentences.append(' '.join(fused))
        
        return fused_sentences

    def ensemble_texts(self):
        """
        Ensemble texts by first aligning then fusing sentences.
        
        Args:
            accuracy_to_sentences (dict): Dict[float, List[str]]
                Example: {0.78: ["I sit down", "I go home"], 0.65: ["I sat down", "I went home"]}
        
        Returns:
            list: List of final ensembled sentences
        """
        # Sort accuracies (descending - best first) and extract sentences
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        sorted_items = sorted(self.input_to_fusion.items(), key=lambda x: x[0], reverse=True)
        accuracies = [a for a, _ in sorted_items]
        sentence_lists = [sents for _, sents in sorted_items]
        # Compute weights based on accuracies
        weights = self.compute_weights(accuracies)
        # Align all sentences
        LOGGER.info("Step 1: Aligning sentences...")
        aligned_results = self.align_all_sentences(sentence_lists)
        # Fuse aligned sentences using weighted voting
        LOGGER.info("Step 2: Fusing aligned sentences...")
        fused_sentences = self.fuse_aligned_sentences(aligned_results, weights)
        return fused_sentences
    

class EnsembleInferenceOld:
    def __init__(self):
        pass
    
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
    
    @staticmethod
    def find_common_anchors(reference_tokens, candidate_tokens):
        """
        Find common words between reference and candidate to use as alignment anchors.
        
        Args:
            reference_tokens (list): Reference sentence tokens
            candidate_tokens (list): Candidate sentence tokens
            
        Returns:
            list: List of (ref_idx, cand_idx) tuples for anchor points
        """
        anchors = []
        used_cand_indices = set()
        
        for ref_idx, ref_token in enumerate(reference_tokens):
            for cand_idx, cand_token in enumerate(candidate_tokens):
                if (ref_token.lower() == cand_token.lower() and 
                    cand_idx not in used_cand_indices):
                    anchors.append((ref_idx, cand_idx))
                    used_cand_indices.add(cand_idx)
                    break
        
        return anchors
    
    def align_sentences(self, reference_tokens, candidate_tokens):
        """
        Align candidate to reference using SequenceMatcher but maintain reference length.
        Returns operation tokens and candidate values for each reference position.
        
        Args:
            reference_tokens (list): Reference sentence tokens (longest sentence)
            candidate_tokens (list): Candidate sentence tokens to align
            
        Returns:
            tuple: (operations, candidate_values) where both are same length as reference
                   operations: list of operation types (<KEEP>, <SUBSTITUTE>, <DELETE>, <MERGE>)
                   candidate_values: list of candidate tokens or None for each position
        """
        if not candidate_tokens:
            return (["<DELETE>"] * len(reference_tokens), 
                    [None] * len(reference_tokens))
        
        if not reference_tokens:
            return ([], [])
        
        # Use SequenceMatcher to get alignment operations
        matcher = SequenceMatcher(None, reference_tokens, candidate_tokens)
        operations = ["<DELETE>"] * len(reference_tokens)
        candidate_values = [None] * len(reference_tokens)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Tokens match exactly - keep reference tokens
                for i in range(i1, i2):
                    operations[i] = "<KEEP>"
                    candidate_values[i] = reference_tokens[i]  # Same as reference
                    
            elif tag == 'replace':
                ref_span = i2 - i1
                cand_span = j2 - j1
                
                if ref_span == cand_span:
                    # 1-to-1 substitution
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        operations[i] = "<SUBSTITUTE>"
                        candidate_values[i] = candidate_tokens[j]
                elif ref_span > cand_span:
                    # Multiple reference tokens -> fewer candidate tokens (merge scenario)
                    # Mark first position as MERGE with all candidate tokens
                    if cand_span > 0:
                        operations[i1] = "<MERGE>"
                        candidate_values[i1] = candidate_tokens[j1:j2]  # List of tokens
                        # Mark remaining reference positions as DELETE
                        for i in range(i1 + 1, i2):
                            operations[i] = "<DELETE>"
                            candidate_values[i] = None
                    else:
                        # No candidate tokens - all deletes
                        for i in range(i1, i2):
                            operations[i] = "<DELETE>"
                            candidate_values[i] = None
                else:
                    # Fewer reference tokens -> more candidate tokens
                    # This shouldn't happen since reference is longest, but handle it
                    for i, j in zip(range(i1, i2), range(j1, j1 + ref_span)):
                        operations[i] = "<SUBSTITUTE>"
                        candidate_values[i] = candidate_tokens[j]
                    
            elif tag == 'delete':
                # Reference tokens are deleted
                for i in range(i1, i2):
                    operations[i] = "<DELETE>"
                    candidate_values[i] = None
                    
            elif tag == 'insert':
                # This should never happen since reference is longest
                pass
        
        return operations, candidate_values
    
    def align_all_sentences(self, sentence_lists):
        """
        Align all sentence lists using the longest sentence as reference.
        
        Args:
            sentence_lists (list): List of sentence lists from different models
            
        Returns:
            list: List of alignment results for each sentence position
        """
        # Handle missing sentences by padding with empty strings
        num_sentences = max(len(sents) for sents in sentence_lists)
        for sents in sentence_lists:
            while len(sents) < num_sentences:
                sents.append("")
        
        aligned_results = []
        
        for i in range(num_sentences):
            # Extract tokenized sentences for position i
            versions = []
            for sent_list in sentence_lists:
                sent = sent_list[i].strip()
                tokens = sent.split() if sent else []
                versions.append(tokens)
            
            # Select longest sentence as reference
            reference = max(versions, key=len, default=[])
            
            # Align all versions to the reference
            sentence_alignments = []
            for tokens in versions:
                operations, candidate_values = self.align_sentences(reference, tokens)
                sentence_alignments.append({
                    'operations': operations,
                    'candidate_values': candidate_values
                })
            
            aligned_results.append({
                'reference': reference,
                'alignments': sentence_alignments
            })
        
        return aligned_results

class EnsembleInferenceRefactored:
    def __init__(self):
        self._input_to_fusion = {}
        self._fusion_results = []
        self._processed_results = []
        self._overall_metrics = None
    
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
    
    def combine_models_transcriptions(self, *models, missing_value=''):
        """
        Align samples from multiple models into an audio-centric structure.
        
        Args:
            *models: Model objects with samples_info + overall_metrics
            missing_value: Placeholder when a model has no prediction for a sample.
        
        Returns:
            dict: {audio_path: {score: transcript}}
            Also sets self.sorted_audio_paths for reference
        """
        if not models:
            return {}

        # Collect all audio keys
        all_keys = set()
        for model in models:
            if hasattr(model, 'samples_info') and model.samples_info:
                all_keys.update(model.samples_info.keys())

        # Sort for consistency
        self.sorted_audio_paths = sorted(all_keys)

        # Build audio-centric dict
        combined = {audio_path: {} for audio_path in self.sorted_audio_paths}

        for model in models:
            score = None
            if hasattr(model, 'overall_metrics') and model.overall_metrics:
                score = model.overall_metrics.get('average_score')

            if score is None:
                continue  # Skip models with no score

            for audio_path in self.sorted_audio_paths:
                if (hasattr(model, 'samples_info') and 
                    model.samples_info and 
                    audio_path in model.samples_info):
                    transcript = model.samples_info[audio_path].get('normalized_prediction', missing_value)
                else:
                    transcript = missing_value

                combined[audio_path][score] = transcript

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
    
    @staticmethod
    def validate_anchor_quality(reference, transcription):
        """
        Validate that anchors represent meaningful common structure using SequenceMatcher
        
        Returns:
            tuple: (is_valid, anchor_info)
        """
        anchors = EnsembleInferenceRefactored.find_first_anchors(reference, transcription)
        if not anchors:
            return False, None
            
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
        min_len = min(len(ref_remaining), len(trans_remaining))
        if min_len <= 2:
            is_valid = additional_matches == min_len
        else:
            is_valid = additional_matches >= 2
            
        return is_valid, {
            "positions": anchors, 
            "additional_matches": additional_matches,
            "min_required": 2 if min_len > 2 else min_len
        }

    # Done reference
    def get_reference_from_transcriptions(self, transcriptions):
        def get_longest_reference(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            successful_alignments = 0
            total_comparisons = len(valid_transcriptions) - 1
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, anchor_info = self.validate_anchor_quality(longest_reference, transcription)
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            return success, longest_reference, longest_original_index

        def get_longest_reference_fallback(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            return True, longest_reference, longest_original_index
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                words1 = sentence1.split()
                words2 = sentence2.split()
                matcher = SequenceMatcher(None, words1, words2)
                return matcher.ratio()
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None

            best_total_similarity = -1
            best_reference = None
            best_index = None
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
            
            if best_reference and best_total_similarity > 0:
                return True, best_reference, best_index
            
            return False, None, None

        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, longest_index = get_longest_reference(transcriptions)
        if longest_success:
            return "longest", longest_reference, longest_index
        else:
            common_words_success, common_words_reference, common_words_index = get_common_words_reference(transcriptions)
            if common_words_success:
                return "common_words", common_words_reference, common_words_index
            else:
                _, longest_reference, longest_original_index = get_longest_reference_fallback(transcriptions)
                return "longest_fallback", longest_reference, longest_original_index

    def align_transcriptions_to_reference(self, weights, reference, reference_type, reference_index, transcriptions):
        """
        Simple alignment function that extracts operations and candidate values.
        
        Args:
            reference (str): Reference transcription
            transcription (str): Candidate transcription to align
            
        Returns:
            tuple: (operations, candidate_values)
        """
        # Wont change
        def align_with_longest_strategy(weights, reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            alignment_results = []
            ref_words = reference.split()
            for model_index, transcription in enumerate( transcriptions):
                alignment_result = {
                    'model_weight': weights[model_index],
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                if not transcription:
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
                                candidate_values[i] = ref_words[i]  # Same as trans_words[trans_start + (i - ref_start)]
                                
                        elif op == 'replace':
                            for i in range(ref_start, ref_end):
                                operations[i] = "<REPLACE>"
                                trans_idx = trans_start + (i - ref_start)
                                if trans_idx < trans_end:
                                    candidate_values[i] = trans_words[trans_idx]
                                else:
                                    candidate_values[i] = None
                                    
                        elif op == 'delete':
                            for i in range(ref_start, ref_end):
                                operations[i] = "<DELETE>"
                                candidate_values[i] = None
                                
                        elif op == 'insert':
                            # Words exist in transcription but not in reference
                            # These don't affect our reference-based operations array
                            # but could be logged for debugging
                            pass
                    alignment_result['operations'] = operations
                    alignment_result['tokens'] = candidate_values
                alignment_results.append(alignment_result)
            return alignment_results        

        # Maybe needs small tweaking
        def align_with_common_words_strategy(weights, reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            ref_words = reference.split()
            max_length = max(len(t.split()) for _, t in valid_transcriptions)
            longest_idx, longest_trans = max(valid_transcriptions, key=lambda x: len(x[1].split()))
            longest_words = longest_trans.split()
            padded_transcriptions_words = {}
            padded_ref_words = ref_words.copy()
            for model_index, transcription in enumerate(transcriptions):
                if model_index == longest_idx:
                    padded_transcriptions_words[model_index] = longest_words
                    continue

                if model_index == reference_index:
                    matcher = SequenceMatcher(None, longest_words, padded_ref_words)
                    insert_positions = []
                    for op, long_start, long_end, ref_start, ref_end in matcher.get_opcodes():
                        if op == 'delete':
                            for i in range(long_start, long_end):
                                insert_positions.append((ref_start, longest_words[i]))
                    for pos, _ in sorted(insert_positions, key=lambda x: x[0]):
                        padded_ref_words.insert(pos, None)
                    padded_ref_words = padded_ref_words[:max_length] if len(padded_ref_words) > max_length else padded_ref_words + [None] * (max_length - len(padded_ref_words))
                    padded_transcriptions_words[model_index] = padded_ref_words
                else:
                    if transcription is None:
                        continue
                    else:
                        transcription_words = transcription.split()
                        matcher = SequenceMatcher(None, longest_words, transcription_words)
                        insert_positions = []
                        for op, long_start, long_end, ref_start, ref_end in matcher.get_opcodes():
                            if op == 'delete':
                                for i in range(long_start, long_end):
                                    insert_positions.append((ref_start, longest_words[i]))
                        for pos, _ in sorted(insert_positions, key=lambda x: x[0]):
                            transcription_words.insert(pos, None)
                        transcription_words = transcription_words[:max_length] if len(transcription_words) > max_length else transcription_words + [None] * (max_length - len(transcription_words))
                        # Save padded transcription words
                        padded_transcriptions_words[model_index] = transcription_words
                        # transcriptions[model_index] = ' '.join([w for w in transcription_words if w is not None])

            padded_ref_length = len(padded_ref_words)

            # Step 2: Align all transcriptions to padded reference
            inserted_words = {}
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_weight': weights[model_index],
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                if model_index in padded_transcriptions_words:
                    trans_words = padded_transcriptions_words[model_index]
                else:
                    trans_words = []

                if not trans_words:
                    alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                    alignment_result['tokens'] = [None] * padded_ref_length
                else:
                    operations = ["<DELETE>"] * padded_ref_length
                    tokens = [None] * padded_ref_length
                    matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                    inserted_words[model_index] = []
                    for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                        # ref_text = padded_ref_words[ref_start:ref_end]
                        # trans_text = trans_words[trans_start:trans_end]

                        # print(f"{op.upper():<9} | "
                        #     f"ref[{ref_start}:{ref_end}] = '{ref_text}' | "
                        #     f"trans[{trans_start}:{trans_end}] = '{trans_text}'")
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
                                else:
                                    tokens[i] = None
                        elif op == 'delete':
                            for i in range(ref_start, ref_end):
                                operations[i] = "<DELETE>"
                                tokens[i] = None
                        elif op == 'insert':
                            inserted_words[model_index].extend(trans_words[trans_start:trans_end])
                    alignment_result['operations'] = operations
                    alignment_result['tokens'] = tokens
                    # print("--------------")
                alignment_results.append(alignment_result)

            return alignment_results
        
        def align_with_longest_fallback_strategy(weights, reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            pass
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results = align_with_longest_strategy(weights=weights, reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words":
            alignment_results = align_with_common_words_strategy(weights=weights, reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        else: # Handle it later
            alignment_results = align_with_longest_strategy(weights=weights, reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results

    def unweighted_voting_scheme(self, alignment_results):
        """
        Implement majority voting scheme for operations and random voting for tokens.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with operations, tokens, and metadata
        """
        def collect_position_votes(alignment_results, position):
            """Collect all votes for a specific position"""
            position_votes = {
                'operations': [],
                'tokens': [],
                'model_indices': [],
                'is_reference_flags': []
            }
            
            for result in alignment_results:
                position_votes['operations'].append(result['operations'][position])
                position_votes['tokens'].append(result['tokens'][position])
                position_votes['model_indices'].append(result['model_index'])
                position_votes['is_reference_flags'].append(result['is_reference'])
            
            return position_votes
        
        def vote_for_operation(position_votes):
            """Determine majority operation with tie-breaking"""
            operation_counts = Counter(position_votes['operations'])
            majority_operation = operation_counts.most_common(1)[0][0]
            
            # Handle ties with priority order
            max_count = operation_counts.most_common(1)[0][1]
            tied_operations = [op for op, count in operation_counts.items() if count == max_count]
            if len(tied_operations) > 1:
                priority_order = ["<KEEP>", "<REPLACE>", "<INSERT>", "<SKIP>", "<DELETE>"]
                for preferred_op in priority_order:
                    if preferred_op in tied_operations:
                        majority_operation = preferred_op
                        break
            
            return majority_operation, operation_counts
        
        def vote_for_token(position_votes, majority_operation):
            """Determine final token based on operation with random selection"""
            final_token = None
            
            if majority_operation == "<KEEP>":
                # Find reference token or use first available
                for i, is_ref in enumerate(position_votes['is_reference_flags']):
                    if is_ref and position_votes['operations'][i] in ["<KEEP>", "<REPLACE>"]:
                        final_token = position_votes['tokens'][i]
                        break
                
                if final_token is None:
                    final_token = position_votes['tokens'][0]
                    
            elif majority_operation in ["<REPLACE>", "<INSERT>"]:
                # Random selection among models that chose this operation
                operation_tokens = []
                
                for i, op in enumerate(position_votes['operations']):
                    if op == majority_operation and position_votes['tokens'][i] is not None:
                        operation_tokens.append(position_votes['tokens'][i])
                
                if operation_tokens:
                    final_token = random.choice(operation_tokens)
                    
            elif majority_operation in ["<DELETE>", "<SKIP>"]:
                final_token = None
            
            return final_token
        
        def create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes):
            """Create detailed voting information for a position"""
            token_weights = {}
            
            if majority_operation in ["<REPLACE>", "<INSERT>"]:
                for token in set(position_votes['tokens']):
                    if token is not None:
                        token_weights[token] = sum(
                            1  # Count occurrences instead of weights
                            for i, t in enumerate(position_votes['tokens']) 
                            if t == token and position_votes['operations'][i] == majority_operation
                        )
            
            return {
                'position': position,
                'operation_votes': dict(operation_counts),
                'majority_operation': majority_operation,
                'final_token': final_token,
                'models_voted': len(position_votes['operations']),
                'token_weights': token_weights  # Now shows counts instead of weights
            }
        
        def construct_final_transcription(final_operations, final_tokens):
            """Build the final transcription from operations and tokens"""
            final_transcription_words = []
            
            for operation, token in zip(final_operations, final_tokens):
                if operation in ["<KEEP>", "<REPLACE>", "<INSERT>"] and token is not None:
                    final_transcription_words.append(token)
            
            return " ".join(final_transcription_words)
        
        def calculate_confidence_score(alignment_results, operations_length):
            """Calculate overall confidence based on operation agreement"""
            total_positions = operations_length
            operation_confidence = sum(
                max(Counter([result['operations'][i] for result in alignment_results]).values()) / len(alignment_results)
                for i in range(total_positions)
            ) / total_positions if total_positions > 0 else 0
            
            return operation_confidence
        
        def create_metadata(alignment_results, final_operations):
            """Create metadata about the voting results"""
            return {
                'reference_type': alignment_results[0].get('reference_type', 'unknown'),
                'total_keep': final_operations.count('<KEEP>'),
                'total_replace': final_operations.count('<REPLACE>'),
                'total_insert': final_operations.count('<INSERT>'),
                'total_delete': final_operations.count('<DELETE>'),
                'total_skip': final_operations.count('<SKIP>')
            }
        
        if not alignment_results or len(alignment_results) == 0:
            voting_result = {
                'final_transcription': '',
                'final_operations': [],
                'final_tokens': [],
                'voting_details': {},
                'confidence_score': 0,
                'total_models': 0,
                'operations_length': 0,
                'metadata': {}
            }
            return voting_result
        
        # Main voting logic
        operations_length = len(alignment_results[0]['operations'])
        final_operations = []
        final_tokens = []
        voting_details = []
        
        # Process each position
        for position in range(operations_length):
            position_votes = collect_position_votes(alignment_results, position)
            majority_operation, operation_counts = vote_for_operation(position_votes)
            final_token = vote_for_token(position_votes, majority_operation)
            
            final_operations.append(majority_operation)
            final_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes)
            voting_details.append(voting_detail)
        
        # Construct final results
        final_transcription = construct_final_transcription(final_operations, final_tokens)
        confidence_score = calculate_confidence_score(alignment_results, operations_length)
        metadata = create_metadata(alignment_results, final_operations)
        
        voting_result = {
            'final_transcription': final_transcription,
            'final_operations': final_operations,
            'final_tokens': final_tokens,
            'voting_details': voting_details,
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'operations_length': operations_length,
            'metadata': metadata
        }
        
        return voting_result
    
    def voting_scheme(self, alignment_results):
        """
        Implement majority voting scheme for operations and weighted voting for tokens.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with operations, tokens, and metadata
        """
        
        def collect_position_votes(alignment_results, position):
            """Collect all votes for a specific position"""
            position_votes = {
                'operations': [],
                'tokens': [],
                'weights': [],
                'model_indices': [],
                'is_reference_flags': []
            }
            
            for result in alignment_results:
                position_votes['operations'].append(result['operations'][position])
                position_votes['tokens'].append(result['tokens'][position])
                position_votes['weights'].append(result['model_weight'])
                position_votes['model_indices'].append(result['model_index'])
                position_votes['is_reference_flags'].append(result['is_reference'])
            
            return position_votes
        
        def vote_for_operation(position_votes):
            """Determine majority operation with tie-breaking"""
            operation_counts = Counter(position_votes['operations'])
            majority_operation = operation_counts.most_common(1)[0][0]
            
            # Handle ties with priority order
            max_count = operation_counts.most_common(1)[0][1]
            tied_operations = [op for op, count in operation_counts.items() if count == max_count]
            if len(tied_operations) > 1:
                priority_order = ["<KEEP>", "<REPLACE>", "<INSERT>", "<SKIP>", "<DELETE>"]
                for preferred_op in priority_order:
                    if preferred_op in tied_operations:
                        majority_operation = preferred_op
                        break
            
            return majority_operation, operation_counts
        
        def vote_for_token(position_votes, majority_operation):
            """Determine final token based on operation and weights"""
            final_token = None
            
            if majority_operation == "<KEEP>":
                # Find reference token or use first available
                for i, is_ref in enumerate(position_votes['is_reference_flags']):
                    if is_ref and position_votes['operations'][i] in ["<KEEP>", "<REPLACE>"]:
                        final_token = position_votes['tokens'][i]
                        break
                
                if final_token is None:
                    final_token = position_votes['tokens'][0]
                    
            elif majority_operation in ["<REPLACE>", "<INSERT>"]:
                # Weighted voting among models that chose this operation
                operation_tokens = {}
                
                for i, op in enumerate(position_votes['operations']):
                    if op == majority_operation and position_votes['tokens'][i] is not None:
                        token = position_votes['tokens'][i]
                        weight = position_votes['weights'][i]
                        operation_tokens[token] = operation_tokens.get(token, 0) + weight
                
                if operation_tokens:
                    final_token = max(operation_tokens.items(), key=lambda x: x[1])[0]
                    
            elif majority_operation in ["<DELETE>", "<SKIP>"]:
                final_token = None
            
            return final_token
        
        def create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes):
            """Create detailed voting information for a position"""
            token_weights = {}
            
            if majority_operation in ["<REPLACE>", "<INSERT>"]:
                for token in set(position_votes['tokens']):
                    if token is not None:
                        token_weights[token] = sum(
                            position_votes['weights'][i] 
                            for i, t in enumerate(position_votes['tokens']) 
                            if t == token and position_votes['operations'][i] == majority_operation
                        )
            
            return {
                'position': position,
                'operation_votes': dict(operation_counts),
                'majority_operation': majority_operation,
                'final_token': final_token,
                'models_voted': len(position_votes['operations']),
                'token_weights': token_weights
            }
        
        def construct_final_transcription(final_operations, final_tokens):
            """Build the final transcription from operations and tokens"""
            final_transcription_words = []
            
            for operation, token in zip(final_operations, final_tokens):
                if operation in ["<KEEP>", "<REPLACE>", "<INSERT>"] and token is not None:
                    final_transcription_words.append(token)
            
            return " ".join(final_transcription_words)
        
        def calculate_confidence_score(alignment_results, operations_length):
            """Calculate overall confidence based on operation agreement"""
            total_positions = operations_length
            operation_confidence = sum(
                max(Counter([result['operations'][i] for result in alignment_results]).values()) / len(alignment_results)
                for i in range(total_positions)
            ) / total_positions if total_positions > 0 else 0
            
            return operation_confidence
        
        def create_metadata(alignment_results, final_operations):
            """Create metadata about the voting results"""
            return {
                'reference_type': alignment_results[0].get('reference_type', 'unknown'),
                'total_keep': final_operations.count('<KEEP>'),
                'total_replace': final_operations.count('<REPLACE>'),
                'total_insert': final_operations.count('<INSERT>'),
                'total_delete': final_operations.count('<DELETE>'),
                'total_skip': final_operations.count('<SKIP>')
            }
        
        if not alignment_results or len(alignment_results) == 0:
            voting_result = {
                'final_transcription': '',
                'final_operations': [],
                'final_tokens': [],
                'voting_details': {},
                'confidence_score': 0,
                'total_models': 0,
                'operations_length': 0,
                'metadata': {}
            }
            return voting_result
        
        # Main voting logic
        operations_length = len(alignment_results[0]['operations'])
        final_operations = []
        final_tokens = []
        voting_details = []
        
        # Process each position
        for position in range(operations_length):
            position_votes = collect_position_votes(alignment_results, position)
            majority_operation, operation_counts = vote_for_operation(position_votes)
            final_token = vote_for_token(position_votes, majority_operation)
            
            final_operations.append(majority_operation)
            final_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes)
            voting_details.append(voting_detail)
        
        # Construct final results
        final_transcription = construct_final_transcription(final_operations, final_tokens)
        confidence_score = calculate_confidence_score(alignment_results, operations_length)
        metadata = create_metadata(alignment_results, final_operations)
        
        voting_result = {
            'final_transcription': final_transcription,
            'final_operations': final_operations,
            'final_tokens': final_tokens,
            'voting_details': voting_details,
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'operations_length': operations_length,
            'metadata': metadata
        }
        
        return voting_result

    def fusion(self):
        def fuse_sample_transcriptions(weights, transcriptions):
            reference_type, reference, reference_index = self.get_reference_from_transcriptions(transcriptions)
            alignment_results = self.align_transcriptions_to_reference(weights=weights, reference=reference, reference_type=reference_type, reference_index=reference_index, transcriptions=transcriptions)
            candidates_tokens = [element['tokens'] for element in alignment_results]
            voting_result = self.unweighted_voting_scheme(alignment_results)
            voting_result["candidates_tokens"] = candidates_tokens
            return voting_result

        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        for key, value in self.input_to_fusion.items():
            audio_path = key
            accuracies = []
            transcriptions_lists = []
            for score, transcript in value.items():
                accuracies.append(score)
                transcriptions_lists.append(transcript)

            weights = self.compute_weights(accuracies)
            voting_result = fuse_sample_transcriptions(weights, transcriptions_lists)
            voting_result['audio_path'] = audio_path
            self._fusion_results.append(voting_result)
            self._fusion_results = sorted(self._fusion_results, key=lambda x: x["audio_path"])
        
        # sorted_items = sorted(self.input_to_fusion.items(), key=lambda x: x[0], reverse=True)
        # accuracies = [a for a, _ in sorted_items]
        # transcriptions_lists = [transcriptions for _, transcriptions in sorted_items]
        # weights = self.compute_weights(accuracies)
        # for _, transcriptions_group in enumerate(zip(*transcriptions_lists)):
        #     # print(f"Processing index {i}")
        #     # print(transcriptions_group)
        #     voting_result = fuse_sample_transcriptions(weights, transcriptions_group)
        #     self._fusion_results.append(voting_result)

            # print('--------------------------------------------')
    
    def reorder_samples_info(self, records):
        ordered_info = OrderedDict()
        for record in records:
            audio_path = record["audio_path"]
            if audio_path in self._samples_info:
                ordered_info[audio_path] = self._samples_info[audio_path]
        self._samples_info = ordered_info

    def _process_fusion_results(self):
        final_transcripts = [result['final_transcription'] for result in self._fusion_results]
        self._normalized_hypths = text_processing.StandardArabicTextProcessor.normalize_texts(final_transcripts)
    
    def evaluate(self, normalized_transcriptions):
        self._process_fusion_results()
        self._overall_metrics = metrics.StandardSTTMetrics.evaluate(refs=normalized_transcriptions, hyps=self._normalized_hypths)

    def summary_of_evaluation(self):
        if not hasattr(self, 'overall_metrics') or not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run EnsembleInferenceRefactored.evaluate first.")
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
    def fusion_results(self):
        """
        Get the fusion results.
        """
        if not self._fusion_results:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._fusion_results
    
    @property
    def processed_results(self):
        """
        Get the processed results after fusion.
        """
        if not self._processed_results:
            raise ValueError("Run EnsembleInference._process_fusion_results first.")
        return self._processed_results

    @property
    def overall_metrics(self):
        return self._overall_metrics
