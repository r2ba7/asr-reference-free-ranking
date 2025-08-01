from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
from . import LOGGER

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
                    sample = model.samples_info[audio_path].get('raw_prediction', missing_value)
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
        
        # Align from the anchor point onwards using SequenceMatcher
        ref_suffix = reference_tokens[ref_anchor:]
        cand_suffix = candidate_tokens[cand_anchor:]
        
        matcher = SequenceMatcher(None, ref_suffix, cand_suffix)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
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
                
        # Everything before the anchor in reference should be <DELETE> (already set)
        # Everything before the anchor in candidate is ignored
        
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
                try:
                    sent = sent_list[i].strip()
                    tokens = sent.split() if sent else []
                    versions.append(tokens)
                except Exception as e:
                    LOGGER.error(f"Error processing sentence {i+1}, failed: {e}")
                    versions.append([])
            
            # Select optimal reference using consensus + filtered longest
            reference = self.get_optimal_reference(versions)
            
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
            alignments = result['alignments']
            
            if not reference:
                fused_sentences.append("")
                continue
            
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
        
        # Step 1: Align all sentences
        LOGGER.info("Step 1: Aligning sentences...")
        aligned_results = self.align_all_sentences(sentence_lists)
        
        # Step 2: Compute weights based on accuracies
        weights = self.compute_weights(accuracies)
        
        # Step 3: Fuse aligned sentences using weighted voting
        LOGGER.info("Step 2: Fusing aligned sentences...")
        fused_sentences = self.fuse_aligned_sentences(aligned_results, weights)
        return fused_sentences