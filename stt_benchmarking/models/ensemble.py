from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np

class EnsembleInference:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_weights(errors):
        """
        Compute model weights based on inverse error scores.
        
        Args:
            errors (list): List of error scores for each model
            
        Returns:
            np.array: Normalized weights (higher weight for lower error)
        """
        errors = np.array(errors)
        inv = 1.0 / np.maximum(errors, 1e-10)  # Avoid division by zero
        weights = inv / inv.sum()
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