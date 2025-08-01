import os
import random
import gc

import torch
import random
from collections import defaultdict
import numpy as np


def sample_records_by_duration_buckets(records, bucket_size=10, no_samples=50, seed=None):
    """
    Sample records proportionally from duration buckets.

    Args:
        records (list of dict): Each record must have "audio_duration".
        bucket_size (int): Width of each duration bucket in seconds.
        no_samples (int): Total number of records to sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List[dict]: Sampled records.
    """
    if seed is not None:
        random.seed(seed)

    # 1. Group records into buckets
    buckets = defaultdict(list)
    for record in records:
        duration = record["audio_duration"]
        bucket_id = int(duration // bucket_size)
        buckets[bucket_id].append(record)

    # 2. Compute sampling probabilities
    bucket_counts = {k: len(v) for k, v in buckets.items()}
    total_count = sum(bucket_counts.values())
    sampling_probs = {k: count / total_count for k, count in bucket_counts.items()}

    # 3. Raw allocation (floating point), then floor to integers
    raw_allocations = {k: sampling_probs[k] * no_samples for k in sampling_probs}
    rounded_allocations = {k: int(np.floor(v)) for k, v in raw_allocations.items()}

    # 4. Adjust to match the exact number of samples
    remaining = no_samples - sum(rounded_allocations.values())
    if remaining > 0:
        residuals = {k: raw_allocations[k] - rounded_allocations[k] for k in sampling_probs}
        top_buckets = sorted(residuals.items(), key=lambda x: -x[1])[:remaining]
        for k, _ in top_buckets:
            rounded_allocations[k] += 1

    # 5. Sample without replacement from each bucket
    final_samples = []
    for k, n_samples in rounded_allocations.items():
        bucket_records = buckets[k]
        if len(bucket_records) >= n_samples:
            final_samples.extend(random.sample(bucket_records, n_samples))
        else:
            final_samples.extend(bucket_records)  # fallback: use all available

    return final_samples

def pick_random_records(records, no_samples):
    if no_samples > len(records):
        raise ValueError("Requested more samples than available in the records.")
    
    return random.sample(records, no_samples)


def cleanup_memory(model=None, processor=None):
    if model is not None:
        del model
    if processor is not None:
        del processor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def enhanced_arabic_fusion(error_to_sentences: dict):
    """
    error_to_sentences: Dict[float, List[str]]
        Example:
        {
            45.078: ["I sit down", "I go home"],
            72.401: ["I sat down", "I went home"],
            61.362: ["I sit down", "I return home"]
        }
    Returns: List[str] of fused sentences
    """
    # Sort errors and extract sentences
    sorted_items = sorted(error_to_sentences.items(), key=lambda x: x[0])
    errors = [e for e, _ in sorted_items]
    sentence_lists = [sents for _, sents in sorted_items]
    
    # Handle missing sentences
    num_sentences = max(len(sents) for sents in sentence_lists)
    for sents in sentence_lists:
        while len(sents) < num_sentences:
            sents.append("")  # Pad with empty sentence
    
    weights = compute_weights(errors)
    fused_sentences = []

    for i in range(num_sentences):
        # Extract tokenized sentences
        versions = []
        for sent_list in sentence_lists:
            sent = sent_list[i].strip()
            tokens = sent.split() if sent else []
            versions.append(tokens)

        # Select longest sentence as base
        base = max(versions, key=len, default=[])
        if not base:
            fused_sentences.append("")
            continue

        # Initialize alignment and operation matrices
        alignment_matrix = [defaultdict(float) for _ in range(len(base))]
        operation_matrix = [defaultdict(float) for _ in range(len(base))]

        for j, tokens in enumerate(versions):
            if not tokens:
                # Treat empty sentence as all deletions
                for idx in range(len(base)):
                    operation_matrix[idx]["delete"] += weights[j] * 0.5
                    alignment_matrix[idx][""] += weights[j] * 0.5
                continue

            # Align using SequenceMatcher
            matcher = SequenceMatcher(None, base, tokens)
            aligned = [None] * len(base)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                vote_weight = weights[j]  # Use model weight
                if tag == 'equal':
                    for i, j_idx in zip(range(i1, i2), range(j1, j2)):
                        alignment_matrix[i][tokens[j_idx]] += vote_weight
                        operation_matrix[i]["equal"] += vote_weight
                        aligned[i] = True
                elif tag == 'replace':
                    for i, j_idx in zip(range(i1, i2), range(j1, j2)):
                        pos = min(i, len(alignment_matrix) - 1)
                        alignment_matrix[pos][tokens[j_idx]] += vote_weight
                        operation_matrix[pos]["replace"] += vote_weight
                        aligned[pos] = True
                elif tag == 'insert':
                    for offset, j_idx in enumerate(range(j1, j2)):
                        pos = min(i1 + offset, len(alignment_matrix) - 1)
                        alignment_matrix[pos][tokens[j_idx]] += vote_weight
                        operation_matrix[pos]["insert"] += vote_weight
                        aligned[pos] = True
                elif tag == 'delete':
                    for i in range(i1, i2):
                        operation_matrix[i]["delete"] += vote_weight
                        alignment_matrix[i][""] += vote_weight
                        aligned[i] = True

            # Vote for deletions for unaligned positions
            for idx, flag in enumerate(aligned):
                if flag is None:
                    operation_matrix[idx]["delete"] += weights[j] * 0.5
                    alignment_matrix[idx][""] += weights[j] * 0.5

        # Fuse tokens
        fused = []
        for idx, (op_votes, word_votes) in enumerate(zip(operation_matrix, alignment_matrix)):
            if op_votes:
                best_op = max(op_votes.items(), key=lambda x: x[1])[0]
                total_weight = sum(op_votes.values())
                if best_op == "delete" and op_votes["delete"] > 0.6 * total_weight:
                    continue  # Skip token if deletion strongly supported
                elif best_op in ["equal", "replace", "insert"]:
                    if word_votes:
                        # Check if weights are close (within 10%)
                        valid_words = {k: v for k, v in word_votes.items() if k != ""}
                        if valid_words:
                            max_weight = max(valid_words.values())
                            close_weights = [k for k, v in valid_words.items() if v >= 0.9 * max_weight]
                            if len(close_weights) > 1:
                                # Fallback to majority voting
                                word_counts = defaultdict(int)
                                for j, tokens in enumerate(versions):
                                    if idx < len(tokens):
                                        word_counts[tokens[idx]] += 1
                                best_word = max(word_counts.items(), key=lambda x: x[1])[0]
                            else:
                                # Use highest-weighted word
                                best_word = max(valid_words.items(), key=lambda x: x[1])[0]
                            fused.append(best_word)
        fused_sentences.append(' '.join(fused))

    return fused_sentences