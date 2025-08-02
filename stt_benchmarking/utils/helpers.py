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


