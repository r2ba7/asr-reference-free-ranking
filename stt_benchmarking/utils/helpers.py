import os
import random
from IPython.display import Audio
import gc
import re

import torch
import torchaudio

import random
from collections import defaultdict
import numpy as np
import json

def save_records_to_json(records, path="sampled_records.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

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

def load_audio_transcripts(data_dir, is_egy):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    metadata_path = os.path.join(data_dir, "metadata.txt")
    waves_dir = os.path.join(data_dir, "waves")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'.")

    if not os.path.isdir(waves_dir):
        raise FileNotFoundError(f"Waves directory not found at '{waves_dir}'.")
    
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if is_egy:
                parts = line.strip().split(" ", 1)
                audio_filename, transcription = parts
                audio_filename = audio_filename + ".wav"
                audio_source = "Egypt"
            else:
                parts = line.strip().split("|", 1)
                audio_filename, transcription = parts
                audio_source = "Saudi"
            
            audio_path = os.path.join(waves_dir, audio_filename)  
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = waveform.squeeze(0)
                duration_sec = waveform.shape[-1] / sample_rate
                samples.append({
                    "audio_path": audio_path,
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "transcription": transcription,
                    "audio_source": audio_source,
                    "audio_duration": duration_sec
                })
            else:
                print(f"Path {audio_path}, doesn't exist")

    return samples

def play_sample(sample):
    print(sample['transcription'])
    return Audio(data=sample['waveform'], rate=sample['sample_rate'])

def pick_random_records(records, no_samples):
    if no_samples > len(records):
        raise ValueError("Requested more samples than available in the records.")
    
    return random.sample(records, no_samples)

def clean_arabic_text(text: str) -> str:
    """
    Cleans Arabic text by:
    - Removing Arabic diacritics
    - Stripping redundant whitespace
    - Returning a clean string
    
    Args:
        text (str): Input Arabic text to clean
        
    Returns:
        str: Cleaned Arabic text
    """
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text