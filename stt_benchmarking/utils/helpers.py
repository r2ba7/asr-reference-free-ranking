import os
import random
from IPython.display import Audio
import gc

import torch
import torchaudio

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
                waveform = waveform.squeeze().numpy()
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