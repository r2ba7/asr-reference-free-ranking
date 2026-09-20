import os

import torchaudio
import pandas as pd
import soundfile as sf

from stt_benchmarking.utils import text_processing
from stt_benchmarking.utils.english_normalizer import normalizer
from . import LOGGER

def load_audio_transcripts_analysis(data_dir, sep, metadata_file_name="metadata.txt", **kwargs):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    metadata_path = os.path.join(data_dir, metadata_file_name)
    waves_dir = os.path.join(data_dir, "waves")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'.")

    if not os.path.isdir(waves_dir):
        raise FileNotFoundError(f"Waves directory not found at '{waves_dir}'.")

    samples = []
    transcriptions = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if sep == "|":
                parts = line.strip().split("|", 1)
                audio_filename, transcription = parts
            elif sep == " ":
                parts = line.strip().split(" ", 1)
                audio_filename, transcription = parts
                if "wav" not in audio_filename:
                    audio_filename = audio_filename + ".wav"

            audio_path = f"{waves_dir}/{audio_filename}"
            if os.path.exists(audio_path):
                info = sf.info(audio_path)
                sample_rate = info.samplerate
                duration_sec = info.frames / sample_rate
                samples.append({
                    "audio_path": audio_path,
                    "sample_rate": sample_rate,
                    "transcription": transcription,
                    "audio_duration": duration_sec
                })
                transcriptions.append(transcription)
            else:
                print(f"Path {audio_path}, doesn't exist")

    if transcriptions:
        substitute = kwargs.get("substitute", False)
        normalize_final_letters = kwargs.get("normalize_final_letters", True)
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        normalized_transcriptions = text_processing.StandardArabicTextProcessor.main(texts=transcriptions, normalize_final_letters=normalize_final_letters, substitute=substitute)
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]

    return samples

def load_audio_transcripts(data_dir, sep, metadata_file_name="metadata.txt", **kwargs):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    metadata_path = os.path.join(data_dir, metadata_file_name)
    waves_dir = os.path.join(data_dir, "waves")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'.")

    if not os.path.isdir(waves_dir):
        raise FileNotFoundError(f"Waves directory not found at '{waves_dir}'.")
    
    samples = []
    transcriptions = []  # Collect all transcriptions for bulk processing
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if sep == "|":
                parts = line.strip().split("|", 1)
                audio_filename, transcription = parts
            elif sep == " ":
                parts = line.strip().split(" ", 1)
                audio_filename, transcription = parts
                if "wav" not in audio_filename:
                    audio_filename = audio_filename + ".wav"
            
            audio_path = f"{waves_dir}/{audio_filename}"
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = waveform.squeeze(0)

                # 🔹 Resample if not 16k
                target_sr = 16000
                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                    waveform = resampler(waveform)
                    sample_rate = target_sr

                duration_sec = waveform.shape[-1] / sample_rate
                samples.append({
                    "audio_path": audio_path,
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "transcription": transcription,
                    "audio_duration": duration_sec
                })
                
                # Collect transcription for bulk normalization
                transcriptions.append(transcription)
            else:
                print(f"Path {audio_path}, doesn't exist")

    # Bulk normalize all transcriptions at once
    if transcriptions:
        substitute = kwargs.get("substitute", False)
        normalize_final_letters = kwargs.get("normalize_final_letters", True)
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        normalized_transcriptions = text_processing.StandardArabicTextProcessor.main(texts=transcriptions, normalize_final_letters=normalize_final_letters, substitute=substitute)
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]

    return samples


def load_mozilla_cv_analysis(df, audio_path, **kwargs):
    required_cols = ["path", "sentence", "age", "gender", "accents", "locale", "segment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    paths = df["path"].values
    sentences = df["sentence"].values
    ages = df["age"].values
    genders = df["gender"].values
    accents = df["accents"].values
    locales = df["locale"].values
    segments = df["segment"].values

    full_paths = [os.path.join(audio_path, p) for p in paths]
    valid_mask = [os.path.exists(p) for p in full_paths]
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    if len(valid_indices) < len(full_paths):
        print(f"Skipped {len(full_paths) - len(valid_indices)} missing files")

    samples = []
    transcriptions = []

    for idx in valid_indices:
        full_audio_path = full_paths[idx]
        transcription = str(sentences[idx]).strip()

        info = sf.info(full_audio_path)
        sample_rate = info.samplerate
        duration_sec = info.frames / sample_rate

        samples.append({
            "audio_path": full_audio_path,
            "sample_rate": sample_rate,
            "transcription": transcription,
            "audio_duration": duration_sec,
            "age": ages[idx],
            "gender": genders[idx],
            "accent": accents[idx],
            "locale": locales[idx],
            "segment": segments[idx]
        })
        transcriptions.append(transcription)

    if transcriptions:
        substitute = kwargs.get("substitute", False)
        normalize_final_letters = kwargs.get("normalize_final_letters", True)
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        normalized_transcriptions = text_processing.StandardArabicTextProcessor.main(
            texts=transcriptions,
            normalize_final_letters=normalize_final_letters,
            substitute=substitute
        )
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]

    return samples


def load_mozilla_cv(df, audio_path, **kwargs):
    required_cols = ["path", "sentence", "age", "gender", "accents", "locale", "segment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Pre-extract series to avoid repeated dictionary lookups
    paths = df["path"].values
    sentences = df["sentence"].values
    ages = df["age"].values
    genders = df["gender"].values
    accents = df["accents"].values
    locales = df["locale"].values
    segments = df["segment"].values
    
    # Pre-compute all full paths and filter valid ones upfront
    full_paths = [os.path.join(audio_path, p) for p in paths]
    valid_mask = [os.path.exists(p) for p in full_paths]
    
    # Filter everything by valid mask before processing
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    if len(valid_indices) < len(full_paths):
        print(f"Skipped {len(full_paths) - len(valid_indices)} missing files")
    
    samples = []
    transcriptions = []
    target_sr = 16000
    resampler = None  # Create once if needed
    
    for idx in valid_indices:
        full_audio_path = full_paths[idx]
        transcription = str(sentences[idx]).strip()
        
        waveform, sample_rate = torchaudio.load(full_audio_path)
        waveform = waveform.squeeze(0)
        
        if sample_rate != target_sr:
            if resampler is None or resampler.orig_freq != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        
        duration_sec = waveform.shape[-1] / sample_rate
        samples.append({
            "audio_path": full_audio_path,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "transcription": transcription,
            "audio_duration": duration_sec,
            "age": ages[idx],
            "gender": genders[idx],
            "accent": accents[idx],
            "locale": locales[idx],
            "segment": segments[idx]
        })
        transcriptions.append(transcription)
    
    if transcriptions:
        substitute = kwargs.get("substitute", False)
        normalize_final_letters = kwargs.get("normalize_final_letters", True)
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        normalized_transcriptions = text_processing.StandardArabicTextProcessor.main(
            texts=transcriptions,
            normalize_final_letters=normalize_final_letters,
            substitute=substitute
        )
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]
    
    return samples

def load_reddit_audios(json_path="../../data/scrapped_data/social_media/reddit_df_full.json", 
                       audio_dir="../../data/scrapped_data/social_media/waves/clean", 
                       **kwargs):
    """
    Load Reddit audio files and metadata from JSON.
    
    Args:
        json_path: path to metadata JSON file
        audio_dir: directory containing audio files
        **kwargs: normalize_final_letters, substitute for text processing
        
    Returns:
        list of sample dicts with audio and metadata
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        df = pd.read_json(f, lines=True)
    
    # Pre-extract series
    indices = df.index.values
    texts = df["text"].values
    
    # Pre-compute paths and check validity
    full_paths = [os.path.join(audio_dir, f"{idx}.wav") for idx in indices]
    valid_mask = [os.path.exists(p) for p in full_paths]
    
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    if len(valid_indices) < len(full_paths):
        LOGGER.warning(f"Skipped {len(full_paths) - len(valid_indices)} missing audio files")
    
    samples = []
    transcriptions = []
    target_sr = 16000
    resampler = None
    
    for i in valid_indices:
        full_audio_path = full_paths[i]
        transcription = str(texts[i]).strip()
        
        waveform, sample_rate = torchaudio.load(full_audio_path)
        waveform = waveform.squeeze(0)
        
        if sample_rate != target_sr:
            if resampler is None or resampler.orig_freq != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        
        duration_sec = waveform.shape[-1] / sample_rate
        samples.append({
            "audio_path": full_audio_path,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "transcription": transcription,
            "audio_duration": duration_sec,
            "index": indices[i]
        })
        transcriptions.append(transcription)
    
    if transcriptions:
        substitute = kwargs.get("substitute", False)
        normalize_final_letters = kwargs.get("normalize_final_letters", True)
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
        normalized_transcriptions = text_processing.StandardArabicTextProcessor.main(
            texts=transcriptions,
            normalize_final_letters=normalize_final_letters,
            substitute=substitute
        )
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]
    
    return samples