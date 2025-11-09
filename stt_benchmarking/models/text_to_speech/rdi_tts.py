import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import base64
import os
import soundfile as sf
import librosa
from scipy import signal
from tqdm import tqdm
import numpy as np

from .. import LOGGER
from stt_benchmarking.utils import (
    decorators, 
)

def save_audio(wav_data, filename):
    """ Save decoded audio data to a file. """
    with open(filename, "wb") as file:
        file.write(wav_data)

def load_noise_files(noise_dir="../../data/noise_datasets"):
    """
    Load noise files organized by folder.
    
    Returns:
        dict: Mapping {folder_path: [list_of_files]}
    """
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg']
    folder_files_map = {}

    if not os.path.exists(noise_dir):
        LOGGER.warning(f"Noise directory {noise_dir} not found")
        return {}

    for entry in os.scandir(noise_dir):
        if entry.is_dir():
            files = [
                os.path.join(entry.path, f)
                for f in os.listdir(entry.path)
                if any(f.lower().endswith(ext) for ext in supported_formats)
            ]
            if files:
                folder_files_map[entry.path] = files

    LOGGER.info(f"Loaded {sum(len(v) for v in folder_files_map.values())} noise files "
                f"from {len(folder_files_map)} folders")
    return folder_files_map


def add_noise_to_audio(audio_data, sample_rate, folder_files_map, snr_range=(5, 30)):
    """
    Add noise to audio. Picks one random folder from 16, then one random file from that folder.
    SNR is chosen randomly from continuous range.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of audio
        folder_files_map: dict {folder: [file paths]}
        snr_range: tuple (min_snr, max_snr) in dB
    
    Returns:
        noisy_audio: numpy array of noisy audio samples
    """
    if not folder_files_map:
        LOGGER.error("No noise files available, returning original audio")
        return audio_data

    chosen_folder = random.choice(list(folder_files_map.keys()))
    noise_file = random.choice(folder_files_map[chosen_folder])

    try:
        noise, noise_sr = librosa.load(noise_file, sr=None)
        if noise_sr != sample_rate:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sample_rate)

        audio_len = len(audio_data)
        noise_len = len(noise)

        if noise_len < audio_len:
            repeats = (audio_len // noise_len) + 1
            noise = np.tile(noise, repeats)[:audio_len]
        else:
            start_idx = random.randint(0, noise_len - audio_len)
            noise = noise[start_idx:start_idx + audio_len]

        signal_power = np.mean(audio_data ** 2)
        noise_power = np.mean(noise ** 2)

        snr_db = random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10 ** (snr_db / 10)

        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        noisy_audio = audio_data + noise_scale * noise

        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0.95:
            noisy_audio = noisy_audio * 0.95 / max_val

        return noisy_audio

    except Exception as e:
        LOGGER.error(f"Error adding noise from {noise_file}: {e}")
        return audio_data

def load_noise_files(noise_dir="../../data/noise_datasets"):
    """
    Load noise files with equal probability distribution across folders and files.
    - First, randomly pick a folder (uniformly across folders).
    - Then, randomly pick a file from that folder (uniformly within the folder).
    
    Returns:
        dict: Mapping {folder_path: [list_of_files]} for controlled sampling.
    """
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg']
    folder_files_map = {}

    if not os.path.exists(noise_dir):
        LOGGER.warning(f"Warning: Noise directory {noise_dir} not found!")
        return {}

    # Only consider directories, skip .rar or other files
    for entry in os.scandir(noise_dir):
        if entry.is_dir():  # ✅ only directories
            files = [
                os.path.join(entry.path, f)
                for f in os.listdir(entry.path)
                if any(f.lower().endswith(ext) for ext in supported_formats)
            ]
            if files:  # Only keep non-empty folders
                folder_files_map[entry.path] = files

    LOGGER.info(f"Loaded {sum(len(v) for v in folder_files_map.values())} noise files "
          f"from {len(folder_files_map)} folders in {noise_dir}")
    return folder_files_map

def add_noise_to_audio(audio_data, sample_rate, folder_files_map, snr_choices=(5, 10, 20)):
    """
    Add noise to audio with equal probability across folders and files.
    SNR is chosen randomly from a fixed set (default: 5, 10, 20 dB).
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of audio
        folder_files_map: dict {folder: [file paths]}
        snr_choices: tuple or list of allowed SNR values in dB
    
    Returns:
        noisy_audio: numpy array of noisy audio samples
    """
    if not folder_files_map:
        LOGGER.error("Warning: No noise files available, returning original audio")
        return audio_data

    # Step 1: Pick a folder uniformly
    chosen_folder = random.choice(list(folder_files_map.keys()))
    
    # Step 2: Pick a file uniformly from that folder
    noise_file = random.choice(folder_files_map[chosen_folder])

    try:
        # Load noise
        noise, noise_sr = librosa.load(noise_file, sr=None)
        if noise_sr != sample_rate:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sample_rate)

        # Match lengths
        audio_len = len(audio_data)
        noise_len = len(noise)

        if noise_len < audio_len:
            repeats = (audio_len // noise_len) + 1
            noise = np.tile(noise, repeats)[:audio_len]
        else:
            start_idx = random.randint(0, noise_len - audio_len)
            noise = noise[start_idx:start_idx + audio_len]

        # Power calculation
        signal_power = np.mean(audio_data ** 2)
        noise_power = np.mean(noise ** 2)

        # Pick discrete SNR
        snr_db = random.choice(snr_choices)
        snr_linear = 10 ** (snr_db / 10)

        # Scale noise
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        noisy_audio = audio_data + noise_scale * noise

        # Prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0.95:
            noisy_audio = noisy_audio * 0.95 / max_val

        return noisy_audio

    except Exception as e:
        LOGGER.error(f"Error adding noise from {noise_file}: {e}")
        return audio_data

class RDI_TTS_Inference:
    NOISE_DATASET_DIR = "../../data/noise_datasets"
    URL = "http://34.57.97.217:6017/speak"
    DATA = {
        "format": "json", 
        "lang": "AR", 
        "speakers": ["saudi"], 
        "keep_original_tashkeel": True, 
        "auto_tashkeel": True, 
        "speed": 1, 
        "output_format": "wav"
    }

    def __init__(self, CLEAN_OUTPUT_DIR, NOISY_OUTPUT_DIR):
        self.CLEAN_OUTPUT_DIR = CLEAN_OUTPUT_DIR
        self.NOISY_OUTPUT_DIR = NOISY_OUTPUT_DIR
        self.folder_files_map = load_noise_files(self.NOISE_DATASET_DIR)

    @decorators.Decorators.timeout_with_retry
    def process_single_file(self, text, index):
        payload = self.DATA.copy()
        payload["text"] = text
        
        response = requests.post(self.URL, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        wave_data = base64.urlsafe_b64decode(data['wave'])
        
        filepath = os.path.join(self.CLEAN_OUTPUT_DIR, f"{index}.wav")
        save_audio(wav_data=wave_data, filename=filepath)
        
        return filepath

    def process_noisy_file(self, index):
        """
        Load clean audio, apply noise, save noisy version.
        
        Args:
            index: file index
            
        Returns:
            path to noisy audio file or None if failed
        """
        clean_filepath = os.path.join(self.CLEAN_OUTPUT_DIR, f"{index}.wav")
        if not os.path.exists(clean_filepath):
            LOGGER.error(f"Clean file not found for index {index}")
            return None
        
        try:
            audio_data, sample_rate = sf.read(clean_filepath)
            noisy_audio = add_noise_to_audio(audio_data, sample_rate, self.folder_files_map)
            noisy_filepath = os.path.join(self.NOISY_OUTPUT_DIR, f"{index}.wav")
            sf.write(noisy_filepath, noisy_audio, sample_rate)
            return noisy_filepath
        
        except Exception as e:
            LOGGER.error(f"Failed to create noisy version for index {index}: {e}")
            return None

    @decorators.Decorators.calculate_execution_time
    def run_inference(self, df):
        """
        Generate clean audio files from dataframe.
        Uses dataframe index as filename.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            list of generated clean audio file paths
        """        
        results = {}
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.process_single_file, row['text'], idx): idx 
                for idx, row in df.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating clean audio"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    LOGGER.error(f"Failed index {idx}: {e}")
                    results[idx] = None
        
        paths = [results.get(idx) for idx in df.index]
        success_count = sum(1 for p in paths if p is not None)
        LOGGER.info(f"Generated {success_count}/{len(df)} clean files")
        return paths

    @decorators.Decorators.calculate_execution_time
    def run_noisy_inference(self, df):
        """
        Load clean audio files and generate noisy versions.
        Uses dataframe index to locate clean files.
        
        Args:
            df: DataFrame (uses index to find corresponding clean files)
            
        Returns:
            list of generated noisy audio file paths
        """        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_noisy_file, idx): idx
                for idx in df.index
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating noisy audio"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    LOGGER.error(f"Failed noisy for index {idx}: {e}")
                    results[idx] = None
        
        paths = [results.get(idx) for idx in df.index]
        success_count = sum(1 for p in paths if p is not None)
        
        LOGGER.info(f"Generated {success_count}/{len(df)} noisy files")
        return paths