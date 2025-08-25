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

from . import LOGGER
from stt_benchmarking.utils import (
    decorators, 
)

def read_text_file(file_path):
    """ Read the first line from the text file. """
    with open(file_path, "r") as file:
        return file.readline().strip()

def save_audio(wav_data, filename):
    """ Save decoded audio data to a file. """
    with open(filename, "wb") as file:
        file.write(wav_data)

def generate_filename_with_index(index, output_dir="../data/synthetic_speech_records", suffix=""):
    """ Generate a filename with index and optional suffix. """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{index:06d}{suffix}.wav"  # 6-digit zero-padded index with suffix
    return os.path.join(output_dir, filename)

def save_metadata(text_audio_mapping, output_dir="../data/synthetic_speech_records", filename="metadata.txt"):
    """ Save text-audio mapping as metadata file. """
    metadata_path = os.path.join(output_dir, filename)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for audio_path, text in text_audio_mapping.items():
            filename = os.path.basename(audio_path)
            f.write(f"{filename} {text}\n")
    return metadata_path

def load_noise_files(noise_dir="../data/noise_datasets"):
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

    URL = "http://34.57.97.217:6018/speak"
    DATA = {"format": "json", "lang": "ar", "keep_original_tashkeel": False, "auto_tashkeel": True}
    SPEAKERS = ["male-nabil", "female-eman"]

    def __init__(self, noise_dir=None, create_noisy_version=True):
        self._generated_audio_paths = []
        self._text_audio_mapping = {}
        self._noisy_audio_paths = []
        self._noisy_text_audio_mapping = {}
        
        # Load noise files if directory provided
        self.noise_files = []
        self.create_noisy_version = create_noisy_version
        if noise_dir and create_noisy_version:
            self.noise_files = load_noise_files(noise_dir)

    @decorators.Decorators.timeout_with_retry
    def process_single_file(self, text, index):
        try:
            payload = RDI_TTS_Inference.DATA.copy()
            payload["text"] = text
            payload["speaker"] = random.choice(RDI_TTS_Inference.SPEAKERS)
            response = requests.post(RDI_TTS_Inference.URL, json=payload)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code} - retrying...")
            
            data = response.json()
            wave_base64 = data['wave']
            wave_data = base64.urlsafe_b64decode(wave_base64)
            clean_wav_name = generate_filename_with_index(index, output_dir="../data/clean_synthetic_speech_records/waves", suffix="_clean")
            save_audio(wav_data=wave_data, filename=clean_wav_name)
            
            # ✅ Return both the results AND the original index and text
            results = {
                "clean_path": clean_wav_name, 
                "noisy_path": None,
                "index": index,  # Add original index
                "text": text     # Add original text
            }
            
            if self.create_noisy_version and self.noise_files:
                try:
                    audio_data, sample_rate = sf.read(clean_wav_name)
                    noisy_audio = add_noise_to_audio(audio_data, sample_rate, self.noise_files)
                    noisy_wav_name = generate_filename_with_index(index, output_dir="../data/noisy_synthetic_speech_records/waves", suffix="_noisy")
                    sf.write(noisy_wav_name, noisy_audio, sample_rate)
                    results["noisy_path"] = noisy_wav_name
                    
                except Exception as e:
                    LOGGER.error(f"Error creating noisy version for index {index}: {e}")
            
            return results
               
        except (KeyError, json.JSONDecodeError, base64.binascii.Error) as e:
            LOGGER.error(f"Error processing response: {e}")
            raise e
        except Exception as e:
            raise e
    
    @decorators.Decorators.calculate_execution_time
    def run_inference(self, texts, save_metadata_flag=True):
        """
        Generate audio files from a list of text strings.
        Creates both clean and noisy versions if noise files are available.
        
        Args:
            texts: List of text strings to convert to speech
            save_metadata_flag: Whether to save metadata files
            
        Returns:
            Dict with clean and noisy audio file paths
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Reset mappings
        self._generated_audio_paths = []
        self._text_audio_mapping = {}
        self._noisy_audio_paths = []
        self._noisy_text_audio_mapping = {}
        
        # ✅ Use dictionaries to maintain proper index mapping
        clean_results = {}  # index -> path
        noisy_results = {}  # index -> path
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create future -> (index, text) mapping
            future_to_info = {
                executor.submit(self.process_single_file, text, i): (i, text) 
                for i, text in enumerate(texts)
            }
            
            for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Generating audio files"):
                original_index, original_text = future_to_info[future]
                try:
                    results = future.result()
                    
                    # ✅ Use the original index from the results (double-check)
                    assert results["index"] == original_index, f"Index mismatch: {results['index']} != {original_index}"
                    assert results["text"] == original_text, f"Text mismatch for index {original_index}"
                    
                    # Store clean audio info
                    if results["clean_path"]:
                        clean_results[original_index] = results["clean_path"]
                        self._text_audio_mapping[results["clean_path"]] = original_text
                    
                    # Store noisy audio info
                    if results["noisy_path"]:
                        noisy_results[original_index] = results["noisy_path"]
                        self._noisy_text_audio_mapping[results["noisy_path"]] = original_text
                   
                except Exception as e:
                    LOGGER.error(f"Failed to process text '{original_text[:50]}...' at index {original_index}: {e}")
                    # Keep None values to maintain indexing
                    clean_results[original_index] = None
                    noisy_results[original_index] = None
        
        # ✅ Convert to ordered lists (preserve original order)
        self._generated_audio_paths = [clean_results.get(i) for i in range(len(texts)) if clean_results.get(i) is not None]
        self._noisy_audio_paths = [noisy_results.get(i) for i in range(len(texts)) if noisy_results.get(i) is not None]
        
        # Save metadata files
        if save_metadata_flag:
            if self._text_audio_mapping:
                clean_metadata_path = save_metadata(self._text_audio_mapping, output_dir="../data/clean_synthetic_speech_records", filename="clean_metadata.txt")
                LOGGER.info(f"Clean metadata saved to: {clean_metadata_path}")
            
            if self._noisy_text_audio_mapping:
                noisy_metadata_path = save_metadata(self._noisy_text_audio_mapping, output_dir="../data/noisy_synthetic_speech_records", filename="noisy_metadata.txt")
                LOGGER.info(f"Noisy metadata saved to: {noisy_metadata_path}")
        
        results = {
            "clean_paths": self._generated_audio_paths,
            "noisy_paths": self._noisy_audio_paths,
            "clean_count": len(self._generated_audio_paths),
            "noisy_count": len(self._noisy_audio_paths)
        }
        
        LOGGER.info(f"Successfully generated {results['clean_count']} clean and {results['noisy_count']} noisy audio files")
        return results
    
    @property
    def text_audio_mapping(self):
        """Return mapping for clean audio"""
        return self._text_audio_mapping
    
    @property
    def noisy_text_audio_mapping(self):
        """Return mapping for noisy audio"""
        return self._noisy_text_audio_mapping
    
    @property
    def generated_audio_paths(self):
        """Return list of clean audio file paths"""
        return self._generated_audio_paths
    
    @property
    def noisy_audio_paths(self):
        """Return list of noisy audio file paths"""
        return self._noisy_audio_paths