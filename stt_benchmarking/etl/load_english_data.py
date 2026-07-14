import os

import torchaudio
import pandas as pd
import soundfile as sf

from stt_benchmarking.utils.english_normalizer import normalizer

def load_librispeech_analysis(data_dir, **kwargs):
    eng_normalizer = normalizer.EnglishTextNormalizer()

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    samples = []
    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            trans_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if not os.path.isfile(trans_path):
                continue

            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcription = parts
                    audio_file = os.path.join(chapter_path, f"{utt_id}.flac")

                    if not os.path.exists(audio_file):
                        print(f"Missing audio file: {audio_file}")
                        continue

                    info = sf.info(audio_file)
                    sample_rate = info.samplerate
                    duration_sec = info.frames / sample_rate

                    samples.append({
                        "audio_path": audio_file,
                        "sample_rate": sample_rate,
                        "transcription": transcription,
                        "audio_duration": duration_sec,
                        "normalized_transcription": eng_normalizer(transcription)
                    })
    return samples


def load_librispeech(data_dir, target_sr=16000, **kwargs):
    """
    Load LibriSpeech test (or any) subset structure directly from extracted directories.

    Args:
        data_dir (str): Path to the root of the subset (e.g., 'LibriSpeech/test-clean')
        target_sr (int): Desired sampling rate (default 16k)
        **kwargs: Optional normalization arguments

    Returns:
        list[dict]: [{'audio_path', 'waveform', 'sample_rate', 'transcription', 'audio_duration'}]
    """
    eng_normalizer = normalizer.EnglishTextNormalizer()
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    samples = []
    # Traverse speaker directories recursively
    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            trans_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if not os.path.isfile(trans_path):
                continue

            # Read all utterance transcriptions
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcription = parts
                    audio_file = os.path.join(chapter_path, f"{utt_id}.flac")

                    if not os.path.exists(audio_file):
                        print(f"Missing audio file: {audio_file}")
                        continue

                    waveform, sample_rate = torchaudio.load(audio_file)
                    waveform = waveform.squeeze(0)

                    # Resample if needed
                    if sample_rate != target_sr:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=target_sr
                        )
                        waveform = resampler(waveform)
                        sample_rate = target_sr

                    duration_sec = waveform.shape[-1] / sample_rate

                    samples.append({
                        "audio_path": audio_file,
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "transcription": transcription,
                        "audio_duration": duration_sec,
                        "normalized_transcription": eng_normalizer(transcription)
                    })
    return samples

