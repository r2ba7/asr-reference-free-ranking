import os

import torchaudio

from stt_benchmarking.utils import text_processing

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
    transcriptions = []  # Collect all transcriptions for bulk processing
    
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
                
                # Collect transcription for bulk normalization
                transcriptions.append(transcription)
            else:
                print(f"Path {audio_path}, doesn't exist")

    # Bulk normalize all transcriptions at once
    if transcriptions:
        normalized_transcriptions = text_processing.ArabicTextProcessor.process_texts(transcriptions)
        # Reinsert normalized transcriptions back into samples
        for i, sample in enumerate(samples):
            sample["normalized_transcription"] = normalized_transcriptions[i]

    return samples