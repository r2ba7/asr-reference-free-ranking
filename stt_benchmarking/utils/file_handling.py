import json

from IPython.display import Audio

def save_records_to_json(records, path="sampled_records.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def play_sample(sample):
    print(sample['transcription'])
    return Audio(data=sample['waveform'], rate=sample['sample_rate'])