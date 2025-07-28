import re
import gc

import torch

from RDIreplacement.substitute import substitute

SUB_OBJ = substitute()

def clean_arabic_text(texts):
    """
    Cleans Arabic text by:
    - Removing Arabic diacritics
    - Removing punctuation marks (Arabic and English)
    - Removing special symbols and grammar marks
    - Stripping redundant whitespace
    - Returning a clean string or list of clean strings
    
    Args:
        texts (str or list): Input Arabic text(s) to clean
        
    Returns:
        str or list: Cleaned Arabic text(s) - same type as input
    """
    def clean_single_text(text):
        # Remove Arabic diacritics
        text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]", "", text)
        # Remove Arabic punctuation marks
        text = re.sub(r"[\u060C\u061B\u061F\u0640\u066A\u066B\u066C\u066D]", "", text)
        # Remove English punctuation marks
        text = re.sub(r"[.,;:!?\"'`~@#$%^&*()_+=\[\]{}<>/\\|]", "", text)
        # Remove numbers (Arabic and English)
        text = re.sub(r"[\u0660-\u0669\d]", "", text)
        # Remove additional Arabic symbols and marks
        text = re.sub(r"[\u06DD\u06DE\u06DF\u06E5\u06E6\u06E9\u06FD\u06FE]", "", text)
        # Remove hyphens and dashes
        text = re.sub(r"[-–—_]", "", text)
        # Strip redundant whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    if isinstance(texts, str):
        return clean_single_text(texts)
    elif isinstance(texts, list):
        return [clean_single_text(text) for text in texts]
    else:
        raise TypeError("Input must be either a string or a list of strings")

def normalize_text(texts):
    cleaned_texts = clean_arabic_text(texts)
    replaced_text = SUB_OBJ(text=cleaned_texts, is_corpus=False, is_asmo=False) 
    return replaced_text

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