
import re
from typing import Union, List
import unicodedata as ud

import torch
from arabert import preprocess
import pyarabic.araby as araby

from RDIreplacement.substitute import substitute

SUBSTITUTE_OBJ = substitute()
ARABERT_OBJ = preprocess.ArabertPreprocessor(model_name="bert-base-arabertv2", apply_farasa_segmentation=True, insert_white_spaces=False)

CHAR_MAPPING = {
    'أ': 'ا',  # Replace أ with ا
    'إ': 'ا',
    'آ': 'ا'
    # Add more character mappings here as needed
    # 'إ': 'ا',  # Example: Replace إ with ا
    # 'آ': 'ا',  # Example: Replace آ with ا
    }

WORD_MAPPING = {
    'يئول': 'يقول',  # Example mapping
    'آل': 'قال'
}

WORDS_TO_BE_REMOVED = [
    'ال',  # Remove standalone ال
    'و'
]

class ArabicTextProcessor:
    """
    A comprehensive Arabic text processor that handles cleaning, normalization,
    and custom text replacements for Arabic text processing tasks.
    All functions operate on lists for optimal bulk processing.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def remove_punctuation(texts: List[str]) -> List[str]:
        """
        Remove punctuation from Arabic texts using Unicode categories.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with punctuation removed
        """
        return [''.join(char for char in text if not ud.category(char).startswith('P')) for text in texts]

    @staticmethod
    def normalize_final_letters(texts: List[str]) -> List[str]:
        """
        Normalize final letters in Arabic words to handle common spelling variations.
        - ى (alif maksura) at word end -> ي (ya)
        - ة (ta marbuta) at word end -> ه (ha)
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with normalized final letters
        """
        def normalize_single_text(text: str) -> str:
            words = text.split()
            normalized_words = []
            
            for word in words:
                # Check if word ends with ى and replace with ي
                if word.endswith('ى'):
                    word = word[:-1] + 'ي'
                
                # Check if word ends with ة and replace with ه
                elif word.endswith('ة'):
                    word = word[:-1] + 'ه'
                
                normalized_words.append(word)
            
            return ' '.join(normalized_words)
        
        return [normalize_single_text(text) for text in texts]
    
    @staticmethod
    def apply_character_mappings(texts: List[str]) -> List[str]:
        """
        Apply character-level mappings to the texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with character mappings applied
        """
        def apply_mappings_single(text: str) -> str:
            for original, replacement in CHAR_MAPPING.items():
                text = text.replace(original, replacement)
            return text
        
        return [apply_mappings_single(text) for text in texts]
    
    @staticmethod
    def apply_word_mappings(texts: List[str]) -> List[str]:
        """
        Apply word-level mappings to the texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with word mappings applied
        """
        def apply_mappings_single(text: str) -> str:
            words = text.split()
            mapped_words = []
            
            for word in words:
                # Check if word needs to be replaced
                if word in WORD_MAPPING:
                    mapped_words.append(WORD_MAPPING[word])
                else:
                    mapped_words.append(word)
            
            return ' '.join(mapped_words)
        
        return [apply_mappings_single(text) for text in texts]
    
    @staticmethod
    def remove_unwanted_words(texts: List[str]) -> List[str]:
        """
        Remove unwanted standalone words from the texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with unwanted words removed
        """
        def remove_words_single(text: str) -> str:
            words = text.split()
            filtered_words = [word for word in words if word not in WORDS_TO_BE_REMOVED]
            return ' '.join(filtered_words)
        
        return [remove_words_single(text) for text in texts]

    @staticmethod
    def remove_tashkeel_tatweel(texts: List[str]) -> List[str]:
        def remove_tashkeel_tatweel_single(text: str) -> str:
            text = araby.strip_tashkeel(text)
            text = araby.strip_tatweel(text)
            return text
        
        return [remove_tashkeel_tatweel_single(text) for text in texts]

    @staticmethod
    def process_arabic_text_arabert(texts: List[str]) -> List[str]:
        """
        Clean Arabic texts using AraBERT preprocessor and custom cleaning.
        
        Args:
            texts (List[str]): List of input Arabic texts to clean
            
        Returns:
            List[str]: List of cleaned Arabic texts in the same order
        """
        def clean_preprocessed_text(texts: List[str]) -> List[str]:
            """
            Clean preprocessed texts by removing quotes and extra whitespace.
            
            Args:
                texts (List[str]): List of preprocessed texts
                
            Returns:
                List[str]: List of cleaned texts
            """
            def clean_single(text: str) -> str:
                cleaned_text = text.strip("'")
                cleaned_text = cleaned_text.strip()
                return cleaned_text
            
            return [clean_single(text) for text in texts]
        
        def remove_clitics_with_plus_simple(texts: List[str]) -> List[str]:
            """
            Simple version: Remove any word containing a + sign completely.
            
            Args:
                texts (List[str]): List of preprocessed Arabic texts with clitics marked by + signs
                
            Returns:
                List[str]: List of texts with all words containing + removed
            """
            def remove_clitics_single(text: str) -> str:
                words = text.split()
                cleaned_words = [word for word in words if '+' not in word]
                return ' '.join(cleaned_words)
            
            return [remove_clitics_single(text) for text in texts]
        
        cleaned_texts = [ARABERT_OBJ.preprocess(text) for text in texts]
        cleaned_texts = remove_clitics_with_plus_simple(cleaned_texts)
        cleaned_texts = clean_preprocessed_text(cleaned_texts)
        return cleaned_texts
    
    @staticmethod
    def substitute_words(texts: List[str]) -> List[str]:
        replaced_texts = SUBSTITUTE_OBJ(text=texts, is_corpus=True, is_asmo=False)
        if not isinstance(replaced_texts, list):
            replaced_texts = [replaced_texts] if isinstance(replaced_texts, str) else list(replaced_texts)
        return replaced_texts
    
    @staticmethod
    def substitute_and_arabert(texts: List[str]) -> List[str]:
        """
        Normalize texts using the substitute object and clean them.
        
        Args:
            texts (List[str]): List of input texts to normalize
            
        Returns:
            List[str]: List of normalized and cleaned texts in the same order
        """
        replaced_texts = ArabicTextProcessor.substitute_words(texts)        
        cleaned_texts = ArabicTextProcessor.process_arabic_text_arabert(replaced_texts)
        return cleaned_texts
    
    @staticmethod
    def process_texts(texts: Union[str, List[str]], normalize_final_letters: bool = True) -> Union[str, List[str]]:
        """
        Process text(s) through the complete Arabic text processing pipeline.
        All processing is done in bulk mode for optimal performance.
        
        Args:
            texts (str or List[str]): Input text(s) to process
            normalize_final_letters (bool): Whether to normalize final letters (ى↔ي, ة↔ه)
            
        Returns:
            str or List[str]: Processed text(s) - same type as input
        """
        # Convert single string to list for uniform processing
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]
        elif isinstance(texts, list):
            pass  # Already a list
        else:
            raise TypeError("Input must be either a string or a list of strings")
        
        # Step 0: Remove punctuation from all texts
        processed_texts = ArabicTextProcessor.remove_punctuation(texts)
        
        # Step 1: Normalize texts (includes substitution and basic cleaning)
        processed_texts = ArabicTextProcessor.substitute_and_arabert(processed_texts)
        
        # Step 2: Apply word-level mappings (like يئول -> يقول)
        processed_texts = ArabicTextProcessor.apply_word_mappings(processed_texts)
        
        # Step 3: Apply character-level mappings (like أ -> ا)
        processed_texts = ArabicTextProcessor.apply_character_mappings(processed_texts)
        
        # Step 4: Normalize final letters if requested
        if normalize_final_letters:
            processed_texts = ArabicTextProcessor.normalize_final_letters(processed_texts)
        
        # Step 5: Remove unwanted words (like standalone ال)
        processed_texts = ArabicTextProcessor.remove_unwanted_words(processed_texts)
        
        # Step 6: Clean up any extra whitespace
        processed_texts = [' '.join(text.split()) for text in processed_texts]
        
        # Return single string if input was single string, otherwise return list
        return processed_texts[0] if is_single_input else processed_texts
    
    @staticmethod
    def normalize_texts(texts: Union[str, List[str]]):
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]
        elif isinstance(texts, list):
            pass  # Already a list
        else:
            raise TypeError("Input must be either a string or a list of strings")
        
        processed_texts = ArabicTextProcessor.remove_punctuation(texts)
        processed_texts = ArabicTextProcessor.remove_tashkeel_tatweel(processed_texts)
        replaced_texts = ArabicTextProcessor.substitute_words(processed_texts)
        processed_texts = ArabicTextProcessor.apply_word_mappings(replaced_texts)
        processed_texts = ArabicTextProcessor.apply_character_mappings(processed_texts)
        processed_texts = ArabicTextProcessor.normalize_final_letters(processed_texts)
        processed_texts = [' '.join(text.split()) for text in processed_texts]
        return processed_texts[0] if is_single_input else processed_texts