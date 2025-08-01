
import re
from typing import Union, List
import unicodedata as ud

import torch
from arabert import preprocess

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
    # Add more word mappings here as needed
    # 'مثال1': 'بديل1',
    # 'مثال2': 'بديل2',
}

WORDS_TO_BE_REMOVED = [
    'ال',  # Remove standalone ال
    'و'
]

class ArabicTextProcessor:
    """
    A comprehensive Arabic text processor that handles cleaning, normalization,
    and custom text replacements for Arabic text processing tasks.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """
        Remove punctuation from Arabic text using Unicode categories.
        """
        return ''.join(char for char in text if not ud.category(char).startswith('P'))

    @staticmethod
    def normalize_final_letters(text: str) -> str:
        """
        Normalize final letters in Arabic words to handle common spelling variations.
        - ى (alif maksura) at word end -> ي (ya)
        - ة (ta marbuta) at word end -> ه (ha)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized final letters
        """
        # Split text into words while preserving spaces and punctuation
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
    
    @staticmethod
    def normalize_final_letters_bidirectional(text: str, target_final_ya='ي', target_final_ta='ه') -> str:
        """
        Normalize final letters with configurable target characters.
        This allows you to choose whether to normalize to ى/ي or ة/ه.
        
        Args:
            text (str): Input text
            target_final_ya (str): Target character for final ya/alif maksura ('ي' or 'ى')
            target_final_ta (str): Target character for final ta marbuta/ha ('ه' or 'ة')
            
        Returns:
            str: Text with normalized final letters
        """
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Normalize final ya/alif maksura
            if word.endswith('ى') or word.endswith('ي'):
                word = word[:-1] + target_final_ya
            
            # Normalize final ta marbuta/ha
            elif word.endswith('ة') or word.endswith('ه'):
                word = word[:-1] + target_final_ta
            
            normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    @staticmethod
    def apply_character_mappings(text: str) -> str:
        """
        Apply character-level mappings to the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with character mappings applied
        """
        for original, replacement in CHAR_MAPPING.items():
            text = text.replace(original, replacement)
        return text
    
    @staticmethod
    def apply_word_mappings(text: str) -> str:
        """
        Apply word-level mappings to the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with word mappings applied
        """
        words = text.split()
        mapped_words = []
        
        for word in words:
            # Check if word needs to be replaced
            if word in WORD_MAPPING:
                mapped_words.append(WORD_MAPPING[word])
            else:
                mapped_words.append(word)
        
        return ' '.join(mapped_words)
    
    @staticmethod
    def remove_unwanted_words(text: str) -> str:
        """
        Remove unwanted standalone words from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with unwanted words removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in WORDS_TO_BE_REMOVED]
        return ' '.join(filtered_words)
    
    @staticmethod
    def clean_arabic_text(text: str) -> str:
        """
        Clean Arabic text using AraBERT preprocessor and custom cleaning.
        
        Args:
            text (str): Input Arabic text to clean
            
        Returns:
            str: Cleaned Arabic text
        """
        def remove_clitics_with_plus_simple(text: str) -> str:
            """
            Simple version: Remove any word containing a + sign completely.
            
            Args:
                text (str): Preprocessed Arabic text with clitics marked by + signs
                
            Returns:
                str: Text with all words containing + removed
            """
            words = text.split()
            cleaned_words = [word for word in words if '+' not in word]
            return ' '.join(cleaned_words)
        
        def clean_preprocessed_text(text):
            cleaned_text = text.strip("'")
            cleaned_text = cleaned_text.strip()
            return cleaned_text
        
        # Apply AraBERT preprocessing
        cleaned_text = ARABERT_OBJ.preprocess(text)
        cleaned_text = remove_clitics_with_plus_simple(cleaned_text)
        cleaned_text = clean_preprocessed_text(cleaned_text)
        return cleaned_text
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text using the substitute object and clean it.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized and cleaned text
        """
        replaced_text = SUBSTITUTE_OBJ(text=text, is_corpus=False, is_asmo=False)
        cleaned_text = ArabicTextProcessor.clean_arabic_text(replaced_text)
        return cleaned_text
    
    @staticmethod
    def process_single_text(text: str, normalize_final_letters: bool = True) -> str:
        """
        Process a single text through the complete pipeline.
        
        Args:
            text (str): Input text to process
            normalize_final_letters (bool): Whether to normalize final letters (ى↔ي, ة↔ه)
            
        Returns:
            str: Fully processed text
        """
        processed_text = ArabicTextProcessor.remove_punctuation(text) 
        # Step 1: Normalize text (includes substitution and basic cleaning)
        processed_text = ArabicTextProcessor.normalize_text(processed_text)
        
        # Step 2: Apply word-level mappings (like يئول -> يقول)
        processed_text = ArabicTextProcessor.apply_word_mappings(processed_text)
        
        # Step 3: Apply character-level mappings (like أ -> ا)
        processed_text = ArabicTextProcessor.apply_character_mappings(processed_text)
        
        # Step 4: Normalize final letters if requested
        if normalize_final_letters:
            processed_text = ArabicTextProcessor.normalize_final_letters(processed_text)
        
        # Step 5: Remove unwanted words (like standalone ال)
        processed_text = ArabicTextProcessor.remove_unwanted_words(processed_text)
        
        # Step 6: Clean up any extra whitespace
        processed_text = ' '.join(processed_text.split())
        
        return processed_text
    
    @staticmethod
    def process_texts(texts: Union[str, List[str]], normalize_final_letters: bool = True) -> Union[str, List[str]]:
        """
        Process text(s) through the complete Arabic text processing pipeline.
        
        Args:
            texts (str or List[str]): Input text(s) to process
            normalize_final_letters (bool): Whether to normalize final letters (ى↔ي, ة↔ه)
            
        Returns:
            str or List[str]: Processed text(s) - same type as input
        """
        # Handle single string input
        if isinstance(texts, str):
            return ArabicTextProcessor.process_single_text(texts, normalize_final_letters)
        
        # Handle list input
        elif isinstance(texts, list):
            return [ArabicTextProcessor.process_single_text(text, normalize_final_letters) for text in texts]
        
        else:
            raise TypeError("Input must be either a string or a list of strings")