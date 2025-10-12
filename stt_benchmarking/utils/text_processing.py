
import re
from typing import Union, List
import unicodedata

import torch
from num2words import num2words

from RDIreplacement.substitute import substitute

SUBSTITUTE_OBJ = substitute()

class StandardArabicTextProcessor:
    """
    A comprehensive Arabic text processor for Arabic-only texts, handling cleaning,
    normalization, and custom text replacements for Arabic ASR tasks, following
    the normalization steps from Chowdhury et al. (arXiv:2105.14779).
    All functions operate on lists for optimal bulk processing.
    """
    EASTERN_TO_WESTERN_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    
    # Dictionary for normalizing Hamzas and Maddas
    CHAR_MAPPING = {
        'أ': 'ا',
        'إ': 'ا',
        'آ': 'ا',
        'پ': 'ب',
        'ڤ': 'ف',
        'ؤ': 'و',
        'ئ': 'ي',
        'ء': ''
    }
    
    # Arabic diacritics to remove (Unicode points for common Arabic diacritics)
    DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u065F]')
    
    # Symbols and special characters to remove
    SYMBOLS_TO_REMOVE = re.compile(r'[<>\-_\[\]{}().,;:!?"\'/\\|~`^*+=&$#@%]')
    
    # English letters pattern (a-z, A-Z)
    ENGLISH_LETTERS = re.compile(r'[a-zA-Z]')

    @staticmethod
    def convert_numbers_to_arabic_text(texts: List[str]) -> List[str]:
        """
        Convert numeric values in texts to Arabic words.
        """
        def convert_single_text(text: str) -> str:
            words = text.split()
            converted_words = []
            for word in words:
                if word.isdigit():
                    try:
                        converted_words.append(num2words(int(word), lang="ar"))
                    except Exception:
                        converted_words.append(word)
                else:
                    converted_words.append(word)
            return " ".join(converted_words)
        return [convert_single_text(text) for text in texts]

    @staticmethod
    def remove_symbols_and_english_letters(texts: List[str]) -> List[str]:
        """
        Remove all symbols and English letters while keeping Arabic text and numbers.
        
        Args:
            texts (List[str]): List of input Arabic texts
            
        Returns:
            List[str]: List of texts with symbols and English letters removed
        """
        processed_texts = []
        for text in texts:
            # Remove symbols first
            text = StandardArabicTextProcessor.SYMBOLS_TO_REMOVE.sub('', text)
            # Remove English letters
            text = StandardArabicTextProcessor.ENGLISH_LETTERS.sub('', text)
            processed_texts.append(text)
        return processed_texts

    @staticmethod
    def remove_punctuation(texts: List[str]) -> List[str]:
        """
        Remove punctuation from Arabic texts using Unicode categories, keeping % and @.
        This method is now enhanced to remove more comprehensive symbols.
        
        Args:
            texts (List[str]): List of input Arabic texts
            
        Returns:
            List[str]: List of texts with punctuation removed except % and @
        """
        processed_texts = []
        for text in texts:
            # Remove all punctuation and symbols except % and @
            cleaned_text = ''.join(char for char in text if not (
                unicodedata.category(char).startswith('P') and char not in ['%', '@']
            ) and not (
                # Remove additional symbol categories
                unicodedata.category(char).startswith('S')  # Symbol categories
            ))
            processed_texts.append(cleaned_text)
        return processed_texts

    @staticmethod
    def remove_diacritics(texts: List[str]) -> List[str]:
        """
        Remove Arabic diacritics from texts.
        
        Args:
            texts (List[str]): List of input Arabic texts
            
        Returns:
            List[str]: List of texts with diacritics removed
        """
        diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u065F]')
        return [diacritics_pattern.sub('', text) for text in texts]

    @staticmethod
    def transliterate_digits(texts: List[str]) -> List[str]:
        """
        Convert Eastern Arabic numerals to Western Arabic numerals using translation table.
        
        Args:
            texts (List[str]): List of input Arabic texts
            
        Returns:
            List[str]: List of texts with Arabic numerals converted to Western numerals
        """
        return [text.translate(StandardArabicTextProcessor.EASTERN_TO_WESTERN_NUM) for text in texts]
    
    @staticmethod
    def normalize_characters(texts: List[str]) -> List[str]:
        """
        Normalize Hamzas and Maddas using the provided character mapping.
        
        Args:
            texts (List[str]): List of input Arabic texts
            
        Returns:
            List[str]: List of texts with normalized Hamzas and Maddas
        """
        def replace_chars(text: str) -> str:
            for src_char, tgt_char in StandardArabicTextProcessor.CHAR_MAPPING.items():
                text = text.replace(src_char, tgt_char)
            return text
        return [replace_chars(text) for text in texts]

    @staticmethod
    def clean_whitespace(texts: List[str]) -> List[str]:
        """
        Clean excessive whitespace - remove multiple spaces and trim.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of texts with cleaned whitespace
        """
        return [re.sub(r'\s+', ' ', text.strip()) for text in texts]
    
    @staticmethod
    def normalize_final_char(texts: List[str]) -> List[str]:
        """
        Normalizes the final character of each word in a list of texts.
        Specifically, it replaces 'ى' with 'ي' and 'ة' with 'ه'
        when they appear at the end of a word.

        Args:
            texts (List[str]): A list of input Arabic strings.

        Returns:
            List[str]: A list of processed strings with normalized final characters.
        """
        def replace_final_char(text: str) -> str:
            # This helper function is applied to each word found in the text
            def replace_ending(match):
                word = match.group(0)
                if word.endswith('ى'):
                    return word[:-1] + 'ي'
                if word.endswith('ة'):
                    return word[:-1] + 'ه'
                return word
            
            # Use regex to find all words and apply the replacement logic
            return re.sub(r'\S+', replace_ending, text)
        return [replace_final_char(text) for text in texts]

    @staticmethod
    def main(texts: Union[str, List[str]], normalize_final_letters: bool = True, substitute=False) -> Union[str, List[str]]:
        """
        Process Arabic text(s) through the complete Arabic text processing pipeline.
        All processing is done in bulk mode for optimal performance.
        
        Args:
            texts (str or List[str]): Input Arabic text(s) to process
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

        # Step 1: Remove symbols and English letters (NEW STEP)
        processed_texts = StandardArabicTextProcessor.remove_symbols_and_english_letters(texts)

        # Step 2: Remove punctuation (keeping % and @) - enhanced version
        processed_texts = StandardArabicTextProcessor.remove_punctuation(processed_texts)

        # Step 3: Remove Arabic diacritics
        processed_texts = StandardArabicTextProcessor.remove_diacritics(processed_texts)

        # Step 4: Transliterate Arabic digits to Western numerals
        processed_texts = StandardArabicTextProcessor.transliterate_digits(processed_texts)
        processed_texts = StandardArabicTextProcessor.convert_numbers_to_arabic_text(processed_texts)

        if substitute:
            processed_texts = SUBSTITUTE_OBJ(
                text=processed_texts,
                is_corpus=False,
                is_asmo=False
            )

        if normalize_final_letters:
            processed_texts = [
                text.replace('ى', 'ي').replace('ة', 'ه')
                for text in processed_texts
            ]
        
        processed_texts = StandardArabicTextProcessor.normalize_characters(processed_texts)
        processed_texts = StandardArabicTextProcessor.clean_whitespace(processed_texts)
        return processed_texts[0] if is_single_input else processed_texts