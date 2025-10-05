
import re
from typing import Union, List
import unicodedata as ud

import torch
from arabert import preprocess
import pyarabic.araby as araby
from num2words import num2words

from RDIreplacement.substitute import substitute

SUBSTITUTE_OBJ = substitute()
ARABERT_OBJ = preprocess.ArabertPreprocessor(model_name="bert-base-arabertv2", apply_farasa_segmentation=True, insert_white_spaces=False)

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

WORD_MAPPING = {
    'يئول': 'يقول',  # Example mapping
    'آل': 'قال'
}

WORDS_TO_BE_REMOVED = [
    'ال',  # Remove standalone ال
    'و'
]

# Phase 1.0
class BasicArabicTextProcessing:
    @staticmethod
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

    @staticmethod
    def normalize_texts(texts):
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]
        elif isinstance(texts, list):
            pass  # Already a list
        else:
            raise TypeError("Input must be either a string or a list of strings")
        
        cleaned_texts = BasicArabicTextProcessing.clean_arabic_text(texts)
        replaced_texts = SUBSTITUTE_OBJ(text=cleaned_texts, is_corpus=False, is_asmo=False)
        return [re.sub(r"\s+", " ", t).strip() for t in replaced_texts]

class HeavyArabicTextProcessing:
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
    def convert_numbers_to_arabic_text(texts: List[str]) -> List[str]:
        """
        Convert numeric values in texts to Arabic words.
        
        Args:
            texts (List[str]): List of input texts
        Returns:
            List[str]: List of texts with numbers converted to Arabic words
        """
        def convert_single_text(text: str) -> str:
            words = text.split()
            converted_words = []
            
            for word in words:
                if word.isdigit():
                    try:
                        converted_words.append(num2words(int(word), lang='ar'))
                    except (ValueError, NotImplementedError):
                        converted_words.append(word)  # Keep original if conversion fails
                else:
                    converted_words.append(word)
            
            return ' '.join(converted_words)

        return [convert_single_text(text) for text in texts]
        
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
        replaced_texts = HeavyArabicTextProcessing.substitute_words(texts)        
        cleaned_texts = HeavyArabicTextProcessing.process_arabic_text_arabert(replaced_texts)
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
        processed_texts = HeavyArabicTextProcessing.remove_punctuation(texts)
        processed_texts = HeavyArabicTextProcessing.convert_numbers_to_arabic_text(processed_texts)
        # Step 1: Normalize texts (includes substitution and basic cleaning)
        processed_texts = HeavyArabicTextProcessing.substitute_and_arabert(processed_texts)
        
        # Step 2: Apply word-level mappings (like يئول -> يقول)
        processed_texts = HeavyArabicTextProcessing.apply_word_mappings(processed_texts)
        
        # Step 3: Apply character-level mappings (like أ -> ا)
        processed_texts = HeavyArabicTextProcessing.apply_character_mappings(processed_texts)
        
        # Step 4: Normalize final letters if requested
        if normalize_final_letters:
            processed_texts = HeavyArabicTextProcessing.normalize_final_letters(processed_texts)
        
        # Step 5: Remove unwanted words (like standalone ال)
        processed_texts = HeavyArabicTextProcessing.remove_unwanted_words(processed_texts)
        
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
        
        processed_texts = HeavyArabicTextProcessing.remove_punctuation(texts)
        processed_texts = HeavyArabicTextProcessing.remove_tashkeel_tatweel(processed_texts)
        processed_texts = HeavyArabicTextProcessing.convert_numbers_to_arabic_text(processed_texts)
        replaced_texts = HeavyArabicTextProcessing.substitute_words(processed_texts)
        processed_texts = HeavyArabicTextProcessing.apply_word_mappings(replaced_texts)
        processed_texts = HeavyArabicTextProcessing.apply_character_mappings(processed_texts)
        processed_texts = HeavyArabicTextProcessing.normalize_final_letters(processed_texts)
        processed_texts = [' '.join(text.split()) for text in processed_texts]
        return processed_texts[0] if is_single_input else processed_texts
    

import unicodedata
from typing import List, Union
import re

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
    def normalize_texts(texts: Union[str, List[str]], normalize_final_letters: bool = True, substitute=False) -> Union[str, List[str]]:
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
        # Step 5: Normalize Hamzas and Maddas
        processed_texts = StandardArabicTextProcessor.normalize_characters(processed_texts)

        # Step 6: Optional normalization of final letters (ى to ي, ة to ه)
        if normalize_final_letters:
            processed_texts = [
                text.replace('ى', 'ي').replace('ة', 'ه')
                for text in processed_texts
            ]

        # Step 7: Clean excessive whitespace (NEW STEP)
        processed_texts = StandardArabicTextProcessor.clean_whitespace(processed_texts)
        if substitute:
            replaced_texts = SUBSTITUTE_OBJ(
                text=processed_texts,
                is_corpus=False,
                is_asmo=False
            )
            replaced_texts = [re.sub(r"\s+", " ", t).strip() for t in replaced_texts]
            return replaced_texts[0] if is_single_input else replaced_texts

        # No substitution case
        return processed_texts[0] if is_single_input else processed_texts