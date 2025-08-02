import re
from . import logger

LOGGER = logger.Logger.get_logger(module_name=__name__)

class ValidateText:
    """
    Text validation utilities for Arabic text processing.
    """
    
    @staticmethod
    def validate_text_in_ar(text: str) -> str:
        """
        Validate that at least 70% of the text is in Arabic.
        
        Args:
            text (str): Text to validate
            
        Returns:
            str: Original text if validation passes, empty string otherwise
        """
        def is_arabic_char(char):
            """Check if a character is Arabic."""
            char_code = ord(char)
            arabic_ranges = [
                (0x0600, 0x06FF),  # Arabic block
                (0x0750, 0x077F),  # Arabic Supplement
                (0x08A0, 0x08FF),  # Arabic Extended-A
                (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
                (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
            ]
            return any(start <= char_code <= end for start, end in arabic_ranges)
        
        if not text or not text.strip():
            return ""
        
        # Remove whitespace and punctuation for counting
        filtered_text = re.sub(r'[\s.,!?;:()\[\]{}"\'`~@#$%^&*+=<>/\\|_-]+', '', text)
        
        if not filtered_text:
            return ""
        
        arabic_count = sum(1 for char in filtered_text if is_arabic_char(char))
        total_count = len(filtered_text)
        
        arabic_percentage = arabic_count / total_count
        return text if arabic_percentage >= 0.7 else ""