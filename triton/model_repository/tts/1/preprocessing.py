import re
from typing import Optional, Tuple

from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, basic_cleaners


class VoiceBambaraTextPreprocessor:
    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = self.expand_number(text)
        text = self.transliterate_bambara(text)

        return text

    def transliterate_bambara(self, text):
        """
        Transliterate Bambara text using a specified mapping of special characters.
        Parameters:
        - text (str): The original Bambara text.
        Returns:
        - str: The transliterated text.
        """
        bambara_transliteration = {
            'ɲ': 'ny',
            'ɛ': 'è',
            'ɔ': 'o',
            'ŋ': 'ng',
            'ɟ': 'j',
            'ʔ': "'",
            'ɣ': 'gh',
            'ʃ': 'sh',
            'ߒ': 'n',
            'ߎ': "u",
        }

        # Perform the transliteration
        transliterated_text = "".join(bambara_transliteration.get(char, char) for char in text)

        return transliterated_text

    def expand_number(self, text):
        """
        Normalize Bambara text for TTS by replacing numerical figures with their word equivalents.
        Args:
        text (str): The text to be normalized.
        Returns:
        str: The normalized Bambara text.
        """

        # A regex pattern to match all numbers
        number_pattern = re.compile(r'\b\d+\b')

        # Function to replace each number with its Bambara text
        def replace_number_with_text(match):
            number = int(match.group())
            return self.number_to_bambara(number)

        # Replace each number in the text with its Bambara word equivalent
        normalized_text = number_pattern.sub(replace_number_with_text, text)

        return normalized_text

    def number_to_bambara(self, n):

        """
        Convert a number into its textual representation in Bambara using recursion.
        Args:
        n (int): The number to be converted.
        Returns:
        str: The number expressed in Bambara text.
        Examples:
        >>> number_to_bambara(123)
        'kɛmɛ ni mugan ni saba'
        Notes:
        This function assumes that 'n' is a non-negative integer.
        """

        # Bambara numbering rules
        units = ["", "kɛlɛn", "fila", "saba", "naani", "duuru", "wɔrɔ", "wòlonwula", "sɛɛgin", "kɔnɔntɔn"]
        tens = ["", "tan", "mugan", "bisaba", "binaani", "biduuru", "biwɔrɔ", "biwòlonfila", "bisɛɛgin", "bikɔnɔntɔn"]
        hundreds = ["", "kɛmɛ"]
        thousands = ["", "waga"]
        millions = ["", "milyɔn"]

        # Handle zero explicitly
        if n == 0:
            return ""  # bambara does not support zero

        if n < 10:
            return units[n]
        elif n < 100:
            return tens[n // 10] + (" ni " + self.number_to_bambara(n % 10) if n % 10 > 0 else "")
        elif n < 1000:
            return hundreds[1] + (" " + self.number_to_bambara(n // 100) if n >= 200 else "") + (
                " ni " + self.number_to_bambara(n % 100) if n % 100 > 0 else "")
        elif n < 1_000_000:
            return thousands[1] + " " + self.number_to_bambara(n // 1000) + (
                " ni " + self.number_to_bambara(n % 1000) if n % 1000 > 0 else "")
        else:
            return millions[1] + " " + self.number_to_bambara(n // 1_000_000) + (
                " ni " + self.number_to_bambara(n % 1_000_000) if n % 1_000_000 > 0 else "")


class BambaraTokenizer(VoiceBpeTokenizer):
    """
    A tokenizer for the Bambara language that extends the VoiceBpeTokenizer.
    Attributes:
        preprocessor: An instance of VoiceBambaraTextPreprocessor for text preprocessing.
        char_limits: A dictionary to hold character limits for languages.
    """

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initializes the BambaraTokenizer with a given vocabulary file.
        Args:
            vocab_file: The path to the vocabulary file, defaults to None.
        """
        super().__init__(vocab_file)
        self.preprocessor = VoiceBambaraTextPreprocessor()
        self.char_limits['bm'] = 200  # Set character limit for Bambara language

    def preprocess_text(self, txt: str, lang: str) -> str:
        """
        Preprocesses the input text based on the language.
        Args:
            txt: The text to preprocess.
            lang: The language code of the text.
        Returns:
            The preprocessed text.
        """
        # Delegate preprocessing to the parent class for non-Bambara languages
        if lang != "bm":
            return super().preprocess_text(txt, lang)

        # Apply Bambara-specific preprocessing
        txt = self.preprocessor.preprocess(txt)
        txt = basic_cleaners(txt)
        return txt
