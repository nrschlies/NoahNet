import unittest
import ctypes
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "libtext_utils.so"
text_utils = ctypes.CDLL(str(lib_path))

# Setup the function prototypes
# Assume text_utils is your CDLL-loaded shared library
text_utils.toLowerCase.restype = ctypes.c_char_p
text_utils.stripNonASCII.restype = ctypes.c_char_p
text_utils.removePunctuation.restype = ctypes.c_char_p
text_utils.normalizeWhitespace.restype = ctypes.c_char_p
text_utils.stripNonPrintable.restype = ctypes.c_char_p

class TestUtils(unittest.TestCase):
    def test_toLowerCase(self):
        cases = [
            ("HELLO WORLD!", "hello world!"),
            ("Python3", "python3"),
            ("1234", "1234"),
            ("", ""),
            ("MixED CaSe", "mixed case"), 
            ("ÉTÉ", "été"),  
            ("This—should_work!", "this—should_work!"), 
            ("DATA@SCIENCE99", "data@science99"),  # Numbers and symbols
            ("Straße", "straße"),  # German sharp S
            ("Θεοί", "θεοί"),  # Greek characters
            ("Доброеутро", "доброеутро"),  # Cyrillic characters
            ("Функция", "функция"),  # Russian text
            ("مرحبا", "مرحبا"),  # Arabic text
            ("1234😊ABc!", "1234😊abc!"),  # Numbers, emojis, mixed case
            ("JAVA&java&Java", "java&java&java"),  # Multiple same words, different cases
            ("HTML & CSS", "html & css"),  # Handling ampersands and spaces
            ("2023: A SPACE ODYSSEY", "2023: a space odyssey")  # Numeric and upper case mixed text
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.toLowerCase(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed toLowerCase with input: {input_str}")

    def test_removePunctuation(self):
        cases = [
            ("Hello, World!", "Hello World"),
            ("Here's to the crazy ones.", "Heres to the crazy ones"),
            ("Spacing... 1 2 3", "Spacing 1 2 3"),
            ("", ""),
            ("!@#$%^&*()", ""),  
            ("well-used: words, yes?", "wellused words yes"),  
            ("end.", "end"),
            ("Hello...!!!, World!!", "Hello World"),  # Multiple punctuations
            ("¿Cómo están?", "Cómo están"),  # Spanish punctuation
            ("‘Quote’ and “Quote”", "Quote and Quote"),  # Different quote styles
            ('"Quoted text!"', 'Quoted text'),  # Quotation marks
            ("...Leading", "Leading"),  # Leading ellipsis
            ("trailing!!!", "trailing"),  # Trailing exclamations
            ("punctuation, inside; text.", "punctuation inside text"),  # Mixed punctuation
             ("He said, 'Hello, world!'", "He said Hello world"),  # Embedded single quotes and commas
            ("x^2 + y^2 = z^2; solve for z.", "x2  y2  z2 solve for z")  # Mathematical expression
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.removePunctuation(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed removePunctuation with input: {input_str}")

    def test_stripNonASCII(self):
        cases = [
            ("Hellö, Wörld!", "Hell, Wrld!"),
            ("naïve façade", "nave faade"),
            ("café—bar", "cafbar"),
            ("", ""),
            ("Español es divertido 😊", "Espaol es divertido "),  
            ("中文输入", ""),  
            ("Pусский текст", "P "),
            ("Music ♬ Notes ♫", "Music  Notes "),  # Musical symbols
            ("Password123!@#", "Password123!@#"),  # Mix of ASCII printable and non-ASCII
            ("500€", "500"),  # Currency symbol
            ("😀🎉👍", ""),  # Only emojis
            ("Touché", "Touch"),  # French with diacritical mark
            ("naïve", "nave"),  # Text with diaeresis
            ("Smiley 😀", "Smiley "),  # Text with emoji
            ("数学", ""),  # Chinese characters for 'mathematics'
            ("नमस्ते दुनिया", " "),  # Hindi text
            ("안녕하세요", ""),  # Korean text
            ("Hello🌍World", "HelloWorld")  # Text with planet emoji
        ]

        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.stripNonASCII(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed stripNonASCII with input: {input_str}")
    
    def test_normalizeWhitespace(self):
        cases = [
            ("  This   is  a test  ", "This is a test"),
            ("Line\nBreak", "Line Break"),
            ("\tTabbed\tText\t", "Tabbed Text"),
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.normalizeWhitespace(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed normalizeWhitespace with input: {input_str}")


if __name__ == '__main__':
    unittest.main()
