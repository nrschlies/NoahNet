import unittest
import ctypes
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "libtext_utils.so"
text_utils = ctypes.CDLL(str(lib_path))

# Setup the function prototypes
text_utils.toLowerCase.argtypes = [ctypes.c_char_p]
text_utils.toLowerCase.restype = ctypes.c_char_p

text_utils.removePunctuation.argtypes = [ctypes.c_char_p]
text_utils.removePunctuation.restype = ctypes.c_char_p

text_utils.stripNonASCII.argtypes = [ctypes.c_char_p]
text_utils.stripNonASCII.restype = ctypes.c_char_p

class TestUtils(unittest.TestCase):
    def test_toLowerCase(self):
        cases = [
            ("HELLO WORLD!", "hello world!"),
            ("Python3", "python3"),
            ("1234", "1234"),
            ("", ""),
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
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.removePunctuation(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed removePunctuation with input: {input_str}")

    def test_stripNonASCII(self):
        cases = [
            ("Hellö, Wörld!", "Helloe, Woerld!"),
            ("naïve façade", "naive facade"),
            ("café—bar", "cafebar"),
            ("", ""),
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                result = text_utils.stripNonASCII(input_str.encode('utf-8')).decode('utf-8')
                self.assertEqual(result, expected, f"Failed stripNonASCII with input: {input_str}")

if __name__ == '__main__':
    unittest.main()
