import unittest
import ctypes
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "libtext_bpe.so"
lib = ctypes.CDLL(str(lib_path))

# Define function prototypes
# Set restypes directly
lib.train.restype = ctypes.c_char_p
lib.encode.restype = ctypes.c_char_p
lib.decode.restype = ctypes.c_char_p

