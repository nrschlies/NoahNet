import ctypes
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "libtext_byte_pair_encoding.so"
bpe_lib = ctypes.CDLL(str(lib_path))

# Helper function to convert Python string to C string
def to_c_string(string):
    return ctypes.c_char_p(string.encode('utf-8'))

class BytePairEncoding:
    def __init__(self, max_vocab_size):
        self.obj = bpe_lib.BytePairEncoding_new(max_vocab_size)
        
    def __del__(self):
        if self.obj:
            bpe_lib.BytePairEncoding_delete(self.obj)
        
    def train(self, texts):
        c_texts = (ctypes.c_char_p * len(texts))(*map(to_c_string, texts))
        bpe_lib.BytePairEncoding_train(self.obj, c_texts, len(texts))
        
    def encode(self, text):
        c_text = to_c_string(text)
        bpe_lib.BytePairEncoding_encode.restype = ctypes.c_void_p  # Use c_void_p to get the raw pointer
        encoded_tokens_ptr = bpe_lib.BytePairEncoding_encode(self.obj, c_text)
        
        # Convert from c_void_p to POINTER(c_char_p) to read the array
        ptr_type = ctypes.POINTER(ctypes.c_char_p)
        ptr = ctypes.cast(encoded_tokens_ptr, ptr_type)
        
        result = []
        i = 0
        while ptr[i] is not None:
            result.append(ptr[i].decode('utf-8'))
            i += 1
        
        # Free the memory allocated in C++ (assuming free_result is properly implemented)
        bpe_lib.free_result(encoded_tokens_ptr)
        return result
    
    def decode(self, tokens):
        c_tokens = (ctypes.c_char_p * len(tokens))(*map(to_c_string, tokens))
        bpe_lib.BytePairEncoding_decode.restype = ctypes.c_char_p
        decoded_text_ptr = bpe_lib.BytePairEncoding_decode(self.obj, c_tokens, len(tokens))
        
        # Convert C string back to Python string
        result = ctypes.string_at(decoded_text_ptr).decode('utf-8')
        
        # Free the allocated C string memory from C++ side
        bpe_lib.free_result(decoded_text_ptr)
        return result

# Set argument and result types for the C functions
bpe_lib.BytePairEncoding_new.argtypes = [ctypes.c_size_t]
bpe_lib.BytePairEncoding_new.restype = ctypes.c_void_p

bpe_lib.BytePairEncoding_delete.argtypes = [ctypes.c_void_p]
bpe_lib.BytePairEncoding_delete.restype = None

bpe_lib.BytePairEncoding_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
bpe_lib.BytePairEncoding_train.restype = None

bpe_lib.BytePairEncoding_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
bpe_lib.BytePairEncoding_encode.restype = ctypes.c_void_p  # Return type is void* to raw pointer

bpe_lib.BytePairEncoding_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
bpe_lib.BytePairEncoding_decode.restype = ctypes.c_char_p  # Return type is char* to C string

bpe_lib.free_result.argtypes = [ctypes.c_void_p]
bpe_lib.free_result.restype = None

# Example usage
if __name__ == "__main__":
    bpe = BytePairEncoding(100)
    bpe.train(["hello world", "hello there", "hello hello"])
    
    encoded = bpe.encode("hello world")
    print("Encoded:", encoded)
    
    decoded = bpe.decode(encoded)
    print("Decoded:", decoded)
