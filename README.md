By applying BPE, we can observe the following tokenization:

- Initial Vocabulary: `{"Hello", "world", "This", "is", "a", "test", "NoahNet", "awesome"}`
- After BPE Training:
  - `BytePairEncoding::train(corpus)`:
    - Vocabulary: `{"Hello", "world", "This", "is", "a", "test", "NoahNet", "awesome", "is_", "Th", "is a", "is aw", "No", "ah", "Net", "Noah", "Net is", "NoahNet ", "a", "t", "est", "NoahNet is"}`
- Encoding: `BytePairEncoding::encode("This is a test")` results in `["Th", "is", "is", "a", "t", "est"]`
- Decoding: `BytePairEncoding::decode(["Th", "is", "is", "a", "t", "est"])` yields `"This is a test"`

## Text Utilities
The `text_utils.hpp` and `text_utils.cpp` files contain utility functions used in text preprocessing. These utilities include:

- `normalizeWhitespace`: A function to normalize whitespace in the text.
- `toLowerCase`: A function to convert text to lowercase.
- `stripNonASCII`: A function to remove non-ASCII characters from the text.
- `stripNonPrintable`: A function to remove non-printable characters from the text.
- `removePunctuation`: A function to remove punctuation from the text.
- `normalize`: A function to apply a series of text normalization steps.

### Example: Text Utilities
Suppose we have the following text: `"Hello! How are you?"`

- After applying text utilities:
  - `normalizeWhitespace("Hello! How are you?")` results in `"Hello! How are you?"`
  - `toLowerCase("Hello! How are you?")` results in `"hello! how are you?"`
  - `stripNonASCII("Hello! How are you?")` results in `"Hello! How are you?"`
  - `stripNonPrintable("Hello! \nHow are you?")` results in `"Hello! How are you?"`
  - `removePunctuation("Hello! How are you?")` results in `"Hello How are you"`
  - `normalize("Hello! \nHow are you?")` results in `"hello how are you"`
  
## Building NoahNet
To build NoahNet, you can include the provided source files in your project and customize them according to your requirements. After implementing BPE and tokenization, the next steps would involve working on the loss function and parsing tokens to indices for further training.

## Usage
To use the Byte Pair Encoding and text utilities provided in NoahNet, follow these steps:

1. Include the necessary header files (`byte_pair_encoding.hpp` and `text_utils.hpp`) in your project.
2. Create an instance of `BytePairEncoding` with the desired maximum vocabulary size.
3. Train the BPE model using a corpus of texts with the `train` method.
4. Encode text using the `encode` method.
5. Decode tokens back to text using the `decode` method.

## Contributing
Contributions to NoahNet are welcome! If you have suggestions, bug reports, or want to contribute code, please feel free to open an issue or submit a pull request on GitHub.

## License
NoahNet is licensed under the [MIT License](LICENSE).
