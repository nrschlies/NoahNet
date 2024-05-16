#ifndef BYTE_PAIR_ENCODING_HPP
#define BYTE_PAIR_ENCODING_HPP

#include <string>
#include <vector>

// Assuming BytePairEncoding is a class that implements the BPE algorithm
class BytePairEncoding {
public:
    // Constructor: Initialize the encoding with a maximum vocabulary size
    explicit BytePairEncoding(size_t max_vocab_size);

    // Trains the BytePairEncoding model on a list of input texts
    void train(const std::vector<std::string>& texts);

    // Encodes a single text string into a vector of encoded tokens
    std::vector<std::string> encode(const std::string& text) const;

    // Decodes a vector of encoded tokens back into the original text
    std::string decode(const std::vector<std::string>& tokens) const;

    // Additional functions related to the model can be declared here
};

// C-style interface for interaction with other languages or systems
extern "C" {
    // Create a new BytePairEncoding instance
    BytePairEncoding* BytePairEncoding_new(size_t max_vocab_size);

    // Delete an existing BytePairEncoding instance
    void BytePairEncoding_delete(BytePairEncoding* instance);

    // Train the BytePairEncoding instance with an array of texts
    void BytePairEncoding_train(BytePairEncoding* instance, const char** texts, size_t count);

    // Encode a text using the BytePairEncoding instance
    // The return type matches the corrected function in your .cpp file
    const char** BytePairEncoding_encode(BytePairEncoding* instance, const char* text);

    // Decode a list of tokens into a single string
    char* BytePairEncoding_decode(BytePairEncoding* instance, const char** tokens, size_t count);

    // Utility function to free memory allocated by encode
    void free_result(char** result);
}

#endif // BYTE_PAIR_ENCODING_HPP
