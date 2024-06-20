#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP

#include <string>
#include <vector>

// Assuming HuffmanEncoding is a class that implements the Huffman coding algorithm
class HuffmanEncoding {
public:
    // Constructor
    explicit HuffmanEncoding();

    // Encodes a single text string into a Huffman encoded string
    std::string encode(const std::string& text) const;

    // Decodes a Huffman encoded string back into the original text
    std::string decode(const std::string& encoded_text) const;

    // Additional functions related to the model can be declared here
};

// C-style interface for interaction with other languages or systems
extern "C" {
    // Create a new HuffmanEncoding instance
    HuffmanEncoding* HuffmanEncoding_new();

    // Delete an existing HuffmanEncoding instance
    void HuffmanEncoding_delete(HuffmanEncoding* instance);

    // Encode a text using the HuffmanEncoding instance
    const char* HuffmanEncoding_encode(HuffmanEncoding* instance, const char* text);

    // Decode a Huffman encoded string back into a single string
    const char* HuffmanEncoding_decode(HuffmanEncoding* instance, const char* encoded_text);
}

#endif // HUFFMAN_HPP
