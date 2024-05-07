#ifndef BYTE_PAIR_ENCODING_HPP
#define BYTE_PAIR_ENCODING_HPP

#include "text_utils.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

/**
 * Class for Byte Pair Encoding text compression technique.
 * It allows for training on a set of texts to build a vocabulary,
 * encoding texts into sequences of tokens, and decoding sequences
 * of tokens back into the original texts.
 */
class BytePairEncoding {
public:
    /**
     * Constructor for BytePairEncoding.
     * @param max_vocab_size The maximum size of the vocabulary.
     */
    BytePairEncoding(size_t max_vocab_size);

    /**
     * Trains the BPE model on a collection of texts.
     * @param texts A vector of strings used for training the vocabulary.
     */
    void train(const std::vector<std::string>& texts);

    /**
     * Encodes a given text using the BPE model.
     * @param text The text to encode.
     * @return A vector of tokens representing the encoded text.
     */
    std::vector<std::string> encode(const std::string& text);

    /**
     * Decodes a sequence of tokens back into the original text.
     * @param tokens A vector of tokens to decode.
     * @return The decoded text.
     */
    std::string decode(const std::vector<std::string>& tokens);

private:
    size_t max_vocab_size;  ///< Maximum size of the vocabulary.
    std::unordered_map<std::string, size_t> vocab;  ///< Vocabulary with frequency of tokens.
    std::vector<std::pair<std::string, std::string>> bpe_codes;  ///< List of BPE merge rules.

    /**
     * Adds a token to the vocabulary.
     * @param token The token to add to the vocabulary.
     */
    void add_to_vocab(const std::string& token);

    /**
     * Updates the vocabulary by merging two tokens into a new token.
     * @param best_pair_a The first part of the pair to merge.
     * @param best_pair_b The second part of the pair to merge.
     */
    void update_vocab_with_pair(const std::string& best_pair_a, const std::string& best_pair_b);

    /**
     * Normalizes a text string using external normalization functions.
     * @param text The text to normalize.
     * @return The normalized text.
     */
    std::string normalize(const std::string& text);

    /**
     * Tokenizes a normalized text into a vector of tokens.
     * @param text The text to tokenize.
     * @return A vector of tokens extracted from the text.
     */
    std::vector<std::string> tokenize(const std::string& text);
};

#endif // BYTE_PAIR_ENCODING_HPP
