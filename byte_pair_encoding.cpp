#include "byte_pair_encoding.hpp"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <queue>
#include <map>

// Expose the required functions with C linkage
extern "C" {
BytePairEncoding::BytePairEncoding(size_t max_vocab_size) : max_vocab_size(max_vocab_size) {}

void BytePairEncoding::train(const std::vector<std::string>& texts) {
    std::unordered_map<std::string, size_t> token_frequencies;
    for (const auto& text : texts) {
        std::string normalized_text = normalize(text);
        std::vector<std::string> tokens = tokenize(normalized_text);
        for (const auto& token : tokens) {
            add_to_vocab(token);
            token_frequencies[token]++;
        }
    }
}

std::vector<std::string> BytePairEncoding::encode(const std::string& text) {
    std::string normalized_text = normalize(text);
    std::vector<std::string> tokens = tokenize(normalized_text);
    return tokens;
}

std::string BytePairEncoding::decode(const std::vector<std::string>& tokens) {
    std::string decoded_text;
    for (const auto& token : tokens) {
        decoded_text += (decoded_text.empty() ? "" : " ") + token;
    }
    return decoded_text;
}

void BytePairEncoding::add_to_vocab(const std::string& token) {
    if (vocab.find(token) == vocab.end()) {
        vocab[token] = 1;
    } else {
        vocab[token]++;
    }
}

void BytePairEncoding::update_vocab_with_pair(const std::string& best_pair_a, const std::string& best_pair_b) {
    std::string new_token = best_pair_a + best_pair_b;
    size_t new_freq = std::min(vocab[best_pair_a], vocab[best_pair_b]);
    vocab[new_token] = new_freq;
    vocab[best_pair_a] -= new_freq;
    vocab[best_pair_b] -= new_freq;
}

std::vector<std::string> BytePairEncoding::tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string BytePairEncoding::normalize(const std::string& text) {
    return std::string(normalizeWhitespace(toLowerCase(stripNonASCII(stripNonPrintable(removePunctuation(text.c_str()))))));
}


    __attribute__((visibility("default")))
    BytePairEncoding* BytePairEncoding_new(size_t max_vocab_size) {
        return new BytePairEncoding(max_vocab_size);
    }

    __attribute__((visibility("default")))
    void BytePairEncoding_delete(BytePairEncoding* instance) {
        delete instance;
    }
}