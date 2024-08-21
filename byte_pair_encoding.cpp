#include "byte_pair_encoding.hpp"
#include <cstring>
#include <iostream>

// Constructor
BytePairEncoding::BytePairEncoding(size_t max_vocab_size) {
    this->max_vocab_size = max_vocab_size;
    // Initialize any necessary member variables or data structures here
    // For example, you might initialize a vocabulary map
    this->vocab = {};
}

// Train method
void BytePairEncoding::train(const std::vector<std::string>& texts) {
    // Example training implementation
    for (const auto& text : texts) {
        // Tokenize the text and update the vocabulary
        // This is a placeholder for actual training logic
        for (const char& ch : text) {
            std::string token(1, ch);
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = 1;
            } else {
                vocab[token]++;
            }
        }
    }
    // Limit the vocabulary size to max_vocab_size
    if (vocab.size() > max_vocab_size) {
        // Placeholder for logic to reduce vocabulary size
    }
}

// Encode method
std::vector<std::string> BytePairEncoding::encode(const std::string& text) const {
    std::vector<std::string> encoded_tokens;
    // Example encoding implementation
    for (const char& ch : text) {
        std::string token(1, ch);
        if (vocab.find(token) != vocab.end()) {
            encoded_tokens.push_back(token);
        } else {
            encoded_tokens.push_back("<unk>"); // Unknown token
        }
    }
    return encoded_tokens;
}

// Decode method
std::string BytePairEncoding::decode(const std::vector<std::string>& tokens) const {
    std::string decoded_text;
    // Example decoding implementation
    for (const auto& token : tokens) {
        if (token != "<unk>") {
            decoded_text += token;
        } else {
            decoded_text += "?"; // Placeholder for unknown token
        }
    }
    return decoded_text;
}

// C-style interface
extern "C" {
    BytePairEncoding* BytePairEncoding_new(size_t max_vocab_size) {
        return new BytePairEncoding(max_vocab_size);
    }

    void BytePairEncoding_delete(BytePairEncoding* instance) {
        delete instance;
    }

    void BytePairEncoding_train(BytePairEncoding* instance, const char** texts, size_t count) {
        if (instance && texts) {
            std::vector<std::string> texts_vec(texts, texts + count);
            instance->train(texts_vec);
        }
        instance->train(texts_vec);
    }

    const char** BytePairEncoding_encode(BytePairEncoding* instance, const char* text) {
        if (!instance || !text) return nullptr;
        std::vector<std::string> encoded_tokens = instance->encode(text);
        char** result = new char*[encoded_tokens.size() + 1]();
        for (size_t i = 0; i < encoded_tokens.size(); ++i) {
            result[i] = new char[encoded_tokens[i].length() + 1];
            std::strcpy(result[i], encoded_tokens[i].c_str());
        }
        result[encoded_tokens.size()] = nullptr; // Null-terminated array
        return const_cast<const char**>(result);
    }

    char* BytePairEncoding_decode(BytePairEncoding* instance, const char** tokens, size_t count) {
        if (!instance || !tokens) return nullptr;
        std::vector<std::string> tokens_vec(tokens, tokens + count);
        std::string decoded = instance->decode(tokens_vec);
        char* result = new char[decoded.size() + 1];
        std::strcpy(result, decoded.c_str());
        return result;
    }

    void free_result(char** result) {
        if (!result) return;
        for (size_t i = 0; result[i] != nullptr; ++i) {
            delete[] result[i];
        }
        delete[] result;
    }
}
