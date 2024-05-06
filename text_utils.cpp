#include "text_utils.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <unordered_map>
#include <string>




extern "C" {

const char* toLowerCase(const char* str) {
    static std::string lowerCaseStr;
    lowerCaseStr.clear();
    while (*str) {
        lowerCaseStr += std::tolower((unsigned char)*str);
        str++;
    }
    return lowerCaseStr.c_str();
}

const char* removePunctuation(const char* str) {
    static std::string noPunct;
    noPunct.clear();
    while (*str) {
        if (!std::ispunct((unsigned char)*str)) {
            noPunct += *str;
        }
        str++;
    }
    return noPunct.c_str();
}

const char* tokenize(const char* str) {
    static std::string tokens;
    std::istringstream iss(str);
    std::string token;
    tokens.clear();
    try {
        while (iss >> token) {
            tokens += token + " ";  // Simplified tokenization output for C interface
        }
    } catch (const std::exception& e) {
        std::cerr << "Error tokenizing string: " << e.what() << '\n';
        return "";
    }
    return tokens.c_str();
}

const char* stripNonASCII(const char* str) {
    static std::string asciiStr;
    asciiStr.clear();
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wideStr = converter.from_bytes(str);

    // Map of common accented characters to ASCII
    std::unordered_map<wchar_t, std::string> accents = {
        {L'ä', "ae"}, {L'ö', "oe"}, {L'ü', "ue"},
        {L'Ä', "Ae"}, {L'Ö', "Oe"}, {L'Ü', "Ue"},
        {L'ß', "ss"}, {L'é', "e"},  {L'è', "e"},
        // Add more mappings as necessary
    };

    for (wchar_t c : wideStr) {
        if (c < 128) {
            asciiStr += static_cast<char>(c);
        } else if (accents.find(c) != accents.end()) {
            asciiStr += accents[c];
        }
    }

    return asciiStr.c_str();
}

const char* normalizeWhitespace(const char* str) {
    static std::string normalized;
    normalized.clear();
    std::unique_copy(str, str + std::strlen(str), std::back_inserter(normalized),
                     [](char a, char b) { return std::isspace(a) && std::isspace(b); });
    return normalized.c_str();
}

const char* stripNonPrintable(const char* str) {
    static std::string printableStr;
    printableStr.clear();
    while (*str) {
        if (std::isprint((unsigned char)*str)) {
            printableStr += *str;
        }
        str++;
    }
    return printableStr.c_str();
}

}
