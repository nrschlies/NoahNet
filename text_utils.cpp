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
    // Lowercase
    {L'ä', "ae"}, {L'á', "a"}, {L'ă', "a"}, {L'â', "a"},
    {L'à', "a"}, {L'ã', "a"}, {L'ā', "a"},
    {L'ö', "oe"}, {L'ó', "o"}, {L'ô', "o"}, {L'ò', "o"},
    {L'ø', "oe"}, {L'õ', "o"}, {L'ō', "o"},
    {L'ü', "ue"}, {L'ú', "u"}, {L'û', "u"}, {L'ù', "u"},
    {L'ç', "c"}, {L'č', "c"}, {L'ć', "c"},
    {L'é', "e"}, {L'ě', "e"}, {L'è', "e"}, {L'ê', "e"},
    {L'í', "i"}, {L'î', "i"}, {L'ï', "i"}, {L'ì', "i"},
    {L'ñ', "n"}, {L'ń', "n"},
    {L'ß', "ss"}, {L'ś', "s"}, {L'š', "s"},
    {L'ý', "y"}, {L'ÿ', "y"}, {L'ź', "z"}, {L'ž', "z"}, {L'ż', "z"},
    // Uppercase
    {L'Ä', "Ae"}, {L'Á', "A"}, {L'Ă', "A"}, {L'Â', "A"},
    {L'À', "A"}, {L'Ã', "A"}, {L'Ā', "A"},
    {L'Ö', "Oe"}, {L'Ó', "O"}, {L'Ô', "O"}, {L'Ò', "O"},
    {L'Ø', "Oe"}, {L'Õ', "O"}, {L'Ō', "O"},
    {L'Ü', "Ue"}, {L'Ú', "U"}, {L'Û', "U"}, {L'Ù', "U"},
    {L'Ç', "C"}, {L'Č', "C"}, {L'Ć', "C"},
    {L'É', "E"}, {L'Ě', "E"}, {L'È', "E"}, {L'Ê', "E"},
    {L'Í', "I"}, {L'Î', "I"}, {L'Ï', "I"}, {L'Ì', "I"},
    {L'Ñ', "N"}, {L'Ń', "N"},
    {L'Ý', "Y"}, {L'Ÿ', "Y"}, {L'Ź', "Z"}, {L'Ž', "Z"}, {L'Ż', "Z"}
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
