#include "text_utils.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <string>
#include <unordered_map>

extern "C" {
    __attribute__((visibility("default")))
    
    const char* toLowerCase(const char* str) {
        static std::string utf8Result;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wideStr = converter.from_bytes(str);

        // Convert characters to lowercase using the specified locale
        for (auto& c : wideStr) {
            c = std::tolower(c, std::locale("en_US.UTF-8"));
        }

        utf8Result = converter.to_bytes(wideStr);
        return utf8Result.c_str();
    }

    const char* normalizeWhitespace(const char* str) {
        static std::string normalized;
        normalized.clear();
        std::string temp(str);
        bool lastWasSpace = true;
        for (char& c : temp) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!lastWasSpace) {
                    normalized += ' ';
                    lastWasSpace = true;
                }
            } else {
                normalized += c;
                lastWasSpace = false;
            }
        }
        if (!normalized.empty() && normalized.back() == ' ') {
            normalized.pop_back(); // Remove trailing space
        }
        return normalized.c_str();
    }

    const char* stripNonASCII(const char* str) {
        static std::string utf8Result;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wideStr = converter.from_bytes(str);
        utf8Result.clear();
        for (wchar_t c : wideStr) {
            if (c < 128) utf8Result += static_cast<char>(c);
        }
        return utf8Result.c_str();
    }

    const char* stripNonPrintable(const char* str) {
        static std::string printableStr;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wideStr = converter.from_bytes(str);
        printableStr.clear();
        for (wchar_t c : wideStr) {
            if (std::isprint(static_cast<unsigned char>(c))) {
                printableStr += converter.to_bytes(std::wstring(1, c));
            }
        }
        return printableStr.c_str();
    }

    const char* removePunctuation(const char* str) {
        static std::string result;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wideStr = converter.from_bytes(str);  // Convert UTF-8 string to wide string

        // Remove punctuation using standard library calls
        std::wstring noPunct;
        std::remove_copy_if(wideStr.begin(), wideStr.end(), std::back_inserter(noPunct), 
                            [](wchar_t c){ return std::ispunct(c); });

        result = converter.to_bytes(noPunct);  // Convert wide string back to UTF-8
        return result.c_str();
    }
}

