#ifndef TEXT_UTILS_HPP
#define TEXT_UTILS_HPP

#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

const char* toLowerCase(const char* str);
const char* removePunctuation(const char* str);
const char* tokenize(const char* str);  // Simplified return type for interfacing
const char* stripNonASCII(const char* str);
const char* normalizeWhitespace(const char* str);
const char* stripNonPrintable(const char* str);

#ifdef __cplusplus
}
#endif

#endif // TEXT_UTILS_HPP
