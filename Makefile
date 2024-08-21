CXX = g++
CXXFLAGS = -fPIC -Wall -Wextra -O2 -std=c++11 -arch arm64
LDFLAGS = -shared -arch arm64
TARGET_LIB = libtext_byte_pair_encoding.so

# Include all relevant source files here
SRCS = byte_pair_encoding.cpp text_utils.cpp huffman.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean test

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET_LIB)

test:
	python3 testers/bpe_tester.py
