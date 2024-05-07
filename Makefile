CXX=g++
CXXFLAGS=-fPIC -Wall -Wextra -O2 -std=c++11 -arch x86_64
LDFLAGS=-shared -arch x86_64
TARGET_LIB=libtext_bpe.so

SRCS=text_utils.cpp byte_pair_encoding.cpp
OBJS=$(SRCS:.cpp=.o)

.PHONY: all clean test

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET_LIB)

test:
	python3 /testers/bpe_tester.py
