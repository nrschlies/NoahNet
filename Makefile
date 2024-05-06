CXX=g++
CXXFLAGS=-fPIC -Wall -Wextra -O2 -std=c++11 -arch x86_64
LDFLAGS=-shared -arch x86_64
TARGET_LIB=libtext_utils.so

SRCS=text_utils.cpp
OBJS=$(SRCS:.cpp=.o)

.PHONY: all clean test

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET_LIB)

test:
	python3 tester.py
