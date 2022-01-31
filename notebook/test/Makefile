
CXX = g++
CXXFLAGS += -g -Wall  -std=c++14 -I./ -I../src -I../xtensor-blas/include -I../xtensor/include -I../xtl/include
LDFLAGS = -lcblas
TARGET  = test05
SRCS    = test05.cpp test05_param.cpp
OBJS    = $(SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

%.train.o: %.cpp
	$(CXX) -c -D_TRAIN $(CXXFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS) *.gcno *.gcov *~
	find . -name "*.gcda" | xargs -r rm

