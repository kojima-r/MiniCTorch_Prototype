
CXX = g++
CXXFLAGS += -g -Wall  -std=c++14 -I./ -I../src -I../xtensor-blas/include -I../xtensor/include -I../xtl/include
LDFLAGS = -lcblas
TARGET  = bbb
SRCS    = bbb.cpp bbb_param.cpp
OBJS    = $(SRCS:.cpp=.o)


TRAIN_SRCS    = bbb.cpp bbb_param.cpp bbb_train.cpp bbb_data.cpp
TRAIN_TARGET  = bbb_train
TRAIN_OBJS    = $(TRAIN_SRCS:.cpp=.train.o)

all: $(TARGET) $(TRAIN_TARGET)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) -o $@ $^ -D_TRAIN $(CXXFLAGS) $(LDFLAGS)

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

