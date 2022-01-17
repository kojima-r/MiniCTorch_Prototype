
CXX = g++
CXXFLAGS += -g -Wall  -std=c++14 -I./ -I../src -I../xtensor-blas/include -I../xtensor/include -I../xtl/include
LDFLAGS = -lcblas
TARGET  = vae1
SRCS    = vae1.cpp vae1_param.cpp
OBJS    = $(SRCS:.cpp=.o)


TRAIN_SRCS    = vae1.cpp vae1_param.cpp vae1_train.cpp vae1_data.cpp
TRAIN_TARGET  = vae1_train
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

