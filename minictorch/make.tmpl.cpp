CXX = g++
CXXFLAGS += {{optimize}} -Wall  -std=c++14 -I./ -I{{minictorch_inc}} -I{{xtensor_base}}xtensor-blas/include -I{{xtensor_base}}xtensor/include -I{{xtensor_base}}xtl/include
LDFLAGS = {{libs}}
TARGET  = {{proj}}
SRCS    = {{proj}}_main.cpp {{proj}}_param.cpp
OBJS    = $(SRCS:.cpp=.o)


{% if code == "all" %}
TRAIN_SRCS    = {{proj}}_main.cpp {{proj}}_param.cpp {{proj}}_train.cpp {{proj}}_data.cpp
TRAIN_TARGET  = {{proj}}_train
TRAIN_OBJS    = $(TRAIN_SRCS:.cpp=.train.o)

all: $(TARGET) $(TRAIN_TARGET)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) -o $@ $^ -D_TRAIN $(CXXFLAGS) $(LDFLAGS)
{% else %}
all: $(TARGET)
{% endif %}

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


