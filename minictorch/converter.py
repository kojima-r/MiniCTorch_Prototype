import json

def makefile_generator(gen_filename):
    make_text="""
CXX = g++
CXXFLAGS = -g -Wall  -std=c++14 -I./ -I../xtensor-blas/include -I../xtensor/include -I../xtl/include
#CXXFLAGS = -g -Wall  -fprofile-arcs -ftest-coverage -std=c++14 -I./json/include -I./xtensor-blas/include -I./xtensor/include -I./xtl/include
#LDFLAGS = -L./ -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt  -lcblas
LDFLAGS = -lcblas
CPPUTEST_HOME = ./cpputest/workspace/install
TARGET = mini_c_torch
# SRCS = main_test.cpp main.cpp
SRCS = minictorch.cpp {gen_filename}
OBJS = $(SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

$(OBJS): $(SRCS)
	$(CXX) -c $(CXXFLAGS) $^

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS) *.gcno *.gcov *~
	find . -name "*.gcda" | xargs -r rm

"""
    return make_text

def c_code_generator(obj):
    all_text="""
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;

    int main(){{
        // input data
        Tensor x={{{{1, 2}},
                {{3, 4}}}};
        VariableTensor input_var(x);
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
    output_id=None
    for i,el in enumerate(obj):
        print(el)
        text=""
        if el["op"]=="IO Node":
            if el["name"]=="input/x":
                text="""
                {{
                    forward_result[{i}]=&input_var;
                }}
                """.format(i=i)
            elif el["name"]=="output/output.1":
                assert len(el["in"])>0, "output error"
                output_id=el["in"][0]
            else:
                assert False, "unknown IO:"+el["name"]
        elif el["op"]=="prim::Constant":
            if len(el["shape"])==0:
                val=el["constant_value"]
                text="""
                {{
                    Tensor c=(float){val};
                    forward_result[{i}]=new VariableTensor(c);
                }}
                """.format(i=i,val=str(val))
            else:
                values=el["constant_value"]
                shape=el["shape"]
                shape_flat=1
                for s in el["shape"]:
                    shape_flat*=s

                text="""
                {{
                    Tensor::shape_type shape= {{{shape}}};
                    Tensor t= {{{values}}};
                    t=t.reshape(shape);
                    forward_result[{i}]=new VariableTensor(t);
                }}
                """.format(i=i,shape=",".join(map(str,shape)), values=",".join(map(str,values)))
        elif el["op"]=="aten::mul":
            num_inputs=len(el["in"])
            text="""
                {
                    MulOp* op=new MulOp();
                    MCTNode* p_in;
                """
            for in_id in el["in"]:
                text+="""
                    p_in=forward_result[{in_id}];
                    op->inputs.push_back(p_in);
                    """.format(in_id=in_id)
                
            text+="""
                    forward_result[{i}]=op;
                }}""".format(i=i)
        elif el["op"]=="aten::add":
            num_inputs=len(el["in"])
            text="""
                {
                    AddOp* op=new AddOp();
                    MCTNode* p_in;
                """
            for in_id in el["in"]:
                text+="""
                    p_in=forward_result[{in_id}];
                    op->inputs.push_back(p_in);
                    """.format(in_id=in_id)
                
            text+="""
                    forward_result[{i}]=op;
                }}""".format(i=i)
        else:
            assert False, "unknown op:"+el["op"]
        all_text+=text
        #print(text)

    all_text+="""
        cout<<"### forward computation ..."<<endl;
        forward_result[{output_id}]->forward();
        auto o = forward_result[{output_id}]->output;
        cout<<o<<endl;
    """.format(output_id=output_id)
    all_text+="""
        cout<<"### backward computation ..."<<endl;
        forward_result[{output_id}]->grad=xt::ones_like(forward_result[{output_id}]->grad);
        forward_result[{output_id}]->backward();
        cout<<input_var_x.grad<<endl;
    """.format(output_id=output_id)
    
    all_text+="""
        return 0;
    }
    """
    return all_text


filename = "network/example01.json"
fp=open(filename)
obj=json.load(fp)
code = c_code_generator(obj)
make_code = makefile_generator("example01.gen.cpp")

ofp=open("src/example01.gen.cpp","w")
ofp.write(code)
makefp=open("src/Makefile","w")
makefp.write(make_code)

