import json
import argparse

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

""".format(gen_filename=gen_filename)
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
        text="""
        // {el}
        {{
        """.format(el=str(el))
        ###
        ### shape
        ###
        if "shape" in el or len(el["shape"])>0:
            shape=el["shape"]
            shape_flat=1
            for s in el["shape"]:
                shape_flat*=s
            text+="""
            Tensor::shape_type shape = {{{shape}}};""".format(i=i,shape=",".join(map(str,shape)) )
        ###
        ### operator
        ###
        if el["op"]=="IO Node":
            if el["name"]=="input/x":
                text+="""
            forward_result[{i}]=&input_var;""".format(i=i)
            elif el["name"]=="output/output.1":
                assert len(el["in"])>0, "output error"
                output_id=el["in"][0]
            else:
                #assert False, "unknown IO:"+el["name"]
                print("unknown IO:"+el["name"])
        elif el["op"]=="prim::Constant":
            if "constant_value" not in el:
                text+="""
            forward_result[{i}]=NULL;""".format(i=i)
            elif len(el["shape"])==0:
                val=el["constant_value"]
                text+="""
            Tensor c=(float){val};
            forward_result[{i}]=new VariableTensor(c);""".format(i=i,val=str(val))
            else:
                val=el["constant_value"]
                text+="""
            Tensor t= {{{val}}};
            t=t.reshape(shape);
            forward_result[{i}]=new VariableTensor(t);""".format(i=i,shape=",".join(map(str,shape)), val=",".join(map(str,val)))
        else:
            ###
            ### standard operators
            ###
            if el["op"]=="aten::mul":
                text+="""
            MulOp* op=new MulOp();"""
            elif el["op"]=="aten::add":
                text+="""
            AddOp* op=new AddOp();"""
            else:
                #assert False, "unknown op:"+el["op"]
                text+="""
            AddOp* op=NULL;"""
                print("unknown op:"+el["op"])
            ### setting operator
            text+="""
            forward_result[{i}]=op;""".format(i=i)
            ###
            ### inputs
            ###
            if "in" in el and len(el["in"])>0:
                num_inputs=len(el["in"])
                text+="""
            MCTNode* p_in;"""
                for in_id in el["in"]:
                    text+="""
            p_in=forward_result[{in_id}];
            op->inputs.push_back(p_in);""".format(in_id=in_id)
        text+="""
        }
        """
        all_text+=text

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
        cout<<input_var.grad<<endl;
    """.format(output_id=output_id)
    
    all_text+="""
        return 0;
    }
    """
    return all_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=str, help="computational graph json")
    parser.add_argument(
        "--output", type=str, default="example.gen.cpp", help="config json file"
    )
    parser.add_argument(
        "--path", type=str, default="src", nargs="?", help="config json file"
    )

    args = parser.parse_args()

    filename = args.graph
    fp=open(filename)
    obj=json.load(fp)
    code = c_code_generator(obj)
    make_code = makefile_generator(args.output)

    print("[SAVE]",args.path+"/"+args.output)
    ofp=open(args.path+"/"+args.output,"w")
    ofp.write(code)
    makefp=open(args.path+"/"+"Makefile","w")
    makefp.write(make_code)

if __name__ == "__main__":
    main()
