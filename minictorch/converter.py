import json
import argparse
import numpy as np

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


# 210701 mod mari
import re

pat = re.compile(r'([^\[\]]*)\[(.*)\]')

def get_attr_from_model( s, model ):
    
    arr = s.split("/")
    if len(arr)>1:
        class_name = arr[0]
        xx = arr[1:-1]
        obj_name_list = []
        for e in xx:
            m = pat.match(e)
            if(m):
                class_name= m.group(1)
                obj_name  = m.group(2)
            else:
                obj_name = e
            obj_name_list.append(obj_name)
        m_model = model
        for o in obj_name_list:
            m_model=getattr(m_model,o)
        return m_model
    raise Exception("Unknown attribute:"+s)
    return None


# ex. Net/Linear[fc1]/weight/43 -> fc1_weight
def get_param_name( s1 ):
    
    s2 = s1.split('/')
    s3 = re.findall("(?<=\[).+?(?=\])", s2[1])
    #print(s2)
    s4 = s3[0] + '_' + s2[2]
    return s4


def string_tensor( key, out ):
    
    tmp = out.to('cpu').detach().numpy().copy()
    p1 = np.reshape( tmp,(-1,) )
    n1 = len(p1)
    s1 = 'Tensor ' + key + ' ={ '
    
    num = 8
    nw1 = n1//num
    nw2 = n1% num
    if nw2 == 0:
        nw1 = nw1 - 1
        nw2 = num
            
    l = 0
    for k in range(nw1):
        for i in range(num):
            s1 = s1 + str(p1[l]) + ','
            l = l + 1
        s1 = s1 + '\n' + ' '*24
    if nw2 > 0:
        for i in range(nw2):
            s1 = s1 + str(p1[l])+ ','
            l = l + 1
    s1 = s1 + ' }'
    #print("tensor :",s1)
    
    s2 = key + '.reshape({'
    n2 = len( tmp.shape )
    for i in range(n2-1):
       s2 = s2 + str( tmp.shape[i] ) + ','
    s2 = s2 + str( tmp.shape[n2-1] ) + '})'
    #print("shape :",s2)
    
    return s1, s2
    
def c_param_generator( obj, model, input_data, rand_flag=0 ):
    
    s1,s2 = string_tensor( "xin", input_data )
    
    all_text="""
        // input data
        
        {ivar1};
        {ivar2};
        """.format(ivar1=s1,ivar2=s2)
        
    if rand_flag > 0:
        return all_text
    #all_text=""
    
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            print(el)
            text="""
        // {el}
        """.format(el=str(el))
    
            name = el["name"]
            key = get_param_name( name )
            attr = get_attr_from_model( name, model )
            s1, s2 = string_tensor( key, attr )
            if attr.ndim == 1:
                text+="""
        {ivar1};
        """.format(i=i,ivar1=s1)
            else:
                text+="""
        {ivar1};
        {ivar2};
        """.format(i=i,ivar1=s1,ivar2=s2)
        
            all_text+=text
    
    return all_text

# 210702 mod mari
def c_code_generator( obj, model, param_file="", rand_flag=0 ):
    
    all_text="""
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;

    int main()
    {"""
    
    if len(param_file) > 0:
        text="""
        
    #include "{fn}"
    """.format(fn=param_file)
        all_text += text
    
    text="""
        // input data
        VariableTensor input_var(xin);
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
    all_text += text
    
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
        if el["op"]=="prim::GetAttr":
            pass
        elif "shape" in el or len(el["shape"]) > 0:
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
                if "input" in el["name"]:  # 210702 add mari
                    text+="""
            forward_result[{i}] = &input_var;""".format(i=i)
                elif "output" in el["name"]: 
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
            
        elif el["op"]=="prim::GetAttr": 
            name = el["name"]
            key = get_param_name( name )
            print(name," -> ",key)
            attr = get_attr_from_model( name, model )
            if rand_flag == 0:
                text+="""
            forward_result[{i}] = new VariableTensor({k});""".format(i=i,k=key)
        
            else:
                skey = name.split("/")
                if skey[2] == "weight":
                    sh1 = attr.shape[0]
                    sh2 = attr.shape[1]
                    text+="""
            Tensor::shape_type shape = {{{sh1},{sh2}}};
            double y = sqrt(1.0/(double){sh2});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor(t);""".format(i=i,sh1=sh1,sh2=sh2)
                elif skey[2] == "bias":
                    sh1 = attr.shape[0]
                    text+="""
            Tensor::shape_type shape = {{{sh1}}};
            double y = sqrt(1.0/(double){sh1});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor(t);""".format(i=i,sh1=sh1)
        
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
            elif el["op"]=="aten::sub":     # 210701 add below mari
                text+="""
            SubOp* op = new SubOp();"""
            elif el["op"]=="aten::div":
                text+="""
            DivOp* op = new DivOp();"""
            elif el["op"]=="aten::neg":
                text+="""
            NegOp* op = new NegOp();"""
            elif el["op"]=="aten::pow":     # 210705 add mari
                text+="""
            PowOp* op = new PowOp();"""
            elif el["op"]=="aten::matmul":
                text+="""
            MatMulOp* op = new MatMulOp();"""
            elif el["op"]=="aten::addmm":
                text+="""
            AddMmOp*  op = new AddMmOp();"""
            elif el["op"]=="aten::linear":
                text+="""
            LinearOp*  op = new LinearOp();"""
            elif el["op"]=="aten::sum":
                text+="""
            SumOp*    op = new SumOp();"""
            elif el["op"]=="aten::t":
                text+="""
            TransposeOp* op = new TransposeOp();"""
            elif el["op"]=="aten::sigmoid":
                text+="""
            SigmoidOp* op = new SigmoidOp();"""
            elif el["op"]=="aten::relu":
                text+="""
            ReluOp* op = new ReluOp();"""
            elif el["op"]=="aten::softmax":
                text+="""
            SoftmaxOp* op = new SoftmaxOp();"""
            elif el["op"]=="aten::log_softmax":
                text+="""
            LogSoftmaxOp* op = new LogSoftmaxOp();"""
            elif el["op"]=="aten::tanh":
                text+="""
            TanhOp* op = new TanhOp();"""
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
        all_text += text

    all_text+="""
        cout<<"### forward computation ..."<<endl;
        forward_result[{output_id}]->forward();
        auto o = forward_result[{output_id}]->output;
        cout<<o<<endl;
    """.format(output_id=output_id)
    all_text+="""
        cout<<"### backward computation ..."<<endl;
        forward_result[{output_id}]->grad=xt::ones_like(forward_result[{output_id}]->output); // 210702 mod mari
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
