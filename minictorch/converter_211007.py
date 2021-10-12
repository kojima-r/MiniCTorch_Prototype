import json
import argparse
import numpy as np

#def makefile_generator(gen_filename):  #210719 mod mari
def makefile_generator( project ):
    make_text="""
CXX = g++
CXXFLAGS = -g -Wall  -std=c++14 -I./ -I../xtensor-blas/include -I../xtensor/include -I../xtl/include
#CXXFLAGS = -g -Wall  -fprofile-arcs -ftest-coverage -std=c++14 -I./json/include -I./xtensor-blas/include -I./xtensor/include -I./xtl/include
#LDFLAGS = -L./ -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt  -lcblas
LDFLAGS = -lcblas
CPPUTEST_HOME = ./cpputest/workspace/install
TARGET = mini_c_torch
# SRCS = main_test.cpp main.cpp
SRCS = {proj}.cpp {proj}_param.cpp
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

""".format(proj=project)
#""".format(gen_filename=gen_filename) # 210719 mod mari
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

#
def string_tensor( key, out, type=0 ): # 210719 add type mari
    
    if type == 0:
        tmp = out.to('cpu').detach().numpy().copy()
    else:
        tmp = out
    p1 = np.reshape( tmp,(-1,) )
    n1 = len(p1)
    key2 = key  # 210719 add mari
    if len(key) < 12:
        key2 = key + ' '*(12-len(key))
    s1 = 'Tensor ' + key + ' ={ '
    
    num = 8
    nw1 = n1//num
    nw2 = n1% num
    if nw2 == 0:
        nw1 = nw1 - 1
        nw2 = num
        
    #print("nw1,nw2",nw1,nw2)
    l = 0
    for k in range(nw1):
        if ( (nw1>1000) & (k>0) & (k % 5000 == 0)): # 210929 mod
            print("param:",key," - str loop ",k," / ", nw1)
        s3 = str(p1[l])  +','+str(p1[l+1])+','+str(p1[l+2])+','+str(p1[l+3])+',' \
            +str(p1[l+4])+','+str(p1[l+5])+','+str(p1[l+6])+','+str(p1[l+7])+','
        s1 = s1 + s3 + '\n' + ' '*24
        l = l+num
        #for i in range(num):
        #    #print("for k,i ",k,i)
        #    s1 = s1 + str(p1[l]) + ','
        #    l = l + 1
        #s1 = s1 + '\n' + ' '*24
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
    
# 210904 add mari
def c_data_generator( in_data ):
    
    # type declaration
    all_text="""
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    """
    
    # Data section
    s1,s2 = string_tensor( "indata", in_data, 1 )
    text="""
    // original data
        
    {ivar1};
    
    """.format(ivar1=s1)
    all_text += text
    
    return all_text
    
# 210806 mod mari
def c_param_generator( project, obj, model, input_data ):
    
    all_text="""
    //
    //  {title}_param
    //""".format(title=project)
    
    # type declaration
    all_text +="""
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    """
    
    # Data section
    s1,s2 = string_tensor( "xin", input_data )
    text="""
    // input data
        
    {ivar1};
    """.format(ivar1=s1)
    all_text += text
    
    cno = 0
    for i,el in enumerate(obj):
        name = el["name"]
        #print("name",name,el["op"])
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
    """.format(i=i,ivar1=s1,ivar2=s2)
            all_text+=text
        
        elif el["op"]=="prim::Constant":
            
            if len(el["shape"])>0:
                text="""
    // {el}
    """.format(el=str(el))
        
                cno += 1
                key ="Constant" + str(cno)
                shape=el["shape"]
                val=el["constant_value"]
                v = np.zeros(len(val))
                for k in range(len(val)):
                    v[k] = float(val[k])
                s1, s2 = string_tensor( key, v, 1 )
                text+="""
    {ivar1};
    """.format(i=i,key=key,shape=",".join(map(str,shape)), ivar1=s1)
                #print(name, key, shape[0],shape[1],val[0],val[1],len(val))
                all_text+=text
                
        elif el["op"]=="prim::Constant2": # 210714 del mari
            if len(el["shape"])>0:
                cno += 1
                key ="Constant" + str(cno)
                shape=el["shape"]
                val=el["constant_value"]
                v = np.zeros(len(val))
                for k in range(len(val)):
                    v[k] = val[k].astype(float)
                s1, s2 = string_tensor( key, v, 1 )
                text="""
        {key}= {{{val}}};
        {key}.reshape({{{shape}}});
        """.format(i=i,key=key,shape=",".join(map(str,shape)), val=",".join(map(str,val)))
                print(name, key, shape[0],shape[1],val[0],val[1],len(val))
                all_text+=text

    return all_text


# 210719 mod mari
def c_code_generator_old( obj, model, rand_flag=0 ):
    
    all_text="""
    #include <stdio.h>
    #include <iostream>
    #include <fstream>
    #include <string>
    #include <vector>
    #include "minictorch.hpp"

    using namespace std;
    
    extern void LoadParameter();
    
    extern Tensor  xin;"""  #210719 mod mari
    
    cno = 0  # 210719 add mari
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            text=""
            name = el["name"]
            key = get_param_name( name )
            text="""
    extern Tensor  {key};""".format(key=key)
            all_text+=text
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                cno += 1
                key = "Constant" + str(cno)
                text="""
    extern Tensor  {key};""".format(key=key)
                all_text+=text

    text="""

    int main()
    {
        // load parameters
        LoadParameter();
        """
    all_text += text
    
    text="""
        // input data
        VariableTensor input_var(xin);
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
    all_text += text
    
    cno = 0  # 210719 add mari
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
                val = el["constant_value"]
                text+="""
            Tensor c=(float){val};
            forward_result[{i}] = new VariableTensor(c);""".format(i=i,val=str(val))
            else:
                if len(el["shape"])>0: # 210719 add mari
                    cno += 1
                    key = "Constant" + str(cno)
                    text+="""
            forward_result[{i}] = new VariableTensor( {key} );""".format(i=i,key=key)
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
            elif el["op"]=="aten::exp":     # 210721 add mari
                text+="""
            ExpOp* op = new ExpOp();"""
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
            forward_result[{i}]=op;
            """.format(i=i)
            ###
            ### inputs
            ###
            if "in" in el and len(el["in"])>0:
                for in_id in el["in"]:
                    text+="""
            op->set_inputs( forward_result[{in_id}] );""".format(in_id=in_id)
            
            if "in" in el and len(el["in"])<0:  #210723 del mari
                num_inputs=len(el["in"])
                text+="""
            MCTNode* p_in;"""
                for in_id in el["in"]:
                    text+="""
            op->set_inputs( forward_result[{in_id}] );""".format(in_id=in_id)
            #p_in=forward_result[{in_id}];
            #op->inputs.push_back(p_in);""".format(in_id=in_id)
                in_set = sorted( list(set(el["in"])) )  # 210722 add mari
                #print("unique: ", in_set )
                for k in range(len(in_set)):
                    text+="""
            op->set_unique_inputs( forward_result[{in_id}] );""".format(in_id=in_set[k])
            
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
    
def c_code_generator( project, obj, model, rand_flag=0 ):
    
    all_text="""
    //
    //  {title}
    //""".format(title=project)
    
    all_text +="""
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;
    
    extern Tensor  xin;"""
    
    cno = 0  # Constant no.
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            text=""
            name = el["name"]
            key = get_param_name( name )
            #print(key)
            text="""
    extern Tensor  {key};""".format(key=key)
            all_text+=text
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                cno += 1
                key = "Constant" + str(cno)
                text="""
    extern Tensor  {key};""".format(key=key)
                all_text+=text
                
    text="""
    
    int main()
    {{
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
    all_text += text
    
    for i,el in enumerate(obj):  # 210728 mod
        if el["op"]=="IO Node":
            if "input" in el["name"]:
                shape=el["shape"]
                text="""
        // input data
        Tensor::shape_type shape = {{{shape}}};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        """.format(i=i,shape=",".join(map(str,shape)) )
    all_text += text
    
    cno = 0  # Constant no.
    output_id = None
    for i,el in enumerate(obj):
        print(el)
        text="""
        // {el}
        {{""".format(el=str(el))
        ###
        ### shape
        ###
        if el["op"]=="prim::GetAttr":
            pass
        elif "shape" in el and len(el["shape"])>0:  # 210920 mod or->and
            shape = el["shape"]
            shape_flat=1
            for s in el["shape"]:
                shape_flat *= s
                #print(s,shape_flat)
            text+="""
            Tensor::shape_type shape = {{{shape}}};""".format(i=i,shape=",".join(map(str,shape)) )
        ###
        ### operator
        ###
        if el["op"]=="IO Node":
            
            if "input" in el["name"]:  # 210920 mod
                text+="""
            forward_result[{i}] = &input_var;""".format(i=i)
            elif "output" in el["name"]: 
                assert len(el["in"])>0, "output error"
                output_id = el["in"][0]
            else:
                #assert False, "unknown IO:"+el["name"]
                print("unknown IO:"+el["name"])
            
            #if el["name"]=="input/x":
            #    text+="""
            #forward_result[{i}] = &input_var;""".format(i=i)
            #elif el["name"]=="output/output.1":
            #    assert len(el["in"])>0, "output error"
            #    output_id=el["in"][0]
            #else:
            #    if "input" in el["name"]:
            #        text+="""
            #forward_result[{i}] = &input_var;""".format(i=i)
            #    elif "output" in el["name"]: 
            #        assert len(el["in"])>0, "output error"
            #        output_id = el["in"][0]
            #    else:
            #        #assert False, "unknown IO:"+el["name"]
            #        print("unknown IO:"+el["name"])
        
        elif el["op"]=="prim::Constant":
            
            if "constant_value" not in el:
                text+="""
            forward_result[{i}] = NULL;""".format(i=i)
            elif len(el["shape"])==0:  # 210904 mod float->fprec, add false
                val=el["constant_value"]
                text+="""
            Tensor c = (fprec){val};
            forward_result[{i}] = new VariableTensor( c, false );""".format(i=i,val=str(val))
            else:
                if len(el["shape"])>0: # Constant no.
                    cno += 1
                    key = "Constant" + str(cno)
                    text+="""
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key} );""".format(i=i,key=key)
                else:
                    val=el["constant_value"]
                    text+="""
            Tensor t= {{{val}}};
            t = t.reshape(shape);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,shape=",".join(map(str,shape)), val=",".join(map(str,val)))
        
        elif el["op"]=="prim::GetAttr":
        
            name = el["name"]
            key = get_param_name( name )
            print(name," -> ",key)
            attr = get_attr_from_model( name, model )
            
            if rand_flag < 0:  # 210728 mod delete
                text+="""
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key)
            
            elif rand_flag == 0: # 211005mod
                skey = name.split("/")
                #print("split: ",skey)
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    #print("---",skey[2],w_len,shape)
                    #print(attr)
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key} );""".format(i=i,key=key,shape=shape)
            
            elif rand_flag == -2:  # 211005 del
                skey = name.split("/")
                #print("split: ",skey)
                w_len = len(attr.shape) # 211005 mod
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    print("---",skey[2],shape)
                if skey[2] == "weight":
                    #print('attr',len(attr.shape))
                    #w_len = len(attr.shape)
                    #print(attr)
                    if w_len == 2:  # 210929 add
                        sh1 = attr.shape[0]
                        sh2 = attr.shape[1]
                        text+="""
            Tensor::shape_type shape = {{{sh1},{sh2}}};
            {k}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1,sh2=sh2)
                    elif w_len == 1: # 210929 add
                        sh1 = attr.shape[0]
                        text+="""
            Tensor::shape_type shape = {{{sh1}}};
            {k}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1)
                    elif w_len == 4: # 210929 mod
                        sh1 = attr.shape[0]
                        sh2 = attr.shape[1]
                        sh3 = attr.shape[2]
                        sh4 = attr.shape[3]
                        text+="""
            Tensor::shape_type shape = {{{sh1},{sh2},{sh3},{sh4}}};
            {k}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1,sh2=sh2,sh3=sh3,sh4=sh4)
                elif skey[2] == "bias":
                    sh1 = attr.shape[0]
                    text+="""
            Tensor::shape_type shape = {{{sh1}}};
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1)
                elif skey[2] == "running_mean":  # 210929 add
                    sh1 = attr.shape[0]
                    text+="""
            Tensor::shape_type shape = {{{sh1}}};
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1)
                elif skey[2] == "running_var":  # 210929 add
                    sh1 = attr.shape[0]
                    text+="""
            Tensor::shape_type shape = {{{sh1}}};
            forward_result[{i}] = new VariableTensor( {k} );""".format(i=i,k=key,sh1=sh1)
            
            elif rand_flag == 3:  # 211005 del
                skey = name.split("/")
                #print("split: ",skey)
                w_len = len( attr.size() )
                if skey[2] == "weight":
                    if w_len == 2:  # 210929 add
                        sh1 = attr.shape[0]
                        sh2 = attr.shape[1]
                        text+="""
            Tensor::shape_type shape = {{{sh1},{sh2}}};
            fprec y = sqrt(1.0/(fprec){sh2});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,sh1=sh1,sh2=sh2)
                    elif w_len == 1: # 210929 add
                        sh1 = attr.shape[0]
                        text+="""
            Tensor::shape_type shape = {{{sh1}}};
            fprec y = sqrt(1.0/(fprec){sh1});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,sh1=sh1)
                    elif w_len == 4: # 210929 mod
                        sh1 = attr.shape[0]
                        sh2 = attr.shape[1]
                        sh3 = attr.shape[2]
                        sh4 = attr.shape[3]
                        text+="""
            Tensor::shape_type shape = {{{sh1},{sh2},{sh3},{sh4}}};
            fprec y = sqrt(1.0/(fprec){sh4});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,sh1=sh1,sh2=sh2,sh3=sh3,sh4=sh4)
                elif skey[2] == "bias":
                    sh1 = attr.shape[0]
                    text+="""
            Tensor::shape_type shape = {{{sh1}}};
            fprec y = sqrt(1.0/(double){sh1});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,sh1=sh1)
                elif skey[2] == "running_mean":  # 210929 add
                    sh1 = attr.shape[0]
                    text+="""
            Tensor t = xt::zeros<fprec>({sh1});
            forward_result[{i}] = new VariableTensor( t );""".format(i=i)
                elif skey[2] == "running_var":  # 210929 add
                    sh1 = attr.shape[0]
                    text+="""
            Tensor t = xt::ones<fprec>({sh1});
            forward_result[{i}] = new VariableTensor( t );""".format(i=i)
            
            else:
                skey = name.split("/")
                #print("split: ",skey)
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    print("---",skey[2],w_len,shape)
                    shy = attr.shape[w_len-1]
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            fprec y = sqrt(1.0/(fprec){shy});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t );""".format(i=i,shape=shape,shy=shy)
                
        else:
            ###
            ### standard operators
            ###
            out_id = el["output_id"]  # 2101004 add
            
            if el["op"]=="aten::mul":
                text+="""
            MulOp* op = new MulOp();"""
            elif el["op"]=="aten::add":
                text+="""
            AddOp* op = new AddOp();"""
            elif el["op"]=="aten::sub":
                text+="""
            SubOp* op = new SubOp();"""
            elif el["op"]=="aten::rsub":
                text+="""
            RsubOp* op = new RsubOp();"""
            elif el["op"]=="aten::div":     
                text+="""
            DivOp* op = new DivOp();"""
            elif el["op"]=="aten::neg":     
                text+="""
            NegOp* op = new NegOp();"""
            elif el["op"]=="aten::pow":
                text+="""
            PowOp* op = new PowOp();"""
            elif el["op"]=="aten::exp":
                text+="""
            ExpOp* op = new ExpOp();"""
            elif el["op"]=="aten::log":
                text+="""
            LogOp* op = new LogOp();"""
            elif el["op"]=="aten::matmul":
                text+="""
            MatMulOp* op = new MatMulOp();"""
            elif el["op"]=="aten::addmm":
                text+="""
            AddMmOp*  op = new AddMmOp();"""
            elif el["op"]=="aten::linear":
                text+="""
            LinearOp* op = new LinearOp();"""
            elif el["op"]=="aten::sum":
                text+="""
            SumOp*    op = new SumOp();"""
            elif el["op"]=="aten::t":
                text+="""
            TransposeOp* op = new TransposeOp();"""
            elif el["op"]=="aten::max":  # 210729 add yet
                text+="""
            MaxOp* op = new MaxOp();"""
            elif el["op"]=="aten::min":  # 210729 add yet
                text+="""
            MinOp* op = new MinOp();""" 
            elif el["op"]=="aten::sigmoid":
                text+="""
            SigmoidOp* op = new SigmoidOp();"""
            elif el["op"]=="aten::relu":
                text+="""
            ReluOp* op = new ReluOp();"""
            elif el["op"]=="aten::elu":
                text+="""
            EluOp* op = new EluOp();"""
            elif el["op"]=="aten::leaky_relu":
                text+="""
            LeakyReluOp* op = new LeakyReluOp();"""
            elif el["op"]=="aten::hardtanh":
                text+="""
            HardTanhOp* op = new HardTanhOp();"""
            elif el["op"]=="aten::softplus":  # 211004 add
                text+="""
            SoftplusOp* op = new SoftplusOp();"""
            elif el["op"]=="aten::softmax":
                text+="""
            SoftmaxOp* op = new SoftmaxOp();"""
            elif el["op"]=="aten::log_softmax":
                text+="""
            LogSoftmaxOp* op = new LogSoftmaxOp();"""
            elif el["op"]=="aten::tanh":
                text+="""
            TanhOp* op = new TanhOp();"""
            elif el["op"]=="aten::randn":
                text+="""
            RandnOp* op = new RandnOp();"""
            elif el["op"]=="aten::normal":
                text+="""
            NormalOp* op = new NormalOp();"""
            elif el["op"]=="aten::batch_norm":  #211004 add
                text+="""
            BatchNormOp* op = new BatchNormOp();"""
            elif el["op"]=="aten::dropout":  #211004 add
                text+="""
            DropoutOp* op = new DropoutOp();"""
            elif el["op"]=="aten::mse_loss":
                text+="""
            MseLossOp* op = new MseLossOp();"""
            elif el["op"]=="aten::cross_entropy_loss":
                text+="""
            CrossEntropyLossOp* op = new CrossEntropyLossOp();"""
            elif el["op"]=="aten::binary_cross_entropy":
                text+="""
            BCELossOp* op = new BCELossOp();"""
            elif el["op"]=="aten::nll_loss_nd":  #210824 yet
                text+="""
            NLLLossOp* op = new NLLLossOp();"""
            elif el["op"]=="aten::size":
                text+="""
            SizeOp* op = new SizeOp();"""
            elif el["op"]=="aten::zeros":
                text+="""
            ZerosOp* op = new ZerosOp();"""
            elif el["op"]=="aten::zeros_like":
                text+="""
            FullLikeOp* op = new FullLikeOp( 0.0 );"""
            elif el["op"]=="aten::ones":
                text+="""
            OnesOp* op = new OnesOp();"""
            elif el["op"]=="aten::ones_like":
                text+="""
            FullLikeOp* op = new FullLikeOp( 1.0 );"""
            elif el["op"]=="aten::expand":
                text+="""
            ExpandOp* op = new ExpandOp();"""
            elif el["op"]=="prim::NumToTensor":
                text+="""
            MoveOp* op = new MoveOp( "NumToTensor" );"""
            elif el["op"]=="aten::Int":
                text+="""
            MoveOp* op = new MoveOp( "Int" );"""
            elif el["op"]=="aten::view":  # 211004 add
                text+="""
            ViewOp* op = new ViewOp();"""
            elif el["op"]=="aten::broadcast_tensors":
                text+="""
            BroadcastTensorsOp* op = new BroadcastTensorsOp();"""
            elif el["op"]=="aten::to":
                text+="""
            ListConstructOp* op = new ListConstructOp( "to" );"""
            elif el["op"]=="aten::detach":
                text+="""
            ListUnpackOp* op = new ListUnpackOp( "detach", {k} );""".format(k=out_id)
            elif el["op"]=="prim::ListConstruct":
                text+="""
            ListConstructOp* op = new ListConstructOp();"""
            elif el["op"]=="prim::ListUnpack":
                text+="""
            ListUnpackOp* op = new ListUnpackOp( {k} );""".format(k=out_id)
            else:
                #assert False, "unknown op:"+el["op"]
                text+="""
            AddOp* op = NULL;"""
                print("unknown op:"+el["op"])
            ### setting operator
            text+="""
            forward_result[{i}] = op;
            """.format(i=i)
            ###
            ### inputs
            ###
            if "in" in el and len(el["in"]) > 0:
                num_inputs=len(el["in"])
                for in_id in el["in"]:
                    text+="""
            op->set_inputs( forward_result[{in_id}] );""".format(in_id=in_id)
            
            if "in" in el and len(el["in"]) < 0:  # 210714 del mari
                num_inputs=len(el["in"])
                text+="""
            MCTNode* p_in;"""
                for in_id in el["in"]:
                    text+="""
            p_in = forward_result[{in_id}];
            op->inputs.push_back( p_in );""".format(in_id=in_id)
                
        text+="""
        }
        """
        all_text+=text

    # 210806 mod forward, backward
    all_text+="""
        cout<<"### forward computation ..."<<endl;
        //forward_result[{output_id}]->forward();
        for(int k=0;k<={output_id};k++) {{
            if( forward_result[k] )  
            {{
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }}
        }}
        auto o = forward_result[{output_id}]->output;
        cout<<o<<endl;
    """.format(output_id=output_id)
    
    all_text+="""
        cout<<"### backward computation ..."<<endl;
        forward_result[{output_id}]->grad = xt::ones_like( forward_result[{output_id}]->output );
        //forward_result[{output_id}]->backward();
        for(int k={output_id};k>=0;k--) {{
           if( forward_result[k] )  forward_result[k]->backward();
        }}
        cout<<"input_grad"<<input_var.grad<<endl;
    """.format(output_id=output_id)
    
    all_text+="""
        return 0;
    }
    """
    return all_text
    
    
# convert json file to parameter, cpp, make file
def convert_json( project, folder, model, input_x, json_path, rand_flag=0 ):

    cpp_fname   = project + ".cpp"
    param_fname = project + "_param.cpp"
    cpp_path    = folder + "/" + cpp_fname
    param_path  = folder + "/" + param_fname
    make_path   = folder + "/" + "Makefile"
  
    # save json file
    print( "[JSON]", json_path )
    fp = open( json_path )
    obj = json.load(fp)

    # save parameter file
    code1 = c_param_generator( project, obj, model, input_x )
    if len(code1) > 0:
       print( "[PARAM]", param_path )
       ofparam = open( param_path, "w" )
       ofparam.write( code1 )
    else:
       param_fname=""

    #save cpp file
    code2 = c_code_generator( project, obj, model, rand_flag )

    print("[CPP] ", cpp_path )
    ofp = open( cpp_path, "w" )
    ofp.write( code2 )

    # save make file
    make_code = makefile_generator( project )

    print( "[MAKE]", make_path )
    makefp = open( make_path, "w" )
    makefp.write( make_code )



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
