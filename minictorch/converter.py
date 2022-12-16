import json
import argparse
import numpy as np
import os
import torch
import re

OP_MAPPING={
    "aten::mul":"MulOp",
    "aten::add":"AddOp",
    "aten::sub":"SubOp",
    "aten::rsub":"RsubOp",
    "aten::div":"DivOp",
    "aten::neg":"NegOp",
    "aten::pow":"PowOp",
    "aten::exp":"ExpOp",
    "aten::log":"LogOp",
    "aten::log1p":"Log1pOp",
    "aten::dot":"DotOp",
    "aten::matmul":"MatMulOp",
    "aten::addmm":"AddMmOp",
    "aten::linear":"LinearOp",
    "aten::sum":"SumOp",
    "aten::mean":"MeanOp",
    "aten::stack":"StackOp",
    "aten::select":"SelectOp",
    "aten::copy_":"Copy_Op",
    "aten::t":"TransposeOp",
    "aten::max":"MaxOp",
    "aten::min":"MinOp",
    "aten::sigmoid":"SigmoidOp",
    "aten::relu":"ReluOp",
    "aten::elu":"EluOp",
    "aten::leaky_relu":"LeakyReluOp",
    "aten::hardtanh":"HardTanhOp",
    "aten::softplus":"SoftplusOp",
    "aten::softmax":"SoftmaxOp",
    "aten::log_softmax":"LogSoftmaxOp",
    "aten::tanh":"TanhOp",
    "aten::randn":"RandnOp",
    "aten::normal":"NormalOp",
    "aten::batch_norm":"BatchNormOp",
    "aten::dropout":"DropoutOp",
    "aten::mse_loss":"MseLossOp",
    "aten::cross_entropy_loss":"CrossEntropyLossOp",
    "aten::binary_cross_entropy":"BCELossOp",
    "aten::nll_loss_nd":"NLLLossOp",
    "aten::size":"SizeOp",
    "aten::zeros":"ZerosOp",
    "aten::zeros_like":"ZerosLikeOp",
    "aten::ones":"OnesOp",
    "aten::ones_like":"OnesLikeOp",
    "aten::expand":"ExpandOp",
    "prim::NumToTensor":"NumToTensorOp",
    "aten::Int":"IntOp",
    "aten::view":"ViewOp",
    "aten::broadcast_tensors":"BroadcastTensorsOp",
    "aten::to":"ToOp",
    "aten::detach":"DetachOp",
    "prim::ListConstruct":"ListConstructOp",
    "prim::ListUnpack":"ListUnpackOp",
    "prim::TupleConstruct":"TupleConstructOp",
    "prim::TupleUnpack":"TupleUnpackOp",
    }

OUTPUT_ID_ENABLED_NODE={
    "DetachOp",
    "ListUnpackOp",
    "TupleUnpackOp",
    }
 
def makefile_generator( project, code="all", xtensor_include_base="../", minictorch_include="../src",optimize="-O3",libs="-lmkl_rt"): #libs="-lcblas"):
    make_text="""
CXX = g++
CXXFLAGS += {optimize} -Wall  -std=c++14 -I./ -I{minictorch_inc} -I{xtensor_base}xtensor-blas/include -I{xtensor_base}xtensor/include -I{xtensor_base}xtl/include
LDFLAGS = {libs}
TARGET  = {proj}
SRCS    = {proj}.cpp {proj}_param.cpp
OBJS    = $(SRCS:.cpp=.o)

""".format(proj=project, xtensor_base=xtensor_include_base, minictorch_inc=minictorch_include, optimize=optimize, libs=libs)

    if code == "all":
        make_text+="""
TRAIN_SRCS    = {proj}.cpp {proj}_param.cpp {proj}_train.cpp {proj}_data.cpp
TRAIN_TARGET  = {proj}_train
TRAIN_OBJS    = $(TRAIN_SRCS:.cpp=.train.o)

all: $(TARGET) $(TRAIN_TARGET)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) -o $@ $^ -D_TRAIN $(CXXFLAGS) $(LDFLAGS)
""".format(proj=project)
    else:
        make_text+="""
all: $(TARGET)
"""

    make_text+="""
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

"""  #.format(proj=project,xtensor_base=xtensor_include_base,minictorch_inc=minictorch_include)
    return make_text




# s:     attribute name in computational graph e.g. Net/Linear[fc3]/bias/bias 
# model: pytorch model
# Return:  pytorch parameters  <class 'torch.nn.parameter.Parameter'>
#
def get_attr_from_model( s, model ):
    pat = re.compile(r'([^\[\]]*)\[(.*)\]')
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
    

# ex. Net/Linear[fc1]/weight/43 -> param_fc1_weight
#     VAE/Net[net]/Linear[fc1]/weight/158 -> param_fc1_weight
def get_param_name( s1 ):
    s2 = s1.split('/')
    n = 1
    for i in range(len(s2)):
        if s2[i].find("[")>=0:
            n = i
    s3 = re.findall("(?<=\[).+?(?=\])", s2[n])
    s4 = s3[0] + '_' + s2[n+1]
    return "param_"+s4
    
# 
# fc3_bias
# pytorch tensor: tensor([0.0402, 0.0524, 0.0345], requires_grad=True)
#
#=> result: Tensor fc3_bias ={ 0.040239427,0.052421827,0.034500692, }
#   result_reshape: fc3_bias.reshape({3})
# 
def generate_inline_tensor_c_code( key, out ):
    num_per_line = 8
    n_indent = 24
    
    if torch.is_tensor(out):
        tmp = out.to('cpu').detach().numpy().copy()
    else:
        tmp = out
    flatten_tmp = np.reshape( tmp,(-1,) ) # flatten
    
    ## generating result line
    n1 = len(flatten_tmp)
    result = 'Tensor ' + key + ' ={ '
    
    n_line = n1//num_per_line
    n_remain = n1% num_per_line
    if n_remain == 0:
        n_line -= 1
        n_remain = num_per_line
        
    l = 0
    for k in range(n_line):
        ## progress message
        if ( (n_line>1000) & (k>0) & (k % 10000 == 0)):
            print("param:",key," - str loop ",k," / ", n_line)

        result += ",".join(map(str,flatten_tmp[l:l+num_per_line]))+',\n'
        result += ' '*n_indent ## indent
        l = l+num_per_line
    if n_remain > 0:
        result += ",".join(map(str,flatten_tmp[l:]))+ ','
        print("param:",key," - str loop ",n_line," / ", n_line)
    result += ' }'
    #print("tensor :",result)
    
    ## generating result_reshape line
    n2 = len( tmp.shape )
    if n2 == 0:
        result_reshape = ""
    else:
        result_reshape = key + '.reshape({'
        s=",".join(map(str,list(tmp.shape)))
        result_reshape += s + '})'
    #print("shape :",result_shape)
    return result, result_reshape
    

# export all input data
def generate_data_c_code( **pair_data ):
    if len(pair_data) < 1:
        return ""
    # type declaration
    all_text="""
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    """
    
    # Data section
    for key,val in pair_data.items():
        print("generating data: ", key)
        s1,s2 = generate_inline_tensor_c_code( key, val)
        text="""
        // data
        
        {ivar1};
    
        """.format(ivar1=s1)
        all_text += text
    
    return all_text
    

def get_one_line(indent,s):
    return "\n"+"    "*indent+s+"\n"


def generate_input_and_param_c_code( obj, model, input_data ):
    
    # type declaration
    all_text ="""
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    """
    
    # Data section
    s1,_ = generate_inline_tensor_c_code( "xin", input_data )
    text="""
    // input data
        
    {ivar1};
    """.format(ivar1=s1)
    all_text += text
    
    key_list=[];
    n_constant = 0
    print("... parameters")
    for i,el in enumerate(obj):
        name = el["name"]
        #print("name",name,el["op"])
        if el["op"]=="prim::GetAttr":
            print(i,el)
            text = get_one_line(1,"// {el}").format(el=str(el))
            name = el["name"]
            key = get_param_name( name )
            if( key not in key_list ):
                key_list.append( key )
                attr = get_attr_from_model( name, model )
                s1, _ = generate_inline_tensor_c_code( key, attr )
                text+=get_one_line(1,"{ivar1};").format(i=i,ivar1=s1)
                all_text+=text
        
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                print(i,el)
                text = get_one_line(1,"// {el}").format(el=str(el))
                n_constant += 1
                key ="Constant" + str(n_constant)
                shape=el["shape"]
                val=el["constant_value"]
                v = np.array( val )
                s1, _ = generate_inline_tensor_c_code( key, v )
                text+=get_one_line(1,"{ivar1};").format(i=i,ivar1=s1)
                all_text+=text
    print("...")
    return all_text

    
def generate_graph_c_code(obj, model, chk_shape=0, rand_flag=0 ):
    """
    print("... computational graph")
    for i,el in enumerate(obj):
        print(i,el)
    print("...")
    """

    all_text ="""
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include "minictorch.hpp"
    
    using namespace std;
    
    extern Tensor  xin;"""
    
    key_list = []
    n_constant = 0  # Constant no.
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            text=""
            name = el["name"]
            key = get_param_name( name )
            if( key not in key_list ):
                key_list.append( key )
                all_text += get_one_line(1,"extern Tensor {key};").format(key=str(key))
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                n_constant += 1
                key = "Constant" + str(n_constant)
                all_text += get_one_line(1,"extern Tensor {key};").format(key=str(key))
    all_text +="""
    
    bool train_mode = true;
    
    void build_computational_graph( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {"""
    
    n_constant = 0  # Constant no.
    output_id = None
    for i,el in enumerate(obj):
        text="""
        // {el}
        {{""".format(el=str(el))
        ###
        ### shape
        ###
        if el["op"]=="prim::GetAttr":
            pass
        elif "shape" in el and len(el["shape"])>0:
            shape = el["shape"]
            shape_flat=1
            for s in el["shape"]:
                shape_flat *= s
            text+="""
            Tensor::shape_type shape = {{{shape}}};""".format(i=i,shape=",".join(map(str,shape)) )
        ###
        ### operator
        ###
        if el["op"]=="IO Node":
            
            if "input" in el["name"]:
                text+="""
            forward_result[{i}] = &input_var;""".format(i=i)
            
            elif "output" in el["name"]: 
                assert len(el["in"])>0, "output error"
                output_id = el["in"][0]
            else:
                print("[WARNING] unknown IO:"+el["name"])
        ###
        ### constant
        ###
        elif el["op"]=="prim::Constant":
            if "constant_value" not in el:
                text+="""
            forward_result[{i}] = NULL;""".format(i=i)
            elif len(el["shape"]) == 0:
                val = el["constant_value"]
                text+="""
            Tensor c = (fprec){val};
            forward_result[{i}] = new VariableTensor( c, VAR_CONST );""".format(i=i,val=str(val))
            else:
                if len(el["shape"]) > 0: # Constant no. ## from extern variable
                    n_constant += 1
                    key = "Constant" + str(n_constant)
                    text+="""
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key}, VAR_CONST );""".format(i=i,key=key)
                else:
                    val=el["constant_value"]
                    text+="""
            Tensor t= {{{val}}};
            t = t.reshape(shape);
            forward_result[{i}] = new VariableTensor( t, VAR_CONST );""".format(i=i,shape=",".join(map(str,shape)), val=",".join(map(str,val)))
        
        ###
        ### attr
        ###
        elif el["op"]=="prim::GetAttr":
        
            name = el["name"]
            key = get_param_name( name )
            print(name," -> ",key)
            attr = get_attr_from_model( name, model )
            #print(attr)
            
            if rand_flag == 0:
                skey = name.split("/")
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key}, VAR_ATTR );""".format(i=i,key=key,shape=shape)
            
            else:
                skey = name.split("/")
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    shy = attr.shape[w_len-1]
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            fprec y = sqrt(1.0/(fprec){shy});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t, VAR_ATTR );""".format(i=i,shape=shape,shy=shy)
                
        else:
            ###
            ### standard operators
            ###
            out_id = el["output_id"]
            op=el["op"]
            if op in OP_MAPPING:
                if OP_MAPPING[op] in OUTPUT_ID_ENABLED_NODE:
                    s="{cls}* op = new {cls}({out_id});".format(cls=OP_MAPPING[op],out_id=out_id)
                else:
                    s="{cls}* op = new {cls}();".format(cls=OP_MAPPING[op],out_id=out_id)
                text+=get_one_line(3,s)
            else:
                #assert False, "unknown op:"+el["op"]
                text+=get_one_line(3,"AddOp* op = NULL;")
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
            
            if chk_shape > 0:
                if "shape" in el and len(el["shape"]) > 0:
                    text +="""
            op->set_shape( shape );"""
            
        text+="""
        }
        """
        all_text+=text
        
    all_text +="""
    }
    """
    return all_text

def generate_main_c_code(obj, seed_no=-1, chk_shape=0):
        
    for i,el in enumerate(obj):
        if el["op"]=="IO Node":
            if "input" in el["name"]:
                input_i=i
                input_shape=el["shape"]
            elif "output" in el["name"]: 
                assert len(el["in"])>0, "output error"
                output_id = el["in"][0]

    # 1 forward / backward
    all_text ="""
    void do_forward_backward_test( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N )
    {
        cout<<"### forward computation ..."<<endl;
        for(int k=0;k<=N;k++) {
            if( forward_result[k] )  
            {"""
    if chk_shape > 0:
        all_text+="""
                forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->display_shape();
                forward_result[k]->zerograd();"""
    else:
        all_text+="""
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();"""
    all_text +="""
            }
        }
        auto o = forward_result[N]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[N]->grad = xt::ones_like( forward_result[N]->output );
        for(int k=N;k>=0;k--) {"""
    if chk_shape > 0:
        all_text +="""
            if( forward_result[k] )  
            {
               forward_result[k]->backward();
               forward_result[k]->display_grad_shape();
            }"""
    else:
        all_text +="""
            if( forward_result[k] )  forward_result[k]->backward();"""
    all_text +="""
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    }
    
    """
    
    # main program
    all_text +="""
    #ifdef _TRAIN
    extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N );
    #endif
    
    int main()
    {{
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
    text="""
        // input data:  forward_result[{i}]
        Tensor::shape_type shape = {{{shape}}};
        xin.reshape( shape );
        VariableTensor input_var( xin, VAR_INPUT );
    """.format(i=input_i,shape=",".join(map(str,input_shape)))
    all_text += text
    
    if seed_no >= 0:
        text ="""
        xt::random::seed( {ns} );
    """.format(ns=seed_no)
        all_text += text
    
    all_text +="""
        build_computational_graph( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, {output_id} );
    #else
        do_forward_backward_test( forward_result, input_var, {output_id} );
    #endif
        
        return 0;
    }}
    """.format(output_id=output_id)
    
    return all_text


def unpack_origin_no( obj, target_index ):
    target_node = obj[target_index]
    in_index = target_node["in"][0]
    in_node = obj[in_index]
    out_id = target_node['output_id']
    #print( "unpack",el2["op"], out_id)
    result_index  = target_index
    if in_node["op"] in ['prim::ListConstruct','prim::TupleConstruct']:
        result_index = in_node['in'][out_id]
    elif in_node["op"] == 'aten::broadcast_tensors':
        el3 = obj[in_node['in'][0]]
        if el3["op"] == 'prim::ListConstruct':
            result_index = el3['in'][out_id]
    #print("unpack original no: ",no,no1)
    return result_index
    
def get_unpack_origin( obj, no1, inout ):
    no = no1
    while True:
        el = obj[no]
        if el["op"] in ['prim::ListUnpack','prim::TupleUnpack']:
            no = unpack_origin_no( obj, no ) ## unpackに対応したConstructを見つけようとする
        elif el["op"] == 'prim::ListConstruct':
            for j in range(len(el['in'])):
                k = get_unpack_origin( obj, el['in'][j], inout )
                if inout[k] == 1:
                    return k
            return no1
        else:
            break
    return no

def get_tensor_shape( x ):
    num = len( x.shape )
    text = ""
    if num > 0:# tensor
        text = "{"
        text += ",".join(map(str,x.shape))
        text +="}"
        return True, text
    return False, text

def find_output_node(obj,net_key,loss_key,pred_key,output_id,target_enabled,pred_index, task_type):
    # evaluated no
    pred_no   = -1
    target_no = -1
    pred_pos  = -1
    class_no  = -1
    
    pos1 = 0
    pos2 = 0
    for i,el in enumerate(obj):
        if net_key in el["name"]: # last Net
            pos1 = i+1
        if loss_key in el["name"]: # first Loss
            pos2 = i
            break
    if pos1 > output_id:
        pos1 = 0

    if ( pos1 > 0 ) and ( pos1 < pos2 ):
        pred_pos = pos1
    elif pos2 > 0:
        pred_pos = pos2
    if pos1 > pos2:
        pred_pos = -1

    print("prediction output node index: ",pred_pos)
    print("Net class output node index: ",pos1)
    print("Loss class input node index: ",pos2)
    print("output node index: ",output_id)
    
    # inout option ( 0:input only, 1:input and output both )
    inout = np.zeros( output_id+1, dtype=np.int )
    inout[0] = 1
    for i in range(1,output_id+1):
        el = obj[i]
        if   el['op'] == "prim::Constant":  pass
        elif el['op'] == "aten::GetAttr":   pass
        elif el['op'] == "prim::ListConstruct": pass
        elif el['op'] == "prim::ListUnpack":    pass
        elif el['op'] == "aten::broadcast_tensors":  pass
        elif el['op'] == "prim::TupleConstruct":  pass
        elif el['op'] == "prim::TupleUnpack":  pass
        else:
            nin = 0;
            for j in range(len(el['in'])):
                k = get_unpack_origin( obj, el['in'][j], inout ) # 220203 inout add
                if inout[k] == 1:
                    nin=nin+1
            if nin > 0:
                inout[i] = 1
            
    last = -1
    pred_max = -1
    pred_id  = -1
    for i in range(output_id+1):
        el = obj[i]
        #print("el ",i,inout[i], " : ", el['name'],el['op'],el['in'],el['output_id'])
        if inout[i] > 0:
            print("el ",i,"inout=",inout[i], " : ", el['name'],el['op'],"in=",el['in'],"output_id=",el['output_id'])
            if last < i: last = i
            if pred_pos < 0:  
                pass
            elif i >= pred_pos:
                for j in range(len(el['in'])):
                    k = get_unpack_origin( obj, el['in'][j], inout )
                    if ( k < pred_pos ) and ( inout[k] > 0 ):
                        print(" --- pred el (",j,") :",el['in'][j]," -> ",k)
                        if pred_max < k: 
                            pred_max = k
                            pred_id  = i
    ## pred_idはここまでで決定
    print("pred_id",pred_id,pred_key)

    if pred_id > 0:
        el = obj[pred_id]
        type = 0
        if   el['op'] == 'aten::mse_loss':  type = 1
        elif el['op'] == 'aten::cross_entropy_loss':    type = 2
        elif el['op'] == 'aten::binary_cross_entropy':  type = 2
        elif el['op'] == 'aten::nll_loss_nd':  type = 3
        else:  type = 4
        print("loss pred_id : ",pred_id,pred_max)
        if type > 0:
            pred_no = get_unpack_origin( obj, el['in'][0], inout )
            if len(el['in']) > 1: 
                no2 = get_unpack_origin( obj, el['in'][1], inout )
                if target_enabled > 0: target_no = no2
                if type == 2:      class_no  = no2
                if type == 3:      class_no  = no2
        if pred_no > 0:  
            print("eval1 no :",i," (type=",type,") : ", pred_no,target_no)
            
    if pred_pos < 0:
        if pred_key is not None:
            for i,el in enumerate(obj):
                if pred_key in el['op']:
                    if i > pred_no:  pred_no = i
        else:
            type = 0
            for i,el in enumerate(obj):
                if   el['op'] == 'aten::mse_loss':  type = 1
                elif el['op'] == 'aten::cross_entropy_loss':    type = 2
                elif el['op'] == 'aten::binary_cross_entropy':  type = 2
                elif el['op'] == 'aten::nll_loss_nd':  type = 3
                if type > 0:
                    pred_no = get_unpack_origin( obj, el['in'][0], inout )
                    if len(el['in']) > 1:
                        no2 = get_unpack_origin( obj, el['in'][1], inout )
                        if target_enabled > 0: target_no = no2
                        if type == 2:      class_no  = no2
                        if type == 3:      class_no  = no2
                if pred_no > 0:  
                    if type == 3: pred_no = -1
                    print("eval2 no :",i,"-> type=",type," : ",pred_no,target_no)
                    break
            
        
    print("last cmd:", last) #未使用
    print("------")
    if not "classification" in task_type:
        class_no = -1

    if ( pred_index >= 0 ) and ( pred_index < output_id ):
        if inout[pred_index] > 0:
            pred_no = pred_index

    return pred_no,target_no,class_no

from jinja2 import Template, Environment, FileSystemLoader

def c_train_code_generator( project, folder, obj,
        train_tmpl='train.tmpl.cpp',
        epochs = 200,
        batch_size = 0,
        lr = 0.01,
        net_key = "Net",
        loss_key = "Loss",
        pred_key = None,
        pred_index = -1,
        input_data=None,
        target_data=None,
        shuffle = False,
        pred_output = None,
        task_type = "",
        latent_z=None,
        **kwargs 
        ):
    
    # arguments
    print("task_type :", task_type)
    print("epoch_num : ", epochs )
    print("batch_size : ", batch_size )
    print("lr : ", lr )
    print("net_key :", net_key )
    print("loss_key :", loss_key )    
    print("pred_key :", pred_key )
    print("pred_index :", pred_index )
    print("shuffle : ", shuffle )
        
    input_enabled = False;
    input_s = ""
    if input_data is not None:
        input_enabled, input_s = get_tensor_shape( input_data )
        
    target_enabled = False
    target_s = ""
    if target_data is not None:
        target_enabled, target_s = get_tensor_shape( target_data )
        
    print("input  shape : ", input_enabled, input_s)
    print("target shape : ", target_enabled, target_s)
    
    #pred_output
    if pred_output is None:
        if input_enabled:
            pred_output = input_data.shape[0]
        else:
            pred_output=0
    print("pred_output : ", pred_output)

    # ---------- 
    # output id
    output_id = -1
    for i,el in enumerate(obj):
        if "output" in el['name']:
            assert len(el["in"])>0, "output error"
            output_id = el["in"][0]

    ## e.g. Loss = loss1 + loss2
    output_sum_loss_id=[]
    if output_id >=0:
        el = obj[output_id];
        if el["op"] == "aten::add":  # for vae
            nadd1 = el['in'][0]
            nadd2 = el['in'][1]
            output_sum_loss_id=[nadd1,nadd2]
            print("index list for Loss components : ", output_sum_loss_id)
    ###
    pred_no,target_no,class_no = find_output_node(obj,net_key,loss_key,pred_key,output_id,target_enabled,pred_index, task_type)
    print("pred_no   :", pred_no, obj[pred_no])
    print("target_no :", target_no, obj[target_no])
    print("class_no  :", class_no, obj[class_no])
    ###
    x_shape=""
    y_shape=""
    x_shape_rest=""
    y_shape_rest = ""
    x_shape_rest_enabled = False
    y_shape_rest_enabled = False
    if batch_size > 0:
        if input_enabled:
            ds = input_data.shape
            x_shape = ",".join([str(batch_size)]+[str(ds[k]) for k in range(1,input_data.ndim)])
        
        if target_enabled:
            ds = target_data.shape
            y_shape = ",".join([str(batch_size)]+[str(ds[k]) for k in range(1,target_data.ndim)])

        if input_enabled:
            ds = input_data.shape
            sz = input_data.ndim
            if sz>=2:
                x_shape_rest_enabled = True
                for k in range(2,sz):
                    x_shape_rest += ", xt::all()"
        if target_enabled:
            ds = target_data.shape
            sz = target_data.ndim
            if sz>=2:
                y_shape_rest_enabled = True
                for k in range(2,sz):
                    y_shape_rest += ", xt::all()"
         
    double_loss_enabled= len(output_sum_loss_id) > 0 and class_no < 1  # Loss=Loss1+Loss2
 
    pred_type = 1  ###
    if input_enabled:
        pred_type = len(input_data.shape)
    print("pred_type : ",pred_type)

    pred_op = ""  
    if pred_no>=0:
        el = obj[pred_no]
        pred_op=el['op']

    # latent variable
    z_no  = -1
    if latent_z is not None:
        for i,el in enumerate(obj):
            if el["op"] == "aten::linear":
                if latent_z in el['name']:
                    z_no = el['in'][0]
                    print("vae z: ", z_no," keyword :",  latent_z)
            
    ###
    path=os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
    #train_tmpl = env.get_template('train.tmpl.cpp')
    train_tmpl = env.get_template(train_tmpl)
    params={
        "proj":project,
        "input_enabled":input_enabled,
        "target_enabled":target_enabled,
        "eval_enabled":class_no >= 0,
        "classification_task":class_no >= 0,
        "target_no":target_no,
        "pred_no":pred_no,
        "epochs":epochs,
        "lr":lr,
        "input_shape":input_s,
        "target_shape":target_s,
        "batch_enabled":batch_size>0,
        "batch_size":batch_size,
        "x_shape":x_shape,
        "y_shape":y_shape,
        "x_shape_rest":x_shape_rest,
        "x_shape_rest_enabled": x_shape_rest_enabled,
        "y_shape_rest":y_shape_rest,
        "y_shape_rest_enabled": y_shape_rest_enabled,
        "double_loss_enabled":double_loss_enabled,
        "output_filename":folder + '/' + project + ".out",
        "pred_type":pred_type,
        "pred_op":pred_op,
        "pred_output":pred_output,
        "pred_filename":folder + '/' + project + ".pred",
        "z_filename": folder + '/' + project + ".z",
        "z_no":z_no,
        }
    if len(output_sum_loss_id)>=2:
        params["loss1_no"]=output_sum_loss_id[0]
        params["loss2_no"]=output_sum_loss_id[1]
    print(params)
    rendered_s = train_tmpl.render(params)
    print(rendered_s)
       
    return rendered_s


# convert json file to parameter, cpp, and make file
def convert_cpp_code( project, folder, model, input_x, json_path, rand_flag=0, seed_no=-1, chk_shape=0, code="all" ):

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
    code1 = generate_input_and_param_c_code( obj, model, input_x )
    if len(code1) > 0:
        code1 = """
    //
    //  {}_param
    //""".format(project)+code1
        print( "[PARAM]", param_path )
        ofparam = open( param_path, "w" )
        ofparam.write( code1 )
    else:
        param_fname=""

    # save cpp file
    code2 = generate_graph_c_code(obj, model, chk_shape, rand_flag )
    code2="""
    //
    //  {title}
    //""".format(title=project)+code2
    code3 = generate_main_c_code(obj, seed_no, chk_shape)
    code2+=code3

    print("[CPP] ", cpp_path )
    ofp = open( cpp_path, "w" )
    ofp.write( code2 )

    # save make file
    code3 = makefile_generator( project, code )

    print( "[MAKE]", make_path )
    ofpmake = open( make_path, "w" )
    ofpmake.write( code3 )


def convert_data_file( project, folder, **datas ):

    data_fname = project + "_data.cpp"
    data_path  = folder + "/" + data_fname

    # save data file
    code = generate_data_c_code( **datas )
    if len( code ) > 0:
       print( "[DATA]", data_path )
       ofparam = open( data_path, "w" )
       ofparam.write( code )


def convert_train_code( project, folder, json_path, **kwargs ):

    train_fname = project + "_train.cpp"
    train_path  = folder + "/" + train_fname

    # open json file
    print( "[JSON]", json_path )
    fp = open( json_path )
    obj = json.load(fp)

    # save train_cpp file
    train_code = c_train_code_generator( project, folder, obj, **kwargs )

    print("[TRAIN] ", train_path )
    ofp_train = open( train_path, "w" )
    ofp_train.write( train_code )


def convert_all( project, folder, model, json_path, input_x, data_dict={}, **kwargs ):
    
    os.makedirs(folder,exist_ok=True)
    rand_flag=0
    if "rand_flag" in kwargs:
        rand_flag=kwargs["rand_flag"]
    code="all"
    if "code" in kwargs:
        code=kwargs["code"]
    seed_no = -1
    if "seed" in kwargs:
        seed_no = kwargs["seed"]
    chk_shape = 0
    if "shape" in kwargs:
        chk_shape = kwargs["shape"]
        
    kwargs2 = kwargs.copy()
    kwargs2.update( data_dict )
        
    convert_cpp_code( project, folder, model, input_x, json_path, rand_flag=rand_flag, seed_no=seed_no, chk_shape=chk_shape, code=code )
    if code == "all":
        convert_data_file( project, folder, **data_dict )
        convert_train_code( project, folder, json_path, **kwargs2 )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=str, help="computational graph json")
    parser.add_argument(
        "--project","-p", type=str, default="example", help="project name [example]"
    )
    parser.add_argument(
        "--output_path","-o", type=str, default="output", nargs="?", help="output path [output]"
    )
    parser.add_argument(
        "--model","-m", type=str, default="Net", nargs="?", help="model name [Net]"
    )

    args = parser.parse_args()

    filename = args.graph
    convert_cpp_code( args.project, args.output_path, args.model, input_x, json_path, rand_flag=0)

if __name__ == "__main__":
    main()
