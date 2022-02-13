import json
import argparse
import numpy as np
import os
import torch

def makefile_generator( project, code="all", xtensor_include_base="../", minictorch_include="../src" ):
    make_text="""
CXX = g++
CXXFLAGS += -g -Wall  -std=c++14 -I./ -I{minictorch_inc} -I{xtensor_base}xtensor-blas/include -I{xtensor_base}xtensor/include -I{xtensor_base}xtl/include
LDFLAGS = -lcblas
TARGET  = {proj}
SRCS    = {proj}.cpp {proj}_param.cpp
OBJS    = $(SRCS:.cpp=.o)

""".format(proj=project,xtensor_base=xtensor_include_base,minictorch_inc=minictorch_include)

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
#     VAE/Net[net]/Linear[fc1]/weight/158 -> fc1_weight
def get_param_name( s1 ):
    s2 = s1.split('/')
    n = 1
    for i in range(len(s2)):
      k = s2[i].find("[")
      if( k >= 0 ):  n = i
    s3 = re.findall("(?<=\[).+?(?=\])", s2[n])
    s4 = s3[0] + '_' + s2[n+1]
    return s4
    
#
def string_tensor( key, out ):
    
    if torch.is_tensor(out):
        tmp = out.to('cpu').detach().numpy().copy()
    else:
        tmp = out
    p1 = np.reshape( tmp,(-1,) )
    n1 = len(p1)
    key2 = key
    if len(key) < 12:
        key2 = key + ' '*(12-len(key))
    s1 = 'Tensor ' + key + ' ={ '
    
    num = 8
    nw1 = n1//num
    nw2 = n1% num
    if nw2 == 0:
        nw1 = nw1 - 1
        nw2 = num
        
    l = 0
    for k in range(nw1):
        if ( (nw1>1000) & (k>0) & (k % 10000 == 0)):
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
        print("param:",key," - str loop ",nw1," / ", nw1)  # 220203 add
    s1 = s1 + ' }'
    #print("tensor :",s1)
    
    n2 = len( tmp.shape )
    if n2 == 0:
        s2 = ""
    else:
        s2 = key + '.reshape({'
        for i in range(n2-1):
            s2 = s2 + str( tmp.shape[i] ) + ','
        s2 = s2 + str( tmp.shape[n2-1] ) + '})'
    #print("shape :",s2)
    
    return s1, s2
    

# export all input data
def c_data_generator( **datas ):
    
    key_list = list( datas.keys() )
    if len(key_list) < 1:  return ""
    
    # type declaration
    all_text="""
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    """
    
    # Data section
    val_list = list( datas.values() )
    
    for i in range(len(key_list)):
        print("datafile key : ", key_list[i])
        s1,s2 = string_tensor( key_list[i], val_list[i])
        text="""
        // data
        
        {ivar1};
    
        """.format(ivar1=s1)
        all_text += text
    
    return all_text
    

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
    
    key_list = [];
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
            if( key not in key_list ):
                key_list.append( key )
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
                print(el)
                v = np.array( val )
                s1, s2 = string_tensor( key, v )
                text+="""
    {ivar1};
    """.format(i=i,key=key,shape=",".join(map(str,shape)), ivar1=s1)
                all_text+=text

    return all_text

    
def c_code_generator( project, obj, model, seed_no=-1, chk_shape=0, rand_flag=0 ):
    
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
    #include "minictorch.hpp"
    
    using namespace std;
    
    extern Tensor  xin;"""
    
    key_list = []
    cno = 0  # Constant no.
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            text=""
            name = el["name"]
            key = get_param_name( name )
            if( key not in key_list ):
                key_list.append( key )
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
                
                
    all_text +="""
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {"""
    
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
            
            elif len(el["shape"]) == 0:
                val = el["constant_value"]
                text+="""
            Tensor c = (fprec){val};
            forward_result[{i}] = new VariableTensor( c, 1 );""".format(i=i,val=str(val))
            
            else:
                if len(el["shape"]) > 0: # Constant no.
                    cno += 1
                    key = "Constant" + str(cno)
                    text+="""
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key}, 1 );""".format(i=i,key=key)
                else:
                    val=el["constant_value"]
                    text+="""
            Tensor t= {{{val}}};
            t = t.reshape(shape);
            forward_result[{i}] = new VariableTensor( t, 1 );""".format(i=i,shape=",".join(map(str,shape)), val=",".join(map(str,val)))
        
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
            forward_result[{i}] = new VariableTensor( {key}, 2 );""".format(i=i,key=key,shape=shape)
            
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
            forward_result[{i}] = new VariableTensor( t, 2 );""".format(i=i,shape=shape,shy=shy)
                
        else:
            ###
            ### standard operators
            ###
            out_id = el["output_id"]
            
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
            elif el["op"]=="aten::log1p":
                text+="""
            Log1pOp* op = new Log1pOp();"""
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
            SumOp* op = new SumOp();"""
            elif el["op"]=="aten::mean":
                text+="""
            MeanOp* op = new MeanOp();"""  
            elif el["op"]=="aten::stack":
                text+="""
            StackOp* op = new StackOp();"""  
            elif el["op"]=="aten::select":
                text+="""
            SelectOp* op = new SelectOp();"""
            elif el["op"]=="aten::copy_":
                text+="""
            Copy_Op* op = new Copy_Op();"""
            elif el["op"]=="aten::t":
                text+="""
            TransposeOp* op = new TransposeOp();"""
            elif el["op"]=="aten::max":
                text+="""
            MaxOp* op = new MaxOp();"""
            elif el["op"]=="aten::min":
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
            elif el["op"]=="aten::softplus":
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
            elif el["op"]=="aten::batch_norm":
                text+="""
            BatchNormOp* op = new BatchNormOp();"""
            elif el["op"]=="aten::dropout":
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
            elif el["op"]=="aten::nll_loss_nd":
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
            NumToTensorOp* op = new NumToTensorOp();"""
            elif el["op"]=="aten::Int":
                text+="""
            IntOp* op = new IntOp();"""
            elif el["op"]=="aten::view":
                text+="""
            ViewOp* op = new ViewOp();"""
            elif el["op"]=="aten::broadcast_tensors":
                text+="""
            BroadcastTensorsOp* op = new BroadcastTensorsOp();"""
            elif el["op"]=="aten::to":
                text+="""
            ToOp* op = new ToOp();"""
            elif el["op"]=="aten::detach":
                text+="""
            DetachOp* op = new DetachOp( {k} );""".format(k=out_id)
            elif el["op"]=="prim::ListConstruct":
                text+="""
            ListConstructOp* op = new ListConstructOp();"""
            elif el["op"]=="prim::ListUnpack":
                text+="""
            ListUnpackOp* op = new ListUnpackOp( {k} );""".format(k=out_id)
            elif el["op"]=="prim::TupleConstruct":
                text+="""
            TupleConstructOp* op = new TupleConstructOp();"""
            elif el["op"]=="prim::TupleUnpack":
                text+="""
            TupleUnpackOp* op = new TupleUnpackOp( {k} );""".format(k=out_id)
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

    # 1 forward / backward
    text ="""
    void do_train1( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N )
    {
        cout<<"### forward computation ..."<<endl;
        for(int k=0;k<=N;k++) {
            if( forward_result[k] )  
            {"""
    if chk_shape > 0:
        text+="""
                forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->check_shape();
                forward_result[k]->zerograd();"""
    else:
        text+="""
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();"""
    text +="""
            }
        }
        auto o = forward_result[N]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[N]->grad = xt::ones_like( forward_result[N]->output );
        for(int k=N;k>=0;k--) {"""
    if chk_shape > 0:
        text +="""
            if( forward_result[k] )  
            {
               forward_result[k]->backward();
               forward_result[k]->check_grad_shape();
            }"""
    else:
        text +="""
            if( forward_result[k] )  forward_result[k]->backward();"""
    text +="""
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    }
    
    """
    all_text += text
    
    # main program
    all_text +="""
    #ifdef _TRAIN
    extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N );
    #endif
    
    int main()
    {{
        vector<MCTNode*> forward_result({graph_size});
    """.format(graph_size=len(obj))
        
    for i,el in enumerate(obj):
        if el["op"]=="IO Node":
            if "input" in el["name"]:
                shape=el["shape"]
                text="""
        // input data
        Tensor::shape_type shape = {{{shape}}};
        xin.reshape( shape );
        VariableTensor input_var( xin, 3 );
    """.format(i=i,shape=",".join(map(str,shape)))
    all_text += text
    
    if seed_no >= 0:
        text ="""
        xt::random::seed( {ns} );
    """.format(ns=seed_no)
        all_text += text
    
    all_text +="""
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, {output_id} );
    #else
        do_train1( forward_result, input_var, {output_id} );
    #endif
        
        return 0;
    }}
    """.format(output_id=output_id)
    
    return all_text
    
def unpack_origin_no( obj, no1 ):
    no  = no1
    el1 = obj[no]
    no2 = el1["in"][0]
    out_id = el1['output_id']
    el2 = obj[no2]
    #print( "unpack",el2["op"], out_id)
    if el2["op"] == 'prim::ListConstruct':
        no = el2['in'][out_id]
    elif el2["op"] == 'prim::TupleConstruct':
        no = el2['in'][out_id]
    elif el2["op"] == 'aten::broadcast_tensors':
        el3 = obj[el2['in'][0]]
        if el3["op"] == 'prim::ListConstruct':
            no = el3['in'][out_id]
    #print("unpack original no: ",no,no1)
    return no
    
def get_unpack_origin( obj, no1, inout ):
    no = no1
    while True:
        el = obj[no]
        if el["op"] == 'prim::ListUnpack' or el["op"] == 'prim::TupleUnpack':
            no = unpack_origin_no( obj, no )
        elif el["op"] == 'prim::ListConstruct':  # 220203 add
            #no = unpack_origin_no( obj, no )
            for j in range(len(el['in'])):
                k = get_unpack_origin( obj, el['in'][j], inout )
                if inout[k] == 1: return k
            return no1
        else:
            break
    return no

def get_tensor_shape( x ):
    num = len( x.shape )
    text = ""
    if num > 1:
        text = "{"
        for i in range(num-1):
            text += str(x.shape[i])
            text += ','
        text += str(x.shape[num-1]) + "}"
        return 1,text
    elif num == 1:
        text = "{" + str(x.shape[0]) + "}"
        return 1,text
    return 0,text
    
def c_train_code_generator( project, folder, obj, **kwargs ):
    
    # arguments
    nwargs = len(kwargs)
    if nwargs < 1: return ""
    
    epochs = 200
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
    print("epoch_num : ", epochs )
        
    batchs = 32
    if 'batch' in kwargs:
        batchs = kwargs['batch']
    print("batch_size : ", batchs )
        
    lr = 0.01;
    if 'lr' in kwargs:
        lr = kwargs['lr']
    print("lr : ", lr )
        
    net_key = "Net"
    if 'net_key' in kwargs:
        net_key = kwargs['net_key']
        print("net_key :", net_key )
    
    loss_key = "Loss"
    if 'loss_key' in kwargs:
        loss_key = kwargs['loss_key']
        print("loss_key :", loss_key )
        
    pred_key = ""
    if 'pred_key' in kwargs:
        pred_key = kwargs['pred_key']
        print("pred_key :", pred_key )
        
    pred_index = -1
    if 'pred_index' in kwargs:
        pred_index_no = kwargs['pred_index']
        print("pred_index :", pred_index )
        
    input_opt = 0;
    input_s = ""
    if 'input_data' in kwargs:
        input_data = kwargs[ 'input_data']
        input_opt, input_s = get_tensor_shape( input_data )
        
    target_opt = 0
    target_s = ""
    if 'target_data' in kwargs:
        target_data = kwargs[ 'target_data']
        target_opt, target_s = get_tensor_shape( target_data )
        
    print("input  shape : ", input_opt, input_s)
    print("target shape : ", target_opt, target_s)
    
    shuffle = 0
    if "shuffle" in kwargs:
        shuffle = kwargs["shuffle"]
    print("shuffle : ", shuffle )
    
    pred_output = 0;
    if input_opt > 0:
        pred_output = input_data.shape[0]
        
    if 'pred_output' in kwargs:
        pred_output = kwargs['pred_output']
    print("pred_output : ", pred_output)
    
    div_flag = False;
    if 'div' in kwargs:
        div_flag = kwargs['div']
    #print("div : ", div_flag)
        
    # ---------- 
    
    # output id
    output_id = -1
    for i,el in enumerate(obj):
        if "output" in el['name']:
            assert len(el["in"])>0, "output error"
            output_id = el["in"][0]
            
    nadd1 = 0;
    nadd2 = 0;
    if output_id >=0:
        el = obj[output_id];
        if el["op"] == "aten::add":  # for vae
            nadd1 = el['in'][0]
            nadd2 = el['in'][1]
            print("nadd : ", nadd1,nadd2)
    
    # division number (X)           
    ndiv = []
    if div_flag:
        k = 0
        for i,el in enumerate(obj):
            if el["op"]=="aten::div":
                el2 = obj[ el['in'][1] ]
                if el2["op"] == "prim::Constant":
                    div = el2["constant_value"]
                    print("div value:",div)
                    if abs(div-batchs) < 0.01:
                        ndiv.append( el['in'][1] )
                        print("div_el",el2)
                        k = k+1
        if len(nd) != 2:  div_flag = False
        print("div no :",ndiv)
    
    # evaluated no
    pred_no   = -1
    target_no = -1
    pred_pos  = -1
    class_no  =  0
    
    pos1 = 0
    pos2 = 0
    for i,el in enumerate(obj):
        if net_key in el["name"]:
            pos1 = i+1
        if loss_key in el["name"]:
            pos2 = i
            break
    if pos1 > output_id:  pos1 = 0
    if ( pos1 > 0 ) and ( pos1 < pos2 ):
        pred_pos = pos1
    elif pos2 > 0:
        pred_pos = pos2
    if pos1 > pos2:  pred_pos = -1  # 220127 add
    print("pred_pos : ",pred_pos,pos1,pos2,output_id)
    
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
                if inout[k] == 1: nin=nin+1
            if nin > 0: inout[i] = 1
            
    #set_no = set()
    last = -1
    pred_max = -1
    pred_id  = -1
    for i in range(output_id+1):
        el = obj[i]
        #print("el ",i,inout[i], " : ", el['name'],el['op'],el['in'],el['output_id'])
        if inout[i] > 0:
            print("el ",i,inout[i], " : ", el['name'],el['op'],el['in'],el['output_id'],pred_no)
            if last < i: last = i
            if pred_pos < 0:  
                pass
                #for j in range(len(el['in'])):
                #    k = get_unpack_origin( obj, el['in'][j], inout )
                #    if inout[k] > 0:
                #        print(" --- pred el (",j,") :",el['in'][j]," -> ",k)
                #        set_no.add( k )
            elif i >= pred_pos:
                for j in range(len(el['in'])):
                    k = get_unpack_origin( obj, el['in'][j], inout )
                    if ( k < pred_pos ) and ( inout[k] > 0 ):
                        print(" --- pred el (",j,") :",el['in'][j]," -> ",k)
                        #set_no.add( k )
                        if pred_max < k: 
                            pred_max = k
                            pred_id  = i
    
    print("pred_id",pred_id,pred_key)
    if pred_id > 0:
        el = obj[pred_id]
        type = 0
        if   el['op'] == 'aten::mse_loss':  type = 1
        elif el['op'] == 'aten::cross_entropy_loss':    type = 2
        elif el['op'] == 'aten::binary_cross_entropy':  type = 2
        elif el['op'] == 'aten::nll_loss_nd':  type = 3  # 220203 add
        else:  type = 4
        print("loss pred_id : ",pred_id,pred_max)
        if type > 0:
            pred_no = get_unpack_origin( obj, el['in'][0], inout )
            if len(el['in']) > 1: 
                no2 = get_unpack_origin( obj, el['in'][1], inout )
                if target_opt > 0: target_no = no2
                if type == 2:      class_no  = no2
                if type == 3:      class_no  = no2  # 220203 add
        if pred_no > 0:  
            print("eval1 no :",i," (type=",type,") : ", pred_no,target_no)
            
    if pred_pos < 0:
        if len(pred_key) > 0:
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
                        if target_opt > 0: target_no = no2
                        if type == 2:      class_no  = no2
                        if type == 3:      class_no  = no2
                if pred_no > 0:  
                    if type == 3: pred_no = -1  #220127 add
                    print("eval2 no :",i,"-> type=",type," : ",pred_no,target_no)
                    break
            
        
    print("last cmd:", last)
    #if len(set_no) > 0:
    #    print("set_no :", sorted(set_no,reverse=True))
    print("------")
        
    ###
    sol_kind = "";
    if "sol" in kwargs:
        sol_kind = kwargs["sol"]
    print("solution :", sol_kind)
    if not "clas" in sol_kind:  class_no = 0
    
    if ( pred_index >= 0 ) and ( pred_index < output_id ):
        if inout[pred_index] > 0:  pred_no = pred_index
   
    print("pred_no   :", pred_no)
    print("target_no :", target_no)
    print("class_no  :", class_no)
    
    
    # header section
    all_text="""
    //
    //  {proj}_train
    //
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    extern bool train_mode;
    """.format(proj=project)
    
    text = ""
    if input_opt > 0:
        text +="""
    extern Tensor input_data;"""
    if target_opt > 0:
        text +="""
    extern Tensor target_data;"""
    all_text += text
    
    # train loop program
    all_text +="""
    
    void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int NL )
    {
        auto do_forward=[]( vector<MCTNode*> &op, int n ) 
        {
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->forward();
            }
        };
        auto do_backward=[]( vector<MCTNode*> &op, int n ) 
        {
            op[n]->grad = xt::ones_like( op[n]->output );
            for(int k=n;k>=0;k--) {
              if( op[k] )  op[k]->backward();
            }
        };
        auto do_zerograd=[]( vector<MCTNode*> &op, int n ) 
        {
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->zerograd();
            }
        };
        auto update_params=[]( vector<MCTNode*> &op, int n, fprec lr=0.01 ) 
        {
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->update( lr );
            }
        };"""
    
    if class_no > 0:
        all_text +="""
        auto eval_labels=[]( Tensor& y, Tensor &t )
        {
            auto lb = xt::argmax( y, 1 );
            auto eq = xt::equal( t, lb );
            auto sm = xt::sum( eq );
            return (int)sm[0];
        };"""
    
    # common parameter
    all_text +="""
    
        //xt::random::seed(1);  
        
        fprec lr = {lr};
        int epoch_num = {ne};
        cout<<"epoch_num : "<<epoch_num<<endl;
    """.format(ne=epochs,lr=lr)
    
    text = ""
    if batchs > 0:
        
        if input_opt > 0:
            text +="""
        input_data.reshape( {shape} );
        auto input_shape = input_data.shape();
        """.format(shape=input_s)
        
        if target_opt > 0:
            text +="""
        target_data.reshape( {shape} );
        auto target_shape = target_data.shape();
        """.format(shape=target_s)
    
        text +="""
        int batch_size = {bz};
        int n_batch = (int)input_shape[0] / batch_size;
        cout<<"batch  number  : "<<n_batch<<","<<batch_size<<endl;
        cout<<"learning ratio : "<<lr<<endl;
    
        """.format(bz=batchs)
        
    else:
        
        batchs = 0
        if input_opt > 0:
            text +="""
        input_data.reshape( {shape} );
        auto input_shape = input_data.shape();
        input_var.output = input_data;
        
        """.format(shape=input_s)
        
    all_text += text
    
    text = ""
    if batchs > 0:
        if input_opt > 0:
            ds = input_data.shape
            sz = input_data.ndim
            stri = ""
            for k in range(1,sz):  stri += ", " + str(ds[k])
            text +="""
        Tensor x_tmp = xt::zeros<fprec>( {{ batch_size{ss} }} );""".format(ss=stri)
        
        if target_opt > 0:
            ds = target_data.shape
            sz = target_data.ndim
            if sz == 1:
                text +="""
        Tensor y_tmp = xt::zeros<fprec>( { batch_size } );"""
            elif sz > 1:
                stri = ""
                for k in range(1,sz):  stri += ", " + str(ds[k])
                text +="""
        target_data.reshape( {shape} );
        Tensor y_tmp = xt::zeros<fprec>( {{ batch_size{ss} }} );""".format(shape=target_s,ss=stri)
    
    else:
        if class_no > 0:
            text +="""
        auto labels = forward_result[{nt}]->output;
        auto labels_shape = labels.shape();""".format(nt=class_no)
    
    all_text += text
    
        
    # learning section
    fpath = folder + '/' + project + ".out"
        
    all_text += """
    
        ofstream outputfile("{fn}");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {{
            train_mode = true;""".format(fn=fpath)
            
    text = ""
    if batchs > 0:
        if shuffle > 0:
            text += """
            
            xt::xarray<int> index = xt::arange( (int)input_shape[0] );
            xt::random::shuffle( index );"""
            
    if batchs < 1:  # batchsize == 0
        
        if class_no > 0:  # classification only
        
            text +="""
            do_forward( forward_result, NL );
            
            fprec o = forward_result[NL]->output[0];
            int corrects = eval_labels( forward_result[{ns}]->output, y );
            fprec acc = (fprec)corrects / (fprec)labels_shape[0];
            cout<<"epoch "<<epoch<<" - loss "<<o<<" - accuracy "<<acc<<endl;
            outputfile<<to_string(o)<<","<<to_string(acc)<<endl;
        """.format(ns=pred_no)
        
        else:  # others
            
            text +="""
            do_forward( forward_result, NL );
            
            auto o = forward_result[NL]->output;
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<endl;
            outputfile<<to_string(o[0])<<endl;
        """
        
        text += """
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        """
        
    else:  # minibatch ( batchsize > 0 )
            
        if div_flag:
            for i in range(len(nd)):
                text +="""
            forward_result[{nd1}]->set_output1( (fprec)batch_size );  // div size""".format(nd1=nd[i])
            
        if class_no > 0:  # classification only
            text +="""
            
            fprec total_loss = 0.0;
            int   total_corrects = 0;"""
        else:
            text +="""
            
            fprec total_loss = 0.0;"""
        text +="""
            for(int j=0;j<n_batch;j++)
            {
                int jb = j * batch_size;
                for(int k=0;k<batch_size;k++)
                {"""
            
        if input_opt > 0:
            ds = input_data.shape
            sz = input_data.ndim
            strx = ""
            for k in range(2,sz): strx += ", xt::all()"
            
            if shuffle >0:
                #   xt::row( x_tmp, k ) = xt::row( input_data, index(jb+k) );"""
                if sz == 1:
                    text +="""
                    x_tmp( k ) = input_data( index(jb+k) );"""
                else:
                    text +="""
                    auto xw = xt::view( input_data, index(jb+k){ss} );
                    xt::view( x_tmp, k{ss} ) = xw;""".format(ss=strx)
            else:
                #    xt::row( x_tmp, k ) = xt::row( input_data, jb+k );"""
                if sz == 1:
                    text +="""
                    x_tmp( k ) = input_data( jb+k );"""
                else:
                    text +="""
                    auto xw = xt::view( input_data, jb+k{ss} );
                    xt::view( x_tmp, k{ss} ) = xw;""".format(ss=strx)
                    
        if target_opt > 0:
            ds = target_data.shape
            sz = target_data.ndim
            strx = ""
            for k in range(2,sz): strx += ", xt::all()"
            
            if shuffle > 0:
                    #xt::row( y_tmp, k ) = xt::row( target_data, index(jb+k) );"""
                if sz == 1:
                    text +="""
                    y_tmp( k ) = target_data( index(jb+k) );"""
                else:
                    text +="""
                    auto yw = xt::view( target_data, index(jb+k){ss} );
                    xt::view( y_tmp, k{ss} ) = yw;""".format(ss=strx)
            else:
                    #xt::row( y_tmp, k ) = xt::row( target_data, jb+k );"""
                if sz == 1:
                    text +="""
                    y_tmp( k ) = target_data( jb+k );"""
                else:
                    text +="""
                    auto yw = xt::view( target_data, jb+k{ss} );
                    xt::view( y_tmp, k{ss} ) = yw;""".format(ss=strx)
                    
        text +="""
                }
                
                input_var.output = x_tmp;"""
        if target_no > 0:
            text +="""
                forward_result[{nt}]->output = y_tmp;""".format(nt=target_no)
        
        if class_no > 0:  # classification only
            text +="""
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
                
                int corrects = eval_labels( forward_result[{ns}]->output, y_tmp );
                total_corrects += corrects;
            
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }}
            fprec total_acc = (fprec)total_corrects / (fprec)input_shape[0];
            cout<<"total_loss (batch): epoch "<<epoch<<" : loss "<<total_loss<<" : Acc "<<total_acc<<" "<<total_corrects<<endl;
            """.format(ns=pred_no)
        
        else: # others
        
            text +="""
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
                
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss : epoch "<<epoch<<" - loss "<<total_loss<<endl;
            """
            
        # print loss value per epoch
        text +="""
            train_mode = false;
            
            input_var.output = input_data;"""
            
        if target_no > 0:
            text +="""
            forward_result[{nt}]->output = target_data;""".format(nt=target_no)
            
        if div_flag:
            for i in range(len(nd)):
                text +="""
            forward_result[{nd1}]->set_output1( (fprec)input_shape[0] );  // div size""".format(nd1=nd[i])
            text +="""
            """
            
        if( nadd1 > 0 and nadd2 > 0 and class_no < 1 ):  # 220203 mod
            text +="""
            do_forward( forward_result, NL );
            
            auto o  = forward_result[NL]->output;
            auto o1 = forward_result[{na1}]->output;
            auto o2 = forward_result[{na2}]->output; 
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
            outputfile<<to_string(o[0])<<endl;
            """.format(na1=nadd1,na2=nadd2)
        else:
            if class_no > 0:  # classification only
                text +="""
            do_forward( forward_result, NL );
            
            auto o = forward_result[NL]->output;
            int corrects = eval_labels( forward_result[{ns}]->output, target_data );
            fprec acc = (fprec)corrects / (fprec)input_shape[0];
            cout<<"total_loss (all)  : epoch "<<epoch<<" : loss "<<o[0]<<" : Acc "<<acc<<" "<<corrects<<endl;
            outputfile<<to_string(o[0])<<","<<to_string(acc)<<","<<total_loss<<endl;""".format(ns=pred_no)
            
            else: # others
                text+="""
            do_forward( forward_result, NL );
            
            auto o  = forward_result[NL]->output;
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<endl;
            outputfile<<to_string(o[0])<<endl;
            """
            
    all_text += text
        
    all_text +="""
        }
        outputfile.close();
        
        train_mode = false;"""
    
    # prediction section
    path = folder + '/' + project + ".pred"
    
    text=""
    if pred_no > 0:
        print("pred_no : ",pred_no)
        if batchs < 1:  # batchsize == 0
        
            if input_opt > 0:
                
                el = obj[pred_no]
                text +="""
        input_data.reshape( {shape} );
        {{
            // {ns} : {el1}
            input_var.output = input_data;
            auto inp_shape = input_data.shape();
            int  nx = inp_shape[0];
            """.format(shape=inp_s,ns=pred_no,el1=el['op'])
            
                text += """
            do_forward( forward_result, {ns} );
            auto y_pred = forward_result[{ns}]->get_output();
            
            ofstream outputfile( "{fn}" );
            outputfile<<to_string(nx)<<",1"<<endl;
            for(int i=0;i<nx;i++)
            {{
                outputfile<<to_string(y_pred(i,0))<<endl;
            }}
            outputfile.close();
        }}""".format(fn=path,ns=pred_no)
            
        else:  # minibatch ( batchsize > 0)
        
            if class_no < 1: # not classification only
                el = obj[pred_no]
            
                pred_type = 1  ###
                if input_opt > 0:
                    pred_type = len(input_data.shape)
                print("pred_type : ",pred_type)
                
                if pred_type == 1: ###
                    text +="""
        {{
            // {ns} : {el1}""".format(ns=pred_no,el1=el['op'])
            
                    if input_opt > 0:
                        text +="""
            input_var.output = input_data;"""
                 
                    text +="""
            do_forward( forward_result, {ns} );
            auto y_pred = forward_result[{ns}]->output;
            
            ofstream outputfile( "{fn}" );
            outputfile<<to_string(input_shape[0])<<",1"<<endl;
            
            for(int i=0;i<input_shape[0];i++)
            {{
                outputfile<<to_string(y_pred(i,0))<<endl;
            }}
            outputfile.close();
        }}""".format(fn=path,ns=pred_no)
            
                elif pred_type > 1:
                    el = obj[pred_no]
                    text +="""
            
        {{
            // {ns} : {el1}""".format(ns=pred_no,el1=el['op'])
            
                    if input_opt > 0:
                        text +="""
            input_var.output = input_data;"""
                
                    text +="""
            do_forward( forward_result, {ns} );
            auto y_pred = forward_result[{ns}]->output;
            """.format(ns=pred_no)
            
                    nx_set = 0;
                    if input_opt > 0:
                        if pred_output == input_data.shape[0]:
                            nx_set = 1
                            text +="""
            int nx = input_shape[0];
            """
                    if nx_set == 0:
                        text +="""
            int nx = {nx};
            """.format(nx=pred_output)
            
                    text +="""
            ofstream outputfile( "{fn}" );
            outputfile<<to_string(nx)<<","<<to_string(input_shape[1])<<endl;
            
            for(int i=0;i<nx;i++)
            {{
                for(int j=0;j<input_shape[1]-1;j++)
                {{
                    outputfile<<to_string(y_pred(i,j))<<",";
                }}
                outputfile<<to_string(y_pred(i,input_shape[1]-1))<<endl;
            }}
            outputfile.close();
        }}""".format(fn=path,nx=pred_output)
        
    
    # latent variable
    z_no  = -1
    if "z" in kwargs:   
        keyw = kwargs["z"]
        for i,el in enumerate(obj):
            if el["op"] == "aten::linear":
                if keyw in el['name']:
                    z_no = el['in'][0]
                    print("vae z: ", z_no," keyw :", keyw )
                    
    if z_no >= 0:
        pathz = folder + '/' + project + ".z"
            
        text +="""
        {{
            // {nz} : z output
            auto z_pred = forward_result[{nz}]->get_output();
        
            ofstream outputfile( "{fn}" );
            outputfile<<to_string(input_shape[0])<<","<<to_string(2)<<endl;
        
            for(int k=0;k<input_shape[0];k++)
            {{
                outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
            }}
            outputfile.close();
        }}
        """.format(fn=pathz,nz=z_no)
    
    all_text += text
    all_text +="""
    
    }
    """
    return all_text


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
    code1 = c_param_generator( project, obj, model, input_x )
    if len(code1) > 0:
       print( "[PARAM]", param_path )
       ofparam = open( param_path, "w" )
       ofparam.write( code1 )
    else:
       param_fname=""

    # save cpp file
    code2 = c_code_generator( project, obj, model, seed_no, chk_shape, rand_flag )  # 220120 add seed_no

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
    code = c_data_generator( **datas )
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
    chk_shp = 0
    if "shape" in kwargs:
        chk_shp = kwargs["shape"]
        
    kwargs2 = kwargs.copy()
    kwargs2.update( data_dict )
        
    convert_cpp_code( project, folder, model, input_x, json_path, rand_flag=rand_flag, seed_no=seed_no, chk_shape=chk_shp, code=code )
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
    code = c_code_generator(obj)
    convert_cpp_code( args.project, args.output_path, args.model, input_x, json_path, rand_flag=0)

    make_code = makefile_generator(args.output)

    print("[SAVE]",args.path+"/"+args.output)
    ofp = open( args.path+"/"+args.output, "w" )
    ofp.write( code )
    makefp=open( args.path+"/"+"Makefile", "w" )
    makefp.write( make_code )

if __name__ == "__main__":
    main()
