import json
import argparse
import numpy as np
import os
import torch
import re
from jinja2 import Template, Environment, FileSystemLoader

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
    "aten::sigmoid_":"SigmoidOp",
    "aten::relu":"ReluOp",
    "aten::relu_":"ReluOp",
    "aten::elu":"EluOp",
    "aten::elu_":"EluOp",
    "aten::leaky_relu":"LeakyReluOp",
    "aten::leaky_relu_":"LeakyReluOp",
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
 
def generate_makefile( project, make_target="all", xtensor_include_base="../", minictorch_include="../src",optimize="-O3",libs="-lmkl_rt"): #libs="-lcblas"):
    make_tmpl="make.tmpl.cpp"
    params={
        "proj":project,
        "make_target":make_target,
        "xtensor_base":xtensor_include_base,
        "minictorch_inc":minictorch_include,
        "optimize":optimize,
        "libs":libs,
        }
    print(params)
    path=os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
    make_tmpl = env.get_template(make_tmpl)
    rendered_s = make_tmpl.render(params)
    return rendered_s, params

# attr_name: attribute name in computational graph e.g. Net/Linear[fc3]/bias/bias 
# model: pytorch model
# Return:  pytorch parameters  <class 'torch.nn.parameter.Parameter'>
#
def get_attr_from_model( attr_name, model ):
    pat = re.compile(r'([^\[\]]*)\[(.*)\]')
    arr = attr_name.split("/")
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

"""
### v1
def get_param_name( attr_name ):
    sep_list = attr_name.split('/')
    n = 1
    for i in range(len(sep_list)):
        if sep_list[i].find("[")>=0:
            n = i
    keys = re.findall("(?<=\[).+?(?=\])", sep_list[n])
    out_name = keys[0] + '_' + sep_list[n+1]
    return "param_"+out_name
"""
### v2
def get_param_name( attr_name ):
    sep_list = attr_name.split('/')
    out=[]
    for e in sep_list:
        keys = re.findall("(?<=\[).+?(?=\])", e)
        if len(keys)>0:
            out.append(keys[0])
    last=sep_list[-1]
    name=last.split(".")[0]
    attr_name="_".join(out)+"_"+name
    return "param_"+attr_name
# 
# 
# fc3_bias
# pytorch tensor: tensor([0.0402, 0.0524, 0.0345], requires_grad=True)
#
#=> result: Tensor fc3_bias ={ 0.040239427,0.052421827,0.034500692, }
#   result_reshape: fc3_bias.reshape({3})
# 
def build_inline_tensor_c_code( key, out ):
    num_per_line = 8
    n_indent = 24
    
    if torch.is_tensor(out):
        tmp = out.to('cpu').detach().numpy().copy()
    else:
        tmp = out
    flatten_tmp = np.reshape( tmp,(-1,) ) # flatten
    
    ## generating result line
    n1 = len(flatten_tmp)
    #result = '// size:{} ({})\n'.format(n1, tmp.shape)
    #result += 'Tensor ' + key + ' ={ '
    result = '{'
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
    return n1, result, result_reshape

# export all input data
def generate_data_c_code( pair_data, input_from_file, folder):
    tensor_list=[]
    for key,val in pair_data.items():
        print("generating data: ", key)
        tensor_info={}
        if not input_from_file:
            length, tensor_code, reshape_code = build_inline_tensor_c_code( key, val)
            tensor_info["tensor_code"]=tensor_code
            tensor_info["reshape_code"]=reshape_code
            tensor_info["length"]=length
        tensor_info["shape"]=",".join(map(str,val.shape))
        tensor_info["name"]=key
        if input_from_file:
            filename = "data_"+key+".npy"
            tensor_info["filename"]=filename
            print("[SAVE]", folder+"/"+filename)
            np.save(folder+"/"+filename,val)
        tensor_list.append(tensor_info)
    params={"tensor_list":tensor_list,"input_from_file":input_from_file}
    # 1 forward / backward
    path=os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
    #train_tmpl = env.get_template('train.tmpl.cpp')
    data_tmpl="data.tmpl.cpp"
    data_tmpl = env.get_template(data_tmpl)
    rendered_s = data_tmpl.render(params)
    return rendered_s, params


def get_one_line(indent,s):
    return "\n"+"    "*indent+s+"\n"

def get_input_index(name,name_list):
    bname=name.split("/")[-1]
    if bname in name_list:
        index=name_list.index(bname)
        return index
    else:
        res = re.match(r'.*([0-9]+)', bname)
        if res:
            return int(res[1])-1
        return 0 # single input


def generate_input_and_param_c_code( obj, model, input_list, input_name_list):
    # type declaration
    input_tensor_list=[]
    for input_var, input_name in zip(input_list,input_name_list):
        input_tensor_info={}
        if input_var is not None:
            length, tensor_code, reshape_code = build_inline_tensor_c_code( input_name, input_var)
            input_tensor_info["tensor_code"]=tensor_code
            input_tensor_info["reshape_code"]=reshape_code
            input_tensor_info["length"]=length
            input_tensor_info["shape"]=",".join(map(str,input_var.shape))
            input_tensor_info["name"]=input_name
        input_tensor_list.append(input_tensor_info)
    
    pair_data={}
    comment_data={}
    n_constant = 0
    for i,el in enumerate(obj):
        name = el["name"]
        if el["op"]=="prim::GetAttr":
            key = get_param_name( name )
            if( key not in pair_data ):
                attr = get_attr_from_model( name, model )
                pair_data[key]=attr
                pair_data[key]=(attr,",".join(map(str,attr.shape)))
                comment_data[key]=[(i,str(el))]
            else:
                comment_data[key].append((i,str(el)))
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                n_constant += 1
                key ="Constant" + str(n_constant)
                shape=el["shape"]
                val = np.array(el["constant_value"])
                pair_data[key]=(val,shape)
                comment_data[key]=[(i,str(el))]
    tensor_list=[]
    for key,pair in pair_data.items():
        val,shape=pair
        length, tensor_code, reshape_code = build_inline_tensor_c_code( key, val)
        tensor_info={}
        tensor_info["tensor_code"]=tensor_code
        tensor_info["reshape_code"]=reshape_code
        tensor_info["length"]=length
        tensor_info["shape"]=shape
        tensor_info["name"]=key
        tensor_info["node_list"]=[{"id":i, "text":text} for i, text in comment_data[key]]
        tensor_list.append(tensor_info)
    ######
    make_tmpl="param.tmpl.cpp"
    params={
        "tensor_list":tensor_list,
        "input_tensor_list":input_tensor_list,
        }
    path=os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
    make_tmpl = env.get_template(make_tmpl)
    rendered_s = make_tmpl.render(params)
    return rendered_s, params
    
def build_graph_c_code(obj, model, input_names, chk_shape=0, rand_flag=0 ):
    """
    print("... computational graph")
    for i,el in enumerate(obj):
        print(i,el)
    print("...")
    """
    input_vars={}
    extern_vars=[]
    update_vars={}
    graph_codes=[]
    key_list = []
    n_constant = 0  # Constant no.
    for i,el in enumerate(obj):
        if el["op"]=="prim::GetAttr":
            text=""
            name = el["name"]
            key = get_param_name( name )
            if( key not in key_list ):
                key_list.append( key )
                extern_vars.append(str(key))
        elif el["op"]=="prim::Constant":
            if len(el["shape"])>0:
                n_constant += 1
                key = "Constant" + str(n_constant)
                extern_vars.append(str(key))
    
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
            text+="""
            Tensor::shape_type shape = {{{shape}}};""".format(i=i,shape=",".join(map(str,shape)) )
        ###
        ### operator
        ###
        if el["op"]=="IO Node":
            if "input" in el["name"]:
                index = get_input_index(el["name"],input_names)
                assert index>=0, "wrong input index"
                name = input_names[index]
                print("input: ",el["name"],"->",name)
                if name not in input_vars:
                    input_vars[name]={}
                    input_vars[name]["input_id"]=[]
                    input_vars[name]["shape_str"]=",".join(map(str,el["shape"]))
                    input_vars[name]["shape"]=el["shape"]
                    input_vars[name]["batch_size"]=el["shape"][0]
                    input_vars[name]["shape_rest"]=", xt::all()"*(len(el["shape"])-1)
                    input_vars[name]["index"]=index
                input_vars[name]["input_id"].append(i)
                text+="""
            forward_result[{i}] = input_vars[{index}];""".format(i=i,index=index)
            elif "output" in el["name"]:  # output_id = el["in"][0]
                assert len(el["in"])>0, "output error"
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
            if rand_flag == 0:
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            {key}.reshape( shape );
            forward_result[{i}] = new VariableTensor( {key}, VAR_ATTR );""".format(i=i,key=key,shape=shape)
                    if key not in update_vars:
                        update_vars[key]=[]
                    update_vars[key].append(i)
            
            else: # rand_flag is deprecated
                w_len = len(attr.shape) 
                if w_len > 0:
                    shape = ",".join(map(str,attr.shape))
                    shy = attr.shape[w_len-1]
                    text+="""
            Tensor::shape_type shape = {{{shape}}};
            fprec y = sqrt(1.0/(fprec){shy});
            Tensor t = xt::random::rand(shape,-y,y);
            forward_result[{i}] = new VariableTensor( t, VAR_ATTR );""".format(i=i,shape=shape,shy=shy)
                    if key not in update_vars:
                        update_vars[key]=[]
                    update_vars[key].append(i)
                
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
        graph_codes.append(text)
        
    return extern_vars, update_vars, input_vars, graph_codes

def generate_main_c_code(obj, extern_vars, input_vars, graph_codes, seed_no=-1, chk_shape=0, title=""):
    input_vars_info={}
    for i,el in enumerate(obj):
        if el["op"]=="IO Node":
            #if "input" in el["name"]:
            #    input_id=i
            #   input_shape=el["shape"]
            if "output" in el["name"]: 
                assert len(el["in"])>0, "output error"
                output_id = el["in"][0]

    # 1 forward / backward
    path=os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
    #train_tmpl = env.get_template('train.tmpl.cpp')
    main_tmpl="main.tmpl.cpp"
    train_tmpl = env.get_template(main_tmpl)
    params={
        "title":title,
        "extern_vars":extern_vars,
        "graph_codes":graph_codes,
        "seed_no":seed_no,
        "chk_shape":chk_shape,
        "graph_size":len(obj),
        "input_vars":input_vars,
        "output_id":output_id,
        }
    rendered_s = train_tmpl.render(params)
    return rendered_s, params

def unpack_origin_no( obj, target_index ):
    target_node = obj[target_index]
    in_index = target_node["in"][0]
    in_node = obj[in_index]
    out_id = target_node['output_id'] # output_id は unpack系に付いてListの何番目のものを取り出すか
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
    
# 入力をさかのぼっていって最初に見つかったList/Tuple系以外のノードを見つける
def get_unpack_origin( obj, no1, inout ):
    no = no1
    while True:
        el = obj[no]
        if el["op"] in ['prim::ListUnpack','prim::TupleUnpack']:
            no = unpack_origin_no( obj, no ) ## unpackに対応したConstruct（の入力）を見つけようとする
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
    direct_pos=0
    for i,el in enumerate(obj):
        if net_key in el["name"]: # last Net
            direct_pos=i
            pos1 = i+1   # 一つ先をとっている
        if loss_key in el["name"]: # first Loss
            pos2 = i
            break
    if pos1 > output_id:
        print("[WARN] net_key {} (id={}) is found after output node {}".format(net_key,pos1,output_id))
        pos1 = 0

    if ( 0 < pos1 ) and ( pos1 < pos2 ):
        pred_pos = pos1
    elif pos2 > 0:
        print("[WARN] loss_key {} (id={}) is set as prediction output node".format(loss_key,pos2))
        pred_pos = pos2
    if pos1 > pos2:
        pred_pos = -1

    print("Prediction output node index: ",pred_pos)
    print("Net class output node index: ",pos1)
    print("Loss class input node index: ",pos2)
    print("Output node index: ",output_id)
    
    # inout option ( 0:input only, 1:input and output both )
    # 反転させたときにinputに到達可能なノードを探している？
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
            
    pred_max = -1
    pred_id  = -1
    # pred_pos（net_keyで指定したものの一つ先）より手前で条件を満たす最大IDのノードをとってくる
    for i in range(output_id+1):
        el = obj[i]
        #print("el ",i,"inout=",inout[i], " : ", el['name'],el['op'],"in=",el['in'],"output_id=",el['output_id'])
        if inout[i] > 0:
            #print("el ",i,"inout=",inout[i], " : ", el['name'],el['op'],"in=",el['in'],"output_id=",el['output_id'])
            if pred_pos >= 0 and i >= pred_pos:
                for j in range(len(el['in'])):
                    k = get_unpack_origin( obj, el['in'][j], inout )
                    if ( k < pred_pos ) and ( inout[k] > 0 ):
                        #print(" --- pred el (",j,") :",el['in'][j]," -> ",k)
                        if pred_max < k: 
                            pred_max = k
                            pred_id  = i
    ## pred_idはここまでで決定
    print("direct pred_id:",direct_pos,pred_key)
    print("refine pred_id:",pred_id)

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
            
        
    print("------")
    if not "classification" in task_type:
        class_no = -1

    if ( pred_index >= 0 ) and ( pred_index < output_id ):
        if inout[pred_index] > 0:
            pred_no = pred_index

    return pred_no,target_no,class_no


def generate_train_c_code( project, folder, obj,
        train_tmpl='train.tmpl.cpp',
        epochs = 200,
        batch_size = 0,
        lr = 0.01,
        net_key = "Net",
        loss_key = "Loss",
        pred_key = None,
        pred_index = -1,
        target_data=None,
        shuffle = False,
        pred_output = None,
        task_type = "",
        latent_z=None,
        update_vars=None,
        input_vars=None,
        input_pair_list=None,
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
        
    input_enabled = False
    if input_vars is not None:
        input_enabled = True
        for k,v in input_pair_list:
            input_vars[k]["data_name"]=v
    #input_s = ""
    #print(input_data)
    #if input_data is not None:
    #    input_enabled, input_s = get_tensor_shape( input_data )
    #print("input  shape : ", input_enabled, input_s)
        
    target_enabled = False
    target_s = ""
    if target_data is not None:
        target_enabled, target_s = get_tensor_shape( target_data )
        
    print("target shape : ", target_enabled, target_s)
    
    #pred_output
    if pred_output is None:
        if input_enabled:
            k=next(iter(input_vars))
            pred_output = input_vars[k]["shape"][0]
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
        #if input_enabled:
        #    ds = input_data.shape
        #    x_shape = ",".join([str(batch_size)]+[str(ds[k]) for k in range(1,input_data.ndim)])
        
        if target_enabled:
            ds = target_data.shape
            y_shape = ",".join([str(batch_size)]+[str(ds[k]) for k in range(1,target_data.ndim)])

        #if input_enabled:
        #    ds = input_data.shape
        #    sz = input_data.ndim
        #    if sz>=2:
        #        x_shape_rest_enabled = True
        #        for _ in range(2,sz):
        #            x_shape_rest += ", xt::all()"
        if target_enabled:
            ds = target_data.shape
            sz = target_data.ndim
            if sz>=2:
                y_shape_rest_enabled = True
                for k in range(2,sz):
                    y_shape_rest += ", xt::all()"
         
    double_loss_enabled= len(output_sum_loss_id) > 0 and class_no < 1  # Loss=Loss1+Loss2
 
    pred_type = 1  ###
    if target_enabled:
        pred_type = len(target_data.shape)
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
        "shuffle":shuffle,
        #"input_shape":input_s,
        "input_vars":input_vars,
        "target_shape":target_s,
        "batch_enabled":batch_size>0,
        "batch_size":batch_size,
        #"x_shape":x_shape,
        "y_shape":y_shape,
        #"x_shape_rest":x_shape_rest,
        #"x_shape_rest_enabled": x_shape_rest_enabled,
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
        "update_vars":{k:",".join(map(str,v)) for k,v in update_vars.items()},
        }
    if len(output_sum_loss_id)>=2:
        params["loss1_no"]=output_sum_loss_id[0]
        params["loss2_no"]=output_sum_loss_id[1]
    #print("train code parameters:",params)
    rendered_s = train_tmpl.render(params)
    return rendered_s, params


# convert json file to parameter, cpp, and make file
def convert_cpp_code( project, folder, model, inputs, input_names, json_path, rand_flag=0, seed_no=-1, chk_shape=0, make_target="all", makefile_name="Makefile"):
    cpp_fname        = project + "_main.cpp"
    cpp_json_fname   = project + "_main.json"
    param_fname      = project + "_param.cpp"
    param_json_fname = project + "_param.json"
    cpp_path        = folder + "/" + cpp_fname
    cpp_json_path   = folder + "/" + cpp_json_fname
    param_path      = folder + "/" + param_fname
    param_json_path = folder + "/" + param_json_fname
    make_path       = folder + "/" + makefile_name
    make_json_path  = folder + "/" + makefile_name+".json"
  
    # load json file
    print( "[LOAD JSON]", json_path )
    fp = open( json_path )
    obj = json.load(fp)

    # save parameter file
    if type(inputs) is list:
        input_list=inputs
    else:
        input_list=[inputs]
    if type(input_names) is list:
        temp_input_name_list=input_names
    else:
        temp_input_name_list=[input_names]
    input_name_list=[]
    input_pair_list=[]
    for x in temp_input_name_list:
        if type(x) is str:
            input_name_list.append(x)
        else:
            input_pair_list.append(x)
            input_name_list.append(x[0])
    code, param = generate_input_and_param_c_code( obj, model, input_list, input_name_list)
    print( "[PARAM]", param_path )
    with open( param_path, "w" ) as ofp:
        ofp.write( code )
    print( "[PARAM]", param_json_path )
    with open(param_json_path, 'w') as ofp:
        json.dump(param, ofp)

    # save cpp file
    extern_vars, update_vars, input_vars, graph_codes = build_graph_c_code(obj, model, input_name_list, chk_shape, rand_flag )
    code,param = generate_main_c_code(obj,extern_vars,input_vars, graph_codes, seed_no, chk_shape,title=project)
    print("[CPP] ", cpp_path )
    with open( cpp_path, "w" ) as ofp:
        ofp.write( code )
    print("[CPP] ", cpp_json_path )
    with open(cpp_json_path, 'w') as ofp:
        json.dump(param, ofp)

    # save make file
    code,param = generate_makefile( project, make_target )
    print( "[MAKE]", make_path )
    with open( make_path, "w" ) as ofp:
        ofp.write( code )
    print( "[MAKE]", make_json_path )
    with open(make_json_path, 'w') as ofp:
        json.dump(param, ofp)
    
    stats={"update_vars":update_vars, "input_vars":input_vars, "input_pair_list":input_pair_list}
    return stats


def convert_data_file( project, folder, pair_data, input_from_file):
    data_fname = project + "_data.cpp"
    data_json_fname = project + "_data.json"
    data_path  = folder + "/" + data_fname
    data_json_path  = folder + "/" + data_json_fname

    # save data file
    code,param = generate_data_c_code( pair_data, input_from_file, folder)
    print( "[DATA]", data_path )
    with open( data_path, "w" ) as ofp:
        ofp.write( code )
    print( "[DATA]", data_json_path )
    with open(data_json_path, 'w') as ofp:
        json.dump(param, ofp)

def convert_train_code( project, folder, json_path, **kwargs ):
    train_fname = project + "_train.cpp"
    train_json_fname = project + "_train.json"
    train_path  = folder + "/" + train_fname
    train_json_path  = folder + "/" + train_json_fname
    #print(kwargs)

    # open json file
    print( "[LOAD JSON]", json_path )
    fp = open( json_path )
    obj = json.load(fp)

    # save train_cpp file
    code_train, param_train = generate_train_c_code( project, folder, obj, **kwargs )
    print("[TRAIN]", train_path )
    with open(train_path, "w" ) as ofp:
        ofp.write( code_train )
    print("[TRAIN]", train_json_path )
    with open(train_json_path, 'w') as ofp:
        json.dump(param_train, ofp)


def convert_all( project, folder, model, json_path, inputs, input_names,  data_dict={}, **kwargs ):
    
    os.makedirs(folder,exist_ok=True)
    rand_flag=0
    if "rand_flag" in kwargs:
        rand_flag=kwargs["rand_flag"]
    make_target="all"
    if "make_target" in kwargs:
        make_target=kwargs["make_target"]
    makefile_name="Makefile"
    if "makefile_name" in kwargs:
        makefile_name=kwargs["makefile_name"]
    seed_no = -1
    if "seed" in kwargs:
        seed_no = kwargs["seed"]
    chk_shape = 0
    if "chk_shape" in kwargs:
        chk_shape = kwargs["chk_shape"]
    if "shape" in kwargs:
        chk_shape = kwargs["shape"]
    
    input_from_file=False
    if "input_from_file" in kwargs:
        input_from_file = kwargs["input_from_file"]
        
    stats = convert_cpp_code(
        project, folder, model, inputs, input_names, json_path,
        rand_flag=rand_flag, seed_no=seed_no, chk_shape=chk_shape,
        make_target=make_target, makefile_name=makefile_name)
    kwargs_train = kwargs.copy()
    kwargs_train.update( data_dict )
    kwargs_train.update(stats)
    if make_target == "all":
        convert_data_file( project, folder, data_dict, input_from_file)
        convert_train_code( project, folder, json_path, **kwargs_train )

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
