import re
import json
import argparse
import pydotplus
import pathlib

"""
# ex. Net/Linear[fc1]/weight/43 -> param_fc1_weight
#     VAE/Net[net]/Linear[fc1]/weight/158 -> param_fc1_weight
### v1
def get_param_name( attr_name ):
    sep_list = attr_name.split('/')
    n = 1
    for i in range(len(sep_list)):
        if sep_list[i].find("[")>=0:
            n = i
    keys = re.findall("(?<=\[).+?(?=\])", sep_list[n])
    out_name = keys[0] + '_' + sep_list[n+1]
    return out_name
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
    return attr_name
 
def title( w, no ):
   #return w + "[" + str(no) + "]"
   return w + " (" + str(no) + ")"

def dot_var( id, name ):
   dot_var_ ='{} [label="{}", color=orange, style=filled]\n'
   return dot_var_.format( id, name )

def dot_func( id, name ):
   dot_func_='{} [label="{}", color=lightblue, style=filled, shape=box]\n'
   return dot_func_.format( id, name )

def dot_edge( id1, id2 ):
   dot_edge_='{} -> {}\n'
   return dot_edge_.format( id1, id2 )

def dot_graph( project, obj ):

   txt = ""
   cno = 0
   for i, el in enumerate(obj):
      name = el["name"]
      no = i+1
      t = " "
      if el["op"]=="IO Node":

         if "input" in el["name"]: 
            t = title("Input",i)
            txt += dot_var(no,t)
         elif "output" in el["name"]: 
            t = title("Output",i)
            txt += dot_var(no,t)
            for k in range(len(el["in"])):
               l = el["in"][k]+1
               txt += dot_edge(l,no)

      elif el["op"]=="prim::Constant":

         key=""
         if "constant_value" not in el:
             key = "Null"
         elif len(el["shape"])==0:
             val = el["constant_value"]
             key = str(val)
         else:
             if len(el["shape"]) > 0:
                 cno += 1
                 key = "Constant" + str(cno)
             else:
                 val = el["constant_value"]
                 key = str(val)
         if len(key) > 0:
             t = title( "Constant:"+key,i)
             txt += dot_var(no,t)

      elif el["op"]=="prim::GetAttr":

         key = get_param_name( name )
         t = title("Attr:"+key,i)
         #print("Attr:"+key)
         txt += dot_var(no,t)
         
      else:

         ekey = el["op"].split(":")
         #print("ekey",ekey)
         if ekey[0] in ["prim","aten"]:
             t=title( ekey[2],i)
         else:
             t=title( el["op"],i)

         if len(t) > 0:
            txt += dot_func(no,t)
            for k in range(len(el["in"])):
               l = el["in"][k]+1
               txt += dot_edge(l,no)

      #print("i,t",i,t,"---", el["op"],el["in"],el["out"],marks[i+1])

   return  'digraph g{\n' + 'graph[label=' + project + ', labelloc="t"];\n' + txt+"\n}\n"

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
    p_file = pathlib.Path(filename)
    png_path = pathlib.Path(p_file.parent,p_file.stem+"_graph.png")
    graph_path = pathlib.Path(p_file.parent,p_file.stem+"_graph.dot")

    #json_path  = folder + project + ".json"
    #png_path   = folder + project + '_graph.png'
    #graph_path = folder + project + '_graph.dot'

    fp = open( filename )
    obj = json.load( fp )

    g = dot_graph( args.project, obj )

    print('json : ', filename )
    print('dot  : ', graph_path )
    print('png  : ', png_path )
    #print(g)

    with open( graph_path, 'w' ) as f:
       f.write( g )
    graph = pydotplus.graphviz.graph_from_dot_file( graph_path )
    graph.write_png(str(png_path))


if __name__ == "__main__":
    main()
