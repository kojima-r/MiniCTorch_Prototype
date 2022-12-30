import re
import json
import argparse
import pydotplus
import pathlib
import os
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=str, help="computational graph json")
    parser.add_argument(
        "--project","-p", type=str, default="example", help="project name [example]"
    )
    parser.add_argument(
        "--output","-o", type=str, default="output", nargs="?", help="output path [output]"
    )
    parser.add_argument(
        "--tmpl","-t", type=str, default=None, nargs="?", help="template"
    )
    parser.add_argument(
        "--template","-x", type=str, default=None, nargs="?", help="template:   gan.tmpl.cpp  gan_train.tmpl.cpp  gan_data.tmpl.cpp  gan_param.tmpl.cpp"
    )

    args = parser.parse_args()
    
    # load json file
    all_params={}
    for filename in args.json_files:
        print("[LOAD JSON]", filename)
        fp = open( filename )
        param = json.load(fp)
        name=pathlib.Path(filename).stem
        all_params[name]=param
    
    if args.tmpl is None:
        path=os.path.dirname(__file__)
        env = Environment(loader=FileSystemLoader(path+'/', encoding='utf8'))
        if args.template is not None:
            data_tmpl = env.get_template(args.template)
        else:
            data_tmpl = "gan.data.tmpl"
            data_tmpl = env.get_template(data_template)
    else:
        env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
        data_tmpl = env.get_template(args.tmpl)
    rendered_s = data_tmpl.render(all_params)
    #print(rendered_s)
    if args.output is not None:
        with open(args.output,"w") as ofp:
            ofp.write(rendered_s)
        #return rendered_s, params
    
    

   
if __name__ == "__main__":
    main()
