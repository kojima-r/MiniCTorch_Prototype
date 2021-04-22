from torch.utils.tensorboard._pytorch_graph import graph
from lark import Lark
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


lark_code = """
?dictionary: "{" [key_value ("," key_value)*] "}"

?list: "[" [value ("," value)*] "]"

// statement
?key_value: symbol ":" value

// expression
?value: fact
     | list
     | dictionary
     | function
function: symbol "(" [arg ("," arg)*] ")"
?arg: value
     | symbol "=" value

?fact: number
     | symbol
symbol: (WORD+) (LETTER+|DIGIT+|"-"|".")*
number: SIGNED_NUMBER

%import common.LETTER
%import common.WORD
%import common.DIGIT
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""


def get_tree(t):
    if not hasattr(t, "children"):
        return str(t)
    if t.data == "key_value":
        return {get_tree(t.children[0]): get_tree(t.children[1])}
    if t.data == "dictionary":
        d = {}
        for ch in t.children:
            d.update(get_tree(ch))
        return d
    if t.data == "function":
        func_data = {}
        func_data["name"] = get_tree(t.children[0])
        func_data["args"] = [
            get_tree(t.children[i + 1]) for i in range(len(t.children) - 1)
        ]
        return ("function", func_data)
    if t.data == "symbol":
        return get_tree(t.children[0])
    if t.data == "number":
        return float(get_tree(t.children[0]))
    if t.data == "arg":
        if len(t.children) == 2:
            return (get_tree(t.children[0]), get_tree(t.children[1]))
        else:
            return get_tree(t.children[0])
    if t.data == "list":
        d = []
        for ch in t.children:
            d.append(get_tree(ch))
        return d


def parse_attr(LP, s):
    t = LP.parse(s)
    s = get_tree(t)
    return s


def extract_graph(g):
    graph_data = {}
    LP = Lark(lark_code, start="dictionary", parser="lalr")
    for i in range(len(g[0].node)):
        data = {}
        name = str(g[0].node[i].name)
        data["name"] = name
        data["op"] = g[0].node[i].op
        data["in"] = [str(el) for el in g[0].node[i].input]
        ## decoding shape of the output tensor
        shape = g[0].node[i].attr["_output_shapes"].list.shape
        shape_data = []
        for el in shape:
            for x in el.dim:
                shape_data.append(x.size)
        data["shape"] = shape_data
        ## Parsing and decoding attributes for constant values
        attr = str(g[0].node[i].attr["attr"].s.decode())
        attr_data = None
        if attr != "" and attr != "{}":
            attr_data = parse_attr(LP, attr)
            if "value" in attr_data and type(attr_data["value"]) is float:
                data["constant_value"] = attr_data["value"]
            elif "value" in attr_data and type(attr_data["value"]) is tuple:
                if (
                    len(attr_data["value"]) == 2
                    and attr_data["value"][0] == "function"
                    and attr_data["value"][1]["name"] == "tensor"
                ):
                    if type(attr_data["value"][1]["args"][0]) is float:
                        data["constant_value"] = attr_data["value"][1]["args"][0]
                    else:
                        x = np.array(attr_data["value"][1]["args"][0])
                        data["constant_value"] = list(np.ravel(x))

        graph_data[name] = data
    return graph_data


def sort_and_assign_id(graph_data):
    # given graph_data
    for name, node in graph_data.items():
        for key in node["in"]:
            if key in graph_data:
                to_node = graph_data[key]
                if "out" not in to_node:
                    to_node["out"] = set()
                to_node["out"].add(name)
            else:
                print("skip:",key)
    start_nodes = []
    sorted_graph = []
    for name, node in graph_data.items():
        if "out" not in node:
            node["out"] = set()
            start_nodes.append(name)
    #print("start node:", start_nodes)

    def visit(name):
        nonlocal sorted_graph
        if name not in graph_data:
            return
        if "visited" in graph_data[name]:
            return
        else:
            node = graph_data[name]
            node["visited"] = True
            for next_node in node["in"]:
                visit(next_node)
            node["sorted_id"] = len(sorted_graph)
            sorted_graph.append(node)

    for name in start_nodes:
        visit(name)
    return sorted_graph


def generate_minictorch_file(model, input_to_model, filename):
    g = graph(model, input_to_model)
    graph_data = extract_graph(g)
    sorted_graph = sort_and_assign_id(graph_data)

    mapping_to_id = {node["name"]: i for i, node in enumerate(sorted_graph)}
    for node in sorted_graph:
        # encoding id
        out_id = [mapping_to_id[el] for el in node["out"]]
        node["out"] = out_id
        in_id=[]
        for el in node["in"]:
            if el in mapping_to_id:
                in_id.append(mapping_to_id[el])
        node["in"] = in_id
        # delete visited
        del node["visited"]
    sorted_graph
    with open(filename, "w") as fp:
        json.dump(sorted_graph, fp)


def main():
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()

        def forward(self, x):
            dist = torch.distributions.Normal(x, 1)
            return dist.sample()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.t = torch.tensor(np.array([[1.0, 2], [3, 4]]))
            self.sampler = Sampler()

            # epsilon = torchrandn.(mean.shape).to(device)
            # return mean + epsilon

        def forward(self, x):
            f1 = 10 * x * x * self.t
            f2 = 5 * x * self.sampler(x)
            y = f1 + f2
            return y

    model = Net()
    input_x = torch.tensor(np.array([[1.0, 2], [3, 4]]), requires_grad=True)
    input_to_model = torch.randn((2, 2))
    model.eval()
    with torch.no_grad():
        generate_minictorch_file(model, input_to_model, "sample.json")


if __name__ == "__main__":
    main()
