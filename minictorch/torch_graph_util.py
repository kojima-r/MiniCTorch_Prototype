##
## This file was extention of _pytorch_graph in pytorch
##
from torch.utils.tensorboard._pytorch_graph import *


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [1, ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        # Kind attribute string is purely descriptive and will be shown
        # in detailed information for the node in TensorBoard's graph plugin.
        #
        # NodePyOP nodes get this from their kind() method.
        self.kind = 'Parameter'
        self.output_id=0
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'

class GraphPy(object):
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization
    with TensorBoard.
    GraphDef generation operates in two passes:
    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.
    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for i,out_info in enumerate(zip(node.outputs, node.outputstensor_size)):
                node_output, outputSize=out_info
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output,
                                                      node.inputs,
                                                      node.scopeName,
                                                      outputSize,
                                                      op_type=node.kind,
                                                      attributes=node.attributes)
                self.nodes_io[node_output].output_id=i

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.debugName

            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = self.shallowest_scope_name + '/' + node.debugName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[node.debugName]

    def to_proto(self):
        """
        Converts graph representation of GraphPy object to TensorBoard
        required format.
        """
        # TODO: compute correct memory usage and CPU time once
        # PyTorch supports it
        nodes = []
        for v in self.nodes_io.values():
            nodes.append(
                {
                 "name":v.debugName,
                 "op":v.kind,
                 "input":v.inputs,
                 "output_id":v.output_id,
                 "outputsize":v.tensor_size,
                 "attributes":v.attributes,
                })
        return nodes

def parse(graph, trace, args=None, omit_useless_nodes=True):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.
    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = len(args)

    scope = {}
    nodes_py = GraphPy()
    for node in graph.inputs():
        if omit_useless_nodes:
            if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, 'input'))

    attr_to_scope: Dict[Any, str] = dict()
    for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            attr_name = node.s('name')
            parent = node.input().node()
            if parent.kind() == GETATTR_KIND:  # If the parent node is not the top-level "self" node
                parent_attr_name = parent.s('name')
                parent_scope = attr_to_scope[parent_attr_name]
                attr_scope = parent_scope.split('/')[-1]
                attr_to_scope[attr_name] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
            else:
                attr_to_scope[attr_name] = '__module.{}'.format(attr_name)
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[attr_name]  # type: ignore[attr-defined]
                nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))

    for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
        node_pyio = NodePyIO(node, 'output')
        node_pyio.debugName = "output.{}".format(i + 1)
        node_pyio.inputs = [node.debugName()]
        nodes_py.append(node_pyio)
    def parse_traced_name(module):
        if isinstance(module, torch.jit.TracedModule):
            module_name = module._name
        else:
            module_name = getattr(module, 'original_name', "Module")
        return module_name

    alias_to_name = dict()
    base_name = parse_traced_name(trace)
    for name, module in trace.named_modules(prefix='__module'):
        mod_name = parse_traced_name(module)
        attr_name = name.split('.')[-1]
        alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)
      
    for node in nodes_py.nodes_op:
        module_aliases = node.scopeName.split('/')
        replacements = [
            alias_to_name[alias]
            if alias in alias_to_name
            else alias.split('.')[-1]
            for alias in module_aliases
        ]
        node.scopeName = base_name
        if any(replacements):
            node.scopeName += '/' + '/'.join(replacements)

    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


