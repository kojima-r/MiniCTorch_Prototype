#include <xtensor/xarray.hpp>

#define fprec float
typedef xt::xarray<fprec> Tensor;

{%- for input_tensor_info in input_tensor_list %}
// Dummy input data ({{input_tensor_info.shape}})
Tensor {{input_tensor_info.name}} = {{input_tensor_info.tensor_code}};
{%- endfor %}
// Tensor data
{%- for tensor_info in tensor_list %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
{%- for node in tensor_info.node_list %}
// Node {{node.id}} : {{node.text}}
{%- endfor %}
Tensor {{tensor_info.name}} = {{tensor_info.tensor_code}};
{%- endfor %}

