#include <xtensor/xarray.hpp>

#define fprec float
typedef xt::xarray<fprec> Tensor;

{%- for tensor_info in discriminator_param.input_tensor_list %}
// Dummy input data ({{tensor_info.shape}})
Tensor {{tensor_info.name}}_d = {{tensor_info.tensor_code}};
{%- endfor %}
{%- for tensor_info in generator_param.input_tensor_list %}
// Dummy input data ({{tensor_info.shape}})
Tensor {{tensor_info.name}}_g = {{tensor_info.tensor_code}};
{%- endfor %}

// Tensor data
{%- for tensor_info in discriminator_param.tensor_list %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
{%- for node in tensor_info.node_list %}
// Node {{node.id}} : {{node.text}}
{%- endfor %}
Tensor {{tensor_info.name}} = {{tensor_info.tensor_code}};
{%- endfor %}

