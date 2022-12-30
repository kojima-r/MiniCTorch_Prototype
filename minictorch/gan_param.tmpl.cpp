#include <xtensor/xarray.hpp>

#define fprec float
typedef xt::xarray<fprec> Tensor;

// Dummy input data ({{discriminator_param.input_tensor_info.shape}})
Tensor {{discriminator_param.input_tensor_info.name}}_d = {{discriminator_param.input_tensor_info.tensor_code}};

// Dummy input data ({{generator_param.input_tensor_info.shape}})
Tensor {{generator_param.input_tensor_info.name}}_g = {{generator_param.input_tensor_info.tensor_code}};

// Tensor data
{%- for tensor_info in discriminator_param.tensor_list %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
{%- for node in tensor_info.node_list %}
// Node {{node.id}} : {{node.text}}
{%- endfor %}
Tensor {{tensor_info.name}} = {{tensor_info.tensor_code}};
{%- endfor %}

