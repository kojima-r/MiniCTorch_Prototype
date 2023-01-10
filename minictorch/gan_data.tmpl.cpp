#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

#define fprec float
typedef xt::xarray<fprec> Tensor;

// data
//Tensor input_data;
{%- for tensor_info in discriminator_data.tensor_list %}
{%- if  discriminator_data.input_from_file %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
Tensor {{tensor_info.name}}_d;
{%- else %}
Tensor {{tensor_info.name}}_d = {{tensor_info.tensor_code}};
{%- endif %}
{%- endfor %}

// data
//Tensor input_data;
{%- for tensor_info in generator_data.tensor_list %}
{%- if  generator_data.input_from_file %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
Tensor {{tensor_info.name}}_g;
{%- else %}
Tensor {{tensor_info.name}}_g = {{tensor_info.tensor_code}};
{%- endif %}
{%- endfor %}


std::vector<Tensor> load_data_all(){
    std::vector<Tensor> out;    
    //discriminator
    {%- if discriminator_data.input_from_file %}
    {%- for tensor_info in discriminator_data.tensor_list %}
    {
        {{tensor_info.name}}_d = xt::load_npy<fprec>("{{tensor_info.filename}}");
        std::cout<<"[LOAD] {{tensor_info.name}}_d ( {{tensor_info.shape}} )"<<std::endl;
        out.push_back({{tensor_info.name}}_d);
    }
    {%- endfor %}
    {%- endif %}
    //generator
    {%- if generator_data.input_from_file %}
    {%- for tensor_info in generator_data.tensor_list %}
    {
        {{tensor_info.name}}_g = xt::load_npy<fprec>("{{tensor_info.filename}}");
        std::cout<<"[LOAD] {{tensor_info.name}}_g ( {{tensor_info.shape}} )"<<std::endl;
        out.push_back({{tensor_info.name}}_g);
    }
    {%- endfor %}
    {%- endif %}

    {%- for tensor_info in discriminator_data.tensor_list %}
    {%- if not discriminator_data.input_from_file %}
    // {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
    {{tensor_info.name}}_d.reshape({ {{tensor_info.shape_str}}  });
    {%- endif %}
    {%- endfor %}

    {%- for tensor_info in generator_data.tensor_list %}
    {%- if not generator_data.input_from_file %}
    // {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
    {{tensor_info.name}}_g.reshape({ {{tensor_info.shape}}  });
    {%- endif %}
    {%- endfor %}
    return out;
};

