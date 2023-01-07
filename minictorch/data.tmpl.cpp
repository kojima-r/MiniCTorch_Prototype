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
{%- for tensor_info in tensor_list %}
{%- if input_from_file %}
// {{tensor_info.name}} ({{tensor_info.length}}) (shape: {{tensor_info.shape}})
Tensor {{tensor_info.name}};
{%- else %}
Tensor {{tensor_info.name}} = {{tensor_info.tensor_code}};
{%- endif %}
{%- endfor %}

std::vector<Tensor> load_data_all(){
    //
    std::vector<Tensor> out;    
    {%- if input_from_file %}
    {%- for tensor_info in tensor_list %}
    {
        {{tensor_info.name}} = xt::load_npy<fprec>("{{tensor_info.filename}}");
        std::cout<<"[LOAD] {{tensor_info.name}} ( {{tensor_info.shape}} )"<<std::endl;
        out.push_back({{tensor_info.name}});
    }
    {%- endfor %}
    {%- endif %}
    return out;
};

