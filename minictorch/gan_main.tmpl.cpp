//
//  {{title}}
//
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include "minictorch.hpp"

using namespace std;

{%- for input_name, var in discriminator_main.input_vars.items() %}
extern Tensor {{input_name}}_d;
{%- endfor %}
{%- for input_name, var in generator_main.input_vars.items() %}
extern Tensor {{input_name}}_g;
{%- endfor %}

{%- for var_name in discriminator_main.extern_vars %}
extern Tensor {{var_name}};
{%- endfor %}

bool train_mode = true;

std::vector<Tensor> load_data_all();

void build_computational_graph_d( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars)
{
    {% for code in discriminator_main.graph_codes %}
    {{code}}
    {% endfor %}
}    

void build_computational_graph_g( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars)
{
    {% for code in generator_main.graph_codes %}
    {{code}}
    {% endfor %}
}    

void do_forward_backward_test( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars, int N )
{
    cout<<"### forward computation ..."<<endl;
    for(int k=0;k<=N;k++) {
        if( forward_result[k] )  
        {
            {% if discriminator_main.chk_shape > 0 %}
            forward_result[k]->set_id( k );
            forward_result[k]->forward();
            forward_result[k]->display_shape();
            forward_result[k]->zerograd();
            {% else %}
            //forward_result[k]->set_id( k );
            forward_result[k]->forward();
            forward_result[k]->zerograd();
            {% endif %}
        }
    }
    auto o = forward_result[N]->output;
    cout<<o<<endl;

    cout<<"### backward computation ..."<<endl;
    forward_result[N]->grad = xt::ones_like( forward_result[N]->output );
    for(int k=N;k>=0;k--) {
        {% if discriminator_main.chk_shape > 0 %}
        if( forward_result[k] )  
        {
           forward_result[k]->backward();
           forward_result[k]->display_grad_shape();
        }
        {% else %}
        if( forward_result[k] )  forward_result[k]->backward();
        {% endif %}
    }
    for(int k=0; k<input_vars.size();k++){
        cout<<"input_grad:"<<input_vars[0]->grad<<endl;
    }
}
    
#ifdef _TRAIN
extern void do_train_loop(
        vector<MCTNode*>& forward_result_d,vector<VariableTensor*> &input_var_d, int N,
        vector<MCTNode*>& forward_result_g,vector<VariableTensor*> &input_var_g, int N_g);
#endif
  
int main(){
    vector<MCTNode*> c_graph_d({{discriminator_main.graph_size}});
    vector<VariableTensor*> input_vars_d;
    {%- for input_name, var in discriminator_main.input_vars.items() %}
    // input data:  c_graph[ {{var.input_id}} ]
    Tensor::shape_type {{input_name}}_d_shape = { {{var.shape_str}} };
    {{input_name}}_d.reshape( {{input_name}}_d_shape );
    VariableTensor* {{input_name}}_d_var=new VariableTensor( {{input_name}}_d, VAR_INPUT );
    input_vars_d.push_back({{input_name}}_d_var);
    {%- endfor %}
    
    vector<MCTNode*> c_graph_g({{generator_main.graph_size}});
    vector<VariableTensor*> input_vars_g;
    {%- for input_name, var in generator_main.input_vars.items() %}
    // input data:  c_graph[ {{var.input_id}} ]
    Tensor::shape_type {{input_name}}_g_shape = { {{var.shape_str}} };
    {{input_name}}_g.reshape( {{input_name}}_g_shape );
    VariableTensor* {{input_name}}_g_var=new VariableTensor( {{input_name}}_g, VAR_INPUT );
    input_vars_g.push_back({{input_name}}_g_var);
    {%- endfor %}
    


    {% if generator_main.seed_no >= 0 %}
    xt::random::seed( {{generator_main.seed_no}} );
    {% endif %}

    build_computational_graph_g( c_graph_g, input_vars_g );
    build_computational_graph_d( c_graph_d, input_vars_d );
#ifdef _TRAIN
    auto vars = load_data_all();
    do_train_loop(
        c_graph_d, input_vars_d, {{discriminator_main.output_id}},
        c_graph_g, input_vars_g, {{generator_main.output_id}});
#else
    do_forward_backward_test( c_graph_g, input_vars_g, {{generator_main.output_id}} );
#endif:w

    return 0;
}
    
   
