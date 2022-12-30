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

extern Tensor xin_g;
extern Tensor xin_d;

{%- for var_name in discriminator_main.extern_vars %}
extern Tensor {{var_name}};
{%- endfor %}

bool train_mode = true;

std::vector<Tensor> load_data_all();

void build_computational_graph_d( vector<MCTNode*>& forward_result, VariableTensor &input_var )
{
    {% for code in discriminator_main.graph_codes %}
    {{code}}
    {% endfor %}
}    

void build_computational_graph_g( vector<MCTNode*>& forward_result, VariableTensor &input_var )
{
    {% for code in generator_main.graph_codes %}
    {{code}}
    {% endfor %}
}    

void do_forward_backward_test( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N )
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
    cout<<"input_grad"<<input_var.grad<<endl;
}
    
#ifdef _TRAIN
extern void do_train_loop(
        vector<MCTNode*>& forward_result, VariableTensor &input_var, int N,
        vector<MCTNode*>& forward_result_g, VariableTensor &input_var_g, int N_g);
#endif
  
int main(){
    vector<MCTNode*> c_graph_d({{discriminator_main.graph_size}});
    // input data:  c_graph[ {{discriminator_main.input_id}} ]
    Tensor::shape_type shape_d = { {{discriminator_main.input_shape}} };
    xin_d.reshape( shape_d );
    VariableTensor input_var_d( xin_d, VAR_INPUT );
    
    vector<MCTNode*> c_graph_g({{generator_main.graph_size}});
    // input data:  c_graph[ {{generator_main.input_id}} ]
    Tensor::shape_type shape_g = { {{generator_main.input_shape}} };
    xin_g.reshape( shape_g );
    VariableTensor input_var_g( xin_g, VAR_INPUT );

    {% if generator_main.seed_no >= 0 %}
    xt::random::seed( {{generator_main.seed_no}} );
    {% endif %}

    build_computational_graph_g( c_graph_g, input_var_g );
    build_computational_graph_d( c_graph_d, input_var_d );
#ifdef _TRAIN
    auto vars = load_data_all();
    do_train_loop(
        c_graph_g, input_var_g, {{generator_main.output_id}},
        c_graph_d, input_var_d, {{discriminator_main.output_id}});
#else
    do_forward_backward_test( c_graph_g, input_var_g, {{generator_main.output_id}} );
#endif
    return 0;
}
    
   
