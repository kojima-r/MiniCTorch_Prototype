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

extern Tensor xin;
{%- for var_name in extern_vars %}
extern Tensor {{var_name}};
{%- endfor %}

bool train_mode = true;

std::vector<Tensor> load_data_all();

void build_computational_graph( vector<MCTNode*>& forward_result, VariableTensor &input_var )
{
    {% for code in graph_codes %}
    {{code}}
    {% endfor %}
}    
void do_forward_backward_test( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N )
{
    cout<<"### forward computation ..."<<endl;
    for(int k=0;k<=N;k++) {
        if( forward_result[k] )  
        {
            {% if chk_shape > 0 %}
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
        {% if chk_shape > 0 %}
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
extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N );
#endif
  
int main(){
    vector<MCTNode*> forward_result({{graph_size}});
    // input data:  forward_result[ {{input_id}} ]
    Tensor::shape_type shape = { {{input_shape}} };
    xin.reshape( shape );
    VariableTensor input_var( xin, VAR_INPUT );
    
    {% if seed_no >= 0 %}
    xt::random::seed( {{seed_no}} );
    {% endif %}
    build_computational_graph( forward_result, input_var );
#ifdef _TRAIN
    auto vars = load_data_all();
    do_train_loop( forward_result, input_var, {{output_id}} );
#else
    do_forward_backward_test( forward_result, input_var, {{output_id}} );
#endif
    return 0;
}
    
   
