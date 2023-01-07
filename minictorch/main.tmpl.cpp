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

{%- for input_name, var in input_vars.items() %}
extern Tensor {{input_name}};
{%- endfor %}
{%- for var_name in extern_vars %}
extern Tensor {{var_name}};
{%- endfor %}

bool train_mode = true;

std::vector<Tensor> load_data_all();
void build_computational_graph( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars)
{
    {% for code in graph_codes %}
    {{code}}
    {% endfor %}
}    
void do_forward_backward_test( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars, int N)
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
    for(int k=0; k<input_vars.size();k++){
        cout<<"input_grad:"<<input_vars[0]->grad<<endl;
    }
}
    
#ifdef _TRAIN
extern void do_train_loop( vector<MCTNode*>& forward_result, vector<VariableTensor*>& input_vars, int N);
#endif
  
int main(){
    vector<MCTNode*> c_graph({{graph_size}});
    vector<VariableTensor*> input_vars;
    // input data:  c_graph[ {{input_id}} ]
    {%- for input_name, var in input_vars.items() %}
    Tensor::shape_type {{input_name}}_shape = { {{var.shape_str}} };
    {{input_name}}.reshape( {{input_name}}_shape );
    VariableTensor* {{input_name}}_var=new VariableTensor( {{input_name}}, VAR_INPUT );
    input_vars.push_back({{input_name}}_var);
    {%- endfor %}
    
    {% if seed_no >= 0 %}
    xt::random::seed( {{seed_no}} );
    {% endif %}
    build_computational_graph( c_graph, input_vars );
#ifdef _TRAIN
    auto vars = load_data_all();
    do_train_loop( c_graph, input_vars, {{output_id}} );
#else
    do_forward_backward_test( c_graph, input_vars, {{output_id}} );
#endif
    return 0;
}
    
   
