//
//  {{proj}}_train
//
#ifdef _NOTEBOOK
#include "../../src/minictorch.hpp"
#else
#include "minictorch.hpp"
#endif
#include <chrono>

extern bool train_mode;
{%- if input_enabled %}
{%- for input_name, var in input_vars.items() %}
extern Tensor {{input_name}};
{%- endfor %}
{%- for input_name, var in input_vars.items() %}
extern Tensor {{var.data_name}};
{%- endfor %}
{%- endif %}
{%- if target_enabled %}
extern Tensor target_data;
{%- endif %}

void do_forward( vector<MCTNode*> &target_c_graph, int n ) {
    for(int k=0;k<=n;k++) {
        if( target_c_graph[k] )  target_c_graph[k]->forward();
    }
};

void do_backward( vector<MCTNode*> &target_c_graph, int n ) {
    target_c_graph[n]->grad = xt::ones_like( target_c_graph[n]->output );
    for(int k=n;k>=0;k--) {
        if( target_c_graph[k] ) target_c_graph[k]->backward();
    }
};

void do_zerograd( vector<MCTNode*> &target_c_graph, int n ) {
    for(int k=0;k<=n;k++) {
        if( target_c_graph[k] )  target_c_graph[k]->zerograd();
    }
};
/*
void update_all_param( vector<MCTNode*> &target_c_graph, int n, fprec lr=0.01 ){
    for(int k=0;k<=n;k++) {
        if( target_c_graph[k] )  target_c_graph[k]->update( lr );
    }
};
*/
void update_params( vector<MCTNode*> &target_c_graph, int n, fprec lr=0.01 ) {
    vector<int> var_list={
        {%- for k, v in update_vars.items() %}
        {{v}}, //{{k}}
        {%- endfor %}
    };
    for(int i=0;i<var_list.size();i++) {
        int j=var_list[i];
        if( target_c_graph[j] )  target_c_graph[j]->update( lr );
    }
};
    
{%- if eval_enabled %}
int eval_labels( Tensor& y, Tensor &t ){
    auto lb = xt::argmax( y, 1 );
    auto eq = xt::equal( t, lb );
    auto sm = xt::sum( eq );
    return (int)sm[0];
};
{%- endif %}


void do_train_loop( vector<MCTNode*>& c_graph, vector<VariableTensor*>& input_vars, int output_id){

    xt::random::seed(1);
    
    fprec lr = {{lr}};
    int epoch_num = {{epochs}};
    cout<<"epoch_num : "<<epoch_num<<endl;
    int data_num=1;
    {%- if input_enabled %}
    {%- for input_name, var in input_vars.items() %}
    {
        {{input_name}}.reshape({ {{var.shape_str}} });
        auto input_shape = {{input_name}}.shape();
        if(input_shape.size()>0 && input_shape[0]>data_num){
            data_num=input_shape[0];
        }
    }
    {%- endfor %}
    {%- endif %}

    {%- if target_enabled %}
    target_data.reshape( {{target_shape}} );
    auto target_shape = target_data.shape();
    {%- endif %}

    int batch_size = {{batch_size}};
    int n_batch = (int)data_num / batch_size;
    cout<<"batch size  : "<<batch_size<<endl;
    cout<<"#batch  : "<<n_batch<<endl;
    cout<<"learning rate : "<<lr<<endl;

    //input_vars.output = input_data;
    
    //Tensor x_tmp = xt::zeros<fprec>({ {{ x_shape }} });
    {%- if target_enabled %}
    Tensor y_tmp = xt::zeros<fprec>({ {{ y_shape }} });
    {%- endif %}

    ofstream outputfile("{{output_filename}}");
    std::chrono::system_clock::time_point  start, end; 
    start = std::chrono::system_clock::now();

    do_zerograd( c_graph, output_id );
    for(int epoch=0;epoch<epoch_num;epoch++){
        train_mode = true;
        xt::xarray<int> index = xt::arange( (int)data_num );
        {%- if shuffle %}
        xt::random::shuffle( index );
        {%- endif %}

        fprec total_loss = 0.0;
        {%- if classification_task %}
        int   total_corrects = 0;
        {%- endif %}
        for(int j=0;j<n_batch;j++){
            int jb = j * batch_size;

            {%- for input_name, var in input_vars.items() %}
            Tensor tmp_{{input_name}} = xt::zeros<fprec>({ {{ var.shape_str }} });
            {%- endfor %}
            for(int k=0;k<batch_size;k++){
                {%- for input_name, var in input_vars.items() %}
                {
                    auto xw = xt::view( {{var.data_name}}, index(jb+k){{var.shape_rest}} );
                    xt::view( tmp_{{input_name}}, k{{var.shape_rest}} ) = xw;

                    {%- if y_shape_rest_enabled %}
                    auto yw = xt::view( target_data, index(jb+k){{y_shape_rest}} );
                    xt::view( y_tmp, k{y_shape_rest} ) = yw;
                    {%- elif target_enabled %}
                    y_tmp( k ) = target_data( index(jb+k) );
                    {%- endif %}
                }
                {%- endfor %}
            }
            {%- for input_name, var in input_vars.items() %}
            input_vars[{{var.index}}]->output = tmp_{{input_name}};
            {%- endfor %}
            {%- if target_no > 0: %}
            c_graph[{{target_no}}]->output = y_tmp;
            {%- endif %}
            do_forward( c_graph, output_id );
            
            auto o = c_graph[output_id]->output;
            total_loss += o[0];
            
            {%- if classification_task %}
            int corrects = eval_labels( c_graph[{{pred_no}}]->output, y_tmp );
            total_corrects += corrects;
            {%- endif %}
        
            do_backward( c_graph, output_id );
            update_params( c_graph, output_id, lr );
            do_zerograd( c_graph, output_id );
        }
        {%- if classification_task %}
        fprec total_acc = (fprec)total_corrects / (fprec)data_num;
        cout<<"total_loss : epoch "<<epoch 
            <<" : total_loss "<<total_loss 
            <<" : batch_average_loss "<<total_loss/n_batch 
            <<" : average_loss "<<total_loss/(n_batch*batch_size) 
            <<" : Acc "<<total_acc 
            <<" ("<<total_corrects<<"/"<<data_num<<")" 
            <<endl;
        {%- else %}
        cout<<"total_loss : epoch "<<epoch
            <<" : total_loss "<<total_loss
            <<" : batch_average_loss "<<total_loss/n_batch 
            <<" : average_loss "<<total_loss/(n_batch*batch_size) 
            <<endl;
        {%- endif %}
        
        /*
        train_mode = false;
        {%- for input_name, var in input_vars.items() %}
        input_vars[{{var.index}}]->output = {{input_name}};
        {%- endfor %}
     

        {%- if target_no > 0 %}
        c_graph[{{target_no}}]->output = target_data;
        {%- endif%}
        
        do_forward( c_graph, output_id );
        auto o = c_graph[output_id]->output;
        
        {%- if double_loss_enabled %}
        auto o1 = c_graph[{{loss1_no}}]->output;
        auto o2 = c_graph[{{loss2_no}}]->output; 
        cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
        outputfile<<to_string(o[0])<<endl;
        {%- endif %}

        {%- if classification_task %}
        int corrects = eval_labels( c_graph[{{pred_no}}]->output, target_data );
        fprec acc = (fprec)corrects / (fprec)data_num;
        cout<<"full_data : epoch "<<epoch<<" : loss "<<o[0]<<" : Acc "<<acc<<" "<<corrects<<endl;
        outputfile<<to_string(o[0])<<","<<to_string(acc)<<","<<total_loss<<endl;
        {%- else %}
        cout<<"full_data : epoch "<<epoch<<" : loss "<<o[0]<<endl;
        outputfile<<to_string(o[0])<<endl;
        {%- endif %}
        */
    }
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout<<"MiniCTorch Time: "<<elapsed<<endl;

    outputfile.close();
    
    train_mode = false;
    {%- if pred_no>=0 %}
    {
        // {{pred_no}} : {pred_op}
        /*
        {%- if input_enabled %}
        {%- for input_name, var in input_vars.items() %}
        input_vars[{{var.index}}]->output = {{input_name}};
        {%- endfor %}
        {%- endif %}
        */
        do_forward( c_graph, {{pred_no}} );
        auto y_pred = c_graph[ {{pred_no}} ]->output;
        
        ofstream outputfile( "{{pred_filename}}" );
        {%- if pred_type == 1 %}
        outputfile<<to_string(data_num)<<",1"<<endl;
        for(int i=0;i<data_num;i++)
        {
            outputfile<<to_string(y_pred(i,0))<<endl;
        }
        {%- else %}
        //int nx = input_shape[0];
        int nx = {{pred_output}};
        outputfile<<to_string(nx)<<","<<to_string(target_shape[1])<<endl;
        
        for(int i=0;i<nx;i++)
        {
            for(int j=0;j<target_shape[1]-1;j++)
            {
                outputfile<<to_string(y_pred(i,j))<<",";
            }
            outputfile<<to_string(y_pred(i,target_shape[1]-1))<<endl;
        }
        {%- endif %}
        outputfile.close();

    }
    {%- endif %}
    {%- if z_no >= 0 %}
    {
        // {{z_no}} : z output
        auto z_pred = c_graph[{{z_no}}]->get_output();
    
        ofstream outputfile( "{{z_filename}}" );
        outputfile<<to_string(input_shape[0])<<","<<to_string(2)<<endl;
    
        for(int k=0;k<input_shape[0];k++)
        {
            outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
        }
        outputfile.close();
    }
    {%- endif %}
}
	
