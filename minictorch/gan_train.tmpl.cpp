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
{%- for input_name, var in discriminator_train.input_vars.items() %}
extern Tensor {{input_name}}_d;
{%- endfor %}
{%- for input_name, var in discriminator_train.input_vars.items() %}
extern Tensor {{var.data_name}}_d;
{%- endfor %}

{%- for input_name, var in generator_train.input_vars.items() %}
extern Tensor {{input_name}}_g;
{%- endfor %}
{%- for input_name, var in generator_train.input_vars.items() %}
extern Tensor {{var.data_name}}_g;
{%- endfor %}

{%- if discriminator_train.itarget_enabled %}
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
void update_params_d( vector<MCTNode*> &target_c_graph, int n, fprec lr=0.01 ) {
    vector<int> var_list={
        {%- for k, v in discriminator_train.update_vars.items() %}
        {{v}}, //{{k}}
        {%- endfor %}
    };
    for(int i=0;i<var_list.size();i++) {
        int j=var_list[i];
        if( target_c_graph[j] )  target_c_graph[j]->update( lr );
    }
};
void update_params_g( vector<MCTNode*> &target_c_graph, int n, fprec lr=0.01 ) {
    vector<int> var_list={
        {%- for k, v in generator_train.update_vars.items() %}
        {{v}}, //{{k}}
        {%- endfor %}
    };
    for(int i=0;i<var_list.size();i++) {
        int j=var_list[i];
        if( target_c_graph[j] )  target_c_graph[j]->update( lr );
    }
};
    
{%- if eval_enabled %}
void eval_labels( Tensor& y, Tensor &t ){
    auto lb = xt::argmax( y, 1 );
    auto eq = xt::equal( t, lb );
    auto sm = xt::sum( eq );
    return (int)sm[0];
};
{%- endif %}



void do_train_loop(
        vector<MCTNode*>& c_graph_d, vector<VariableTensor*>& input_vars_d, int output_id_d,
        vector<MCTNode*>& c_graph_g, vector<VariableTensor*>& input_vars_g, int output_id_g){
    xt::random::seed(1);
    
    fprec lr = {{discriminator_train.lr}};
    int epoch_num = {{discriminator_train.epochs}};
    cout<<"epoch_num : "<<epoch_num<<endl;
    int data_num=0;

    {%- for input_name, var in discriminator_train.input_vars.items() %}
    {
        {{input_name}}_d.reshape({ {{var.shape_str}} });
        auto input_shape = {{input_name}}_d.shape();
        if(input_shape.size()>0 && input_shape[0]>data_num){
            data_num=input_shape[0];
        }
    }
    {%- endfor %}
    
    {%- for input_name, var in generator_train.input_vars.items() %}
    {
        {{input_name}}_g.reshape({ {{var.shape_str}} });
        auto input_shape = {{input_name}}_d.shape();
        if(input_shape.size()>0 && input_shape[0]>data_num){
            data_num=input_shape[0];
        }
    }
    {%- endfor %}
    
    int batch_size = {{discriminator_train.batch_size}};
    int n_batch = (int)data_num / batch_size;
    cout<<"batch size  : "<<batch_size<<endl;
    cout<<"#batch  : "<<n_batch<<endl;
    cout<<"learning rate : "<<lr<<endl;

    //input_var_d.output = input_data_d;
    //input_var_g.output = input_data_g;
    
    /*
    Tensor x_tmp_d = xt::zeros<fprec>({ {{ discriminator_train.x_shape }} });
    {%- if discriminator_train.target_enabled %}
    Tensor y_tmp_d = xt::zeros<fprec>({ {{ discriminator_train.y_shape }} });
    {%- endif %}
    Tensor x_tmp_g = xt::zeros<fprec>({ {{ generator_train.x_shape }} });
    {%- if generator_train.target_enabled %}
    Tensor y_tmp_g = xt::zeros<fprec>({ {{ generator_train.y_shape }} });
    {%- endif %}
    */

    ofstream outputfile("{{output_filename}}");
    std::chrono::system_clock::time_point  start, end; 
    start = std::chrono::system_clock::now();

    do_zerograd( c_graph_d, output_id_d );
    do_zerograd( c_graph_g, output_id_g );
    for(int epoch=0;epoch<epoch_num;epoch++){
        train_mode = true;
        xt::xarray<int> index = xt::arange( (int)data_num );
        {%- if discriminator_train.batch_enabled and discriminator_train.shuffle %}
        xt::random::shuffle( index );
        {%- endif %}

        {%- for input_name, var in discriminator_train.input_vars.items() %}
        Tensor tmp_{{input_name}}_d = xt::zeros<fprec>({ {{ var.shape_str }} });
        {%- endfor %}
        {%- for input_name, var in generator_train.input_vars.items() %}
        Tensor tmp_{{input_name}}_g = xt::zeros<fprec>({ {{ var.shape_str }} });
        {%- endfor %}
        
        fprec total_loss_d = 0.0;
        fprec total_loss_g = 0.0;
        {%- if discriminator_train.classification_task %}
        int   total_corrects = 0;
        {%- endif %}
        for(int j=0;j<n_batch;j++){
            int jb = j * batch_size;
            for(int k=0;k<batch_size;k++){
                {%- for input_name, var in discriminator_train.input_vars.items() %}
                {
                    auto xw = xt::view( {{var.data_name}}_d, index(jb+k){{var.shape_rest}} );
                    xt::view( tmp_{{input_name}}_d, k{{var.shape_rest}} ) = xw;

                    {%- if y_shape_rest_enabled %}
                    auto yw = xt::view( target_data, index(jb+k){{y_shape_rest}} );
                    xt::view( y_tmp_d, k{y_shape_rest} ) = yw;
                    {%- elif target_enabled %}
                    y_tmp_d( k ) = target_data( index(jb+k) );
                    {%- endif %}
                }
                {%- endfor %}
                {%- for input_name, var in generator_train.input_vars.items() %}
                {
                    auto xw = xt::view( {{var.data_name}}_g, index(jb+k){{var.shape_rest}} );
                    xt::view( tmp_{{input_name}}_g, k{{var.shape_rest}} ) = xw;

                    {%- if y_shape_rest_enabled %}
                    auto yw = xt::view( target_data, index(jb+k){{y_shape_rest}} );
                    xt::view( y_tmp_g, k{y_shape_rest} ) = yw;
                    {%- elif target_enabled %}
                    y_tmp_g( k ) = target_data( index(jb+k) );
                    {%- endif %}
                }
                {%- endfor %}
            }
            //input_var_d.output = x_tmp_d;
            //input_var_g.output = x_tmp_g;
            {%- for input_name, var in discriminator_train.input_vars.items() %}
            input_vars_d[{{var.index}}]->output = tmp_{{input_name}}_d;
            {%- endfor %}
            {%- for input_name, var in discriminator_train.input_vars.items() %}
            input_vars_g[{{var.index}}]->output = tmp_{{input_name}}_g;
            {%- endfor %}
            // discriminator
            {
                {%- if discriminator_train.target_no > 0 %}
                c_graph_d[{{ discriminator_train.target_no}}]->output = y_tmp_d;
                {%- endif %}
                do_forward( c_graph_d, output_id_d );
                
                auto o = c_graph_d[output_id_d]->output;
                total_loss_d += o[0];
                
                {%- if discriminator_train.classification_task %}
                int corrects = eval_labels( c_graph_d[{{discriminator_train.pred_no}}]->output, y_tmp );
                total_corrects += corrects;
                {%- endif %}
            
                do_backward( c_graph_d, output_id_d );
                update_params_d( c_graph_d, output_id_d, lr );
                do_zerograd( c_graph_d, output_id_d );
            }
            // generator
            {
                {%- if generator_train.target_no > 0 %}
                c_graph[{{ generator_train.target_no}}]->output = y_tmp;
                {%- endif %}
                do_forward( c_graph_g, output_id_g );
                
                auto o = c_graph_g[output_id_g]->output;
                total_loss_g += o[0];
                
                {%- if generator_train.classification_task %}
                int corrects = eval_labels( c_graph_d[{{generator_train.pred_no}}]->output, y_tmp );
                total_corrects += corrects;
                {%- endif %}
            
                do_backward( c_graph_g, output_id_g );
                update_params_g( c_graph_g, output_id_g, lr );
                do_zerograd( c_graph_g, output_id_g );
            }
        }
        cout<<"total_loss : epoch "<<epoch
            <<" : total_loss_D "<<total_loss_d
            <<" : total_loss_G "<<total_loss_g
            <<" : batch_average_loss_D "<<total_loss_d/n_batch
            <<" : batch_average_loss_G "<<total_loss_g/n_batch
            <<" : average_loss_D "<<total_loss_d/(n_batch*batch_size)
            <<" : average_loss_G "<<total_loss_g/(n_batch*batch_size)
            <<endl;
        /*
        train_mode = false;
        input_var_d.output = input_data_d;
        input_var_g.output = input_data_g;

        {%- if discriminator_train.target_no > 0 %}
        c_graph_d[{{discriminator_train.target_no}}]->output = target_data;
        {%- endif%}
        
        do_forward( c_graph_d, output_id_d );
        auto o = c_graph_d[output_id_d]->output;
        
        {%- if discriminator_train.double_loss_enabled %}
        auto o1 = c_graph_d[{{discriminator_train.loss1_no}}]->output;
        auto o2 = c_graph_d[{{discriminator_train.loss2_no}}]->output; 
        cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
        outputfile<<to_string(o[0])<<endl;
        {%- endif %}
        cout<<"full_data : epoch "<<epoch<<" : loss "<<o[0]<<endl;
        outputfile<<to_string(o[0])<<endl;
        */
    }
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout<<"MiniCTorch Time: "<<elapsed<<endl;

    outputfile.close();
    
    train_mode = false;
    {
        // {{pred_no}} : {pred_op}
        /*
        {%- if discriminator_train.input_enabled %}
        input_var_d.output = input_data_d;
        {%- endif %}
        */
        do_forward( c_graph_d, {{discriminator_train.pred_no}} );
        auto y_pred = c_graph_d[ {{discriminator_train.pred_no}} ]->output;
        
        ofstream outputfile( "{{discriminator_train.pred_filename}}" );
        {%- if discriminator_train.pred_type == 1 %}
        outputfile<<to_string(data_num)<<",1"<<endl;
        for(int i=0;i<data_num;i++)
        {
            outputfile<<to_string(y_pred(i,0))<<endl;
        }
        {%- else %}
        //int nx = input_shape[0];
        int nx = {{discriminator_train.pred_output}};
        outputfile<<to_string(nx)<<","<<to_string(target_shape_d[1])<<endl;
        
        for(int i=0;i<nx;i++)
        {
            for(int j=0;j<target_shape_d[1]-1;j++)
            {
                outputfile<<to_string(y_pred(i,j))<<",";
            }
            outputfile<<to_string(y_pred(i,input_shape_d[1]-1))<<endl;
        }
        {%- endif %}
        outputfile.close();

    }
    {%- if discriminator_train.z_no >= 0 %}
    {
        // {{z_no}} : z output
        auto z_pred = c_graph[{{discriminator_train.z_no}}]->get_output();
    
        ofstream outputfile( "{{discriminator_train.z_filename}}" );
        outputfile<<to_string(input_shape_d[0])<<","<<to_string(2)<<endl;
    
        for(int k=0;k<input_shape_d[0];k++)
        {
            outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
        }
        outputfile.close();
    }
    {%- endif %}
}
	
