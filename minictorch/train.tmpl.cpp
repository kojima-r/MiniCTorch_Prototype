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
extern Tensor input_data;
{%- endif %}
{%- if target_enabled %}
extern Tensor target_data;
{%- endif %}

void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int NL )
{
    auto do_forward=[]( vector<MCTNode*> &op, int n ) 
    {
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->forward();
        }
    };
    auto do_backward=[]( vector<MCTNode*> &op, int n ) 
    {
        op[n]->grad = xt::ones_like( op[n]->output );
        for(int k=n;k>=0;k--) {
          if( op[k] )  op[k]->backward();
        }
    };
    auto do_zerograd=[]( vector<MCTNode*> &op, int n ) 
    {
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->zerograd();
        }
    };
    auto update_params=[]( vector<MCTNode*> &op, int n, fprec lr=0.01 ) 
    {
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->update( lr );
        }
    };
    {%- if eval_enabled %}
    auto eval_labels=[]( Tensor& y, Tensor &t )
    {
        auto lb = xt::argmax( y, 1 );
        auto eq = xt::equal( t, lb );
        auto sm = xt::sum( eq );
        return (int)sm[0];
    };
    {%- endif %}

    xt::random::seed(1);  
    
    fprec lr = {{lr}};
    int epoch_num = {{epochs}};
    cout<<"epoch_num : "<<epoch_num<<endl;
    
    {%- if input_enabled %}
    input_data.reshape( {{input_shape}} );
    auto input_shape = input_data.shape();
    {%- endif %}
    
    {%- if target_enabled %}
    target_data.reshape( {{target_shape}} );
    auto target_shape = target_data.shape();
    {%- endif %}

    int batch_size = {{batch_size}};
    int n_batch = (int)input_shape[0] / batch_size;
    cout<<"batch size  : "<<batch_size<<endl;
    cout<<"#batch  : "<<n_batch<<endl;
    cout<<"learning rate : "<<lr<<endl;

    input_var.output = input_data;
    
    Tensor x_tmp = xt::zeros<fprec>({ {{ x_shape }} });
    {%- if target_enabled %}
    Tensor y_tmp = xt::zeros<fprec>({ {{ y_shape }} });
    {%- endif %}

    ofstream outputfile("{{output_filename}}");
    std::chrono::system_clock::time_point  start, end; 
    start = std::chrono::system_clock::now();

    do_zerograd( forward_result, NL );
    for(int epoch=0;epoch<epoch_num;epoch++){
        train_mode = true;
        {%- if batch_enabled and shuffle %}
        xt::xarray<int> index = xt::arange( (int)input_shape[0] );
        xt::random::shuffle( index );
        {%- endif %}

        fprec total_loss = 0.0;
        {%- if classification_task %}
        int   total_corrects = 0;
        {%- endif %}
        for(int j=0;j<n_batch;j++){
            int jb = j * batch_size;
            for(int k=0;k<batch_size;k++){
                {%- if batch_enabled %}
                {%- if shuffle %}
                {
                    {%- if x_shape_rest_enabled %}
                    auto xw = xt::view( input_data, index(jb+k){{x_shape_rest}} );
                    xt::view( x_tmp, k{{x_shape_rest}} ) = xw;
                    {%- else %}
                    x_tmp( k ) = input_data( index(jb+k) );
                    {%- endif %}

                    {%- if y_shape_rest_enabled %}
                    auto yw = xt::view( target_data, index(jb+k){{y_shape_rest}} );
                    xt::view( y_tmp, k{y_shape_rest} ) = yw;
                    {%- elif target_enabled %}
                    y_tmp( k ) = target_data( index(jb+k) );
                    {%- endif %}
                }
                {%- else %}
                {
                    {%- if x_shape_rest_enabled %}
                    auto xw = xt::view( input_data, jb+k{{x_shape_rest}} );
                    xt::view( x_tmp, k{{x_shape_rest}} ) = xw;
                    {%- else %}
                    x_tmp( k ) = input_data( jb+k );
                    {%- endif %}

                    {%- if y_shape_rest_enabled %}
                    auto yw = xt::view( target_data, jb+k{{y_shape_rest}} );
                    xt::view( y_tmp, k{{y_shape_rest}} ) = yw;
                    {%- elif target_enabled %}
                    y_tmp( k ) = target_data( jb+k );
                    {%- endif %}
                }
                {%- endif %}
                {%- endif %}
            }
            input_var.output = x_tmp;
            {%- if target_no > 0: %}
            forward_result[{{target_no}}]->output = y_tmp;
            {%- endif %}
            do_forward( forward_result, NL );
            
            auto o = forward_result[NL]->output;
            total_loss += o[0];
            
            {%- if classification_task %}
            int corrects = eval_labels( forward_result[{{pred_no}}]->output, y_tmp );
            total_corrects += corrects;
            {%- endif %}
        
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        }
        {%- if classification_task %}
        fprec total_acc = (fprec)total_corrects / (fprec)input_shape[0];
        cout<<"total_loss (batch): epoch "<<epoch<<" : loss "<<total_loss<<" : Acc "<<total_acc<<" ("<<total_corrects<<"/"<<input_shape[0]<<")"<<endl;
        {%- else %}
        cout<<"total_loss : epoch "<<epoch<<" - loss "<<total_loss<<endl;
        {%- endif %}

        train_mode = false;
        input_var.output = input_data;

        {%- if target_no > 0 %}
        forward_result[{{target_no}}]->output = target_data;
        {%- endif%}
        
        do_forward( forward_result, NL );
        auto o = forward_result[NL]->output;
        
        {%- if double_loss_enabled %}
        auto o1 = forward_result[{{loss1_no}}]->output;
        auto o2 = forward_result[{{loss2_no}}]->output; 
        cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
        outputfile<<to_string(o[0])<<endl;
        {%- endif %}

        {%- if classification_task %}
        int corrects = eval_labels( forward_result[{{pred_no}}]->output, target_data );
        fprec acc = (fprec)corrects / (fprec)input_shape[0];
        cout<<"total_loss (all)  : epoch "<<epoch<<" : loss "<<o[0]<<" : Acc "<<acc<<" "<<corrects<<endl;
        outputfile<<to_string(o[0])<<","<<to_string(acc)<<","<<total_loss<<endl;
        {%- else %}
        cout<<"epoch "<<epoch<<" - loss "<<o[0]<<endl;
        outputfile<<to_string(o[0])<<endl;
        {%- endif %}
    }
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout<<"MiniCTorch Time: "<<elapsed<<endl;

    outputfile.close();
    
    train_mode = false;
    {
        // {{pred_no}} : {pred_op}
        {%- if input_enabled %}
        input_var.output = input_data;
        {%- endif %}

        do_forward( forward_result, {{pred_no}} );
        auto y_pred = forward_result[ {{pred_no}} ]->output;
        
        ofstream outputfile( "{{pred_filename}}" );
        {%- if pred_type == 1 %}
        outputfile<<to_string(input_shape[0])<<",1"<<endl;
        for(int i=0;i<input_shape[0];i++)
        {
            outputfile<<to_string(y_pred(i,0))<<endl;
        }
        {%- else %}
        //int nx = input_shape[0];
        int nx = {{pred_output}};
        outputfile<<to_string(nx)<<","<<to_string(input_shape[1])<<endl;
        
        for(int i=0;i<nx;i++)
        {
            for(int j=0;j<input_shape[1]-1;j++)
            {
                outputfile<<to_string(y_pred(i,j))<<",";
            }
            outputfile<<to_string(y_pred(i,input_shape[1]-1))<<endl;
        }
        {%- endif %}
        outputfile.close();

    }
    {%- if z_no >= 0 %}
    {
        // {{z_no}} : z output
        auto z_pred = forward_result[{{z_no}}]->get_output();
    
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
	
