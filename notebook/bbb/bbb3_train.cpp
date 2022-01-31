
    //
    //  bbb3_train
    //
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    extern bool train_mode;
    
    extern Tensor input_data;
    extern Tensor target_data;
    
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
    
        //xt::random::seed(1);  
        
        fprec lr = 0.01;
        int epoch_num = 10;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
        input_data.reshape( {40,1,28,28} );
        auto input_shape = input_data.shape();
        
        target_data.reshape( {40} );
        auto target_shape = target_data.shape();
        
        int batch_size = 4;
        int n_batch = (int)input_shape[0] / batch_size;
        cout<<"batch  number  : "<<n_batch<<","<<batch_size<<endl;
        cout<<"learning ratio : "<<lr<<endl;
    
        
        Tensor x_tmp = xt::zeros<fprec>( { batch_size, 1, 28, 28 } );
        Tensor y_tmp = xt::zeros<fprec>( { batch_size } );
    
        ofstream outputfile("./bbb/bbb3.out");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            train_mode = true;
            
            fprec total_loss = 0.0;
            for(int j=0;j<n_batch;j++)
            {
                int jb = j * batch_size;
                for(int k=0;k<batch_size;k++)
                {
                    auto xw = xt::view( input_data, jb+k, xt::all(), xt::all() );
                    xt::view( x_tmp, k, xt::all(), xt::all() ) = xw;
                    y_tmp( k ) = target_data( jb+k );
                }
                
                input_var.output = x_tmp;
                forward_result[1158]->output = y_tmp;
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
                
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss : epoch "<<epoch<<" - loss "<<total_loss<<endl;
            
            train_mode = false;
            
            input_var.output = input_data;
            forward_result[1158]->output = target_data;
            do_forward( forward_result, NL );
            
            auto o  = forward_result[NL]->output;
            auto o1 = forward_result[1150]->output;
            auto o2 = forward_result[1160]->output; 
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
            outputfile<<to_string(o[0])<<endl;
            
        }
        outputfile.close();
        
        train_mode = false;
    
    }
    