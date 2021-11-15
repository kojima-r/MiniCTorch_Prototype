
    //
    //  mse2_train
    //
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    extern bool train_mode;
    
    extern Tensor inp_data;
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
    
        xt::random::seed(1);
        
        fprec lr = 0.01;
        int epoch_num = 2000;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
        inp_data.reshape( {63,1} );
        auto inp_shape = inp_data.shape();
    
        int batch_size = 9;
        int n_batch = (int)inp_shape[0] / batch_size;
        cout<<"indata shape   : "<<inp_shape[0]<<","<<inp_shape[1]<<endl;
        cout<<"batch  number  : "<<n_batch<<","<<batch_size<<endl;
        cout<<"learning ratio : "<<lr<<endl;
    
        
        target_data.reshape( {63,1} );
        Tensor x_t = xt::zeros<fprec>( { batch_size, (int)inp_shape[1] } );
        Tensor y_t = xt::zeros<fprec>( { batch_size, (int)inp_shape[1] } );
        
        ofstream outputfile("./mse/mse2.out");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            train_mode = true;
            
            xt::xarray<int> index = xt::arange( (int)inp_shape[0] );
            xt::random::shuffle( index );
        
           
            fprec total_loss = 0.0;
            for(int j=0;j<n_batch;j++)
            {
                int jb = j * batch_size;
                for(int k=0;k<batch_size;k++)
                {
                    xt::row( x_t, k ) = xt::flatten( xt::row( inp_data, index(jb+k) ) );
                    xt::row( y_t, k ) = xt::flatten( xt::row( target_data, index(jb+k) ) );
                }
                
                input_var.output = x_t;
                forward_result[12]->output = y_t;  
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
            
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss : epoch "<<epoch<<" : loss "<<total_loss<<endl;
            outputfile<<to_string(total_loss)<<endl;
            
        }
        outputfile.close();
        
        train_mode = false;
    
        {
            input_var.output = inp_data;
            do_forward( forward_result, 11 );
            auto y_pred = forward_result[11]->output;
            
            int nx = inp_data.size();
           
            ofstream outputfile("./mse/mse2.pred");
            for(int i=0;i<nx;i++)
            {
                outputfile<<to_string(inp_data(i,0))<<","<<to_string(y_pred(i,0))<<endl;
            }
            outputfile.close();
        }
        
    }
    