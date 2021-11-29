
    //
    //  vae1_train
    //
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    extern bool train_mode;
    
    extern Tensor inp_data;
    
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
        
        fprec lr = 0.001;
        int epoch_num = 200;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
        inp_data.reshape( {1797,64} );
        auto inp_shape = inp_data.shape();
    
        int batch_size = 32;
        int n_batch = (int)inp_shape[0] / batch_size;
        cout<<"indata shape   : "<<inp_shape[0]<<inp_shape[1]<<endl;
        cout<<"batch  number  : "<<n_batch<<","<<batch_size<<endl;
        cout<<"learning ratio : "<<lr<<endl;
    
        
        Tensor x_tmp = xt::zeros<fprec>( { batch_size, (int)inp_shape[1] } );
    
        ofstream outputfile("./test/vae1.out");
        
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
                    xt::row( x_tmp, k ) = xt::row( inp_data, index(jb+k) );
                }
                
                input_var.output = x_tmp;
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
                
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss : epoch "<<epoch<<" - loss "<<total_loss<<endl;
            
            train_mode = false;
            
            input_var.output = inp_data;
            do_forward( forward_result, NL );
            
            auto o  = forward_result[NL]->output;
            auto o1 = forward_result[59]->output;
            auto o2 = forward_result[74]->output; 
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<" ( "<<o1[0]<<" , "<<o2[0]<<" ) "<<endl;
            outputfile<<to_string(o[0])<<endl;
            
        }
        outputfile.close();
        
        train_mode = false;
            
        {
            // 38 : aten::sigmoid
            input_var.output = inp_data;
            do_forward( forward_result, 38 );
            auto y_pred = forward_result[38]->output;
            
            int nx = 10;
            
            ofstream outputfile( "./test/vae1.pred" );
            outputfile<<to_string(nx)<<","<<to_string(inp_shape[1])<<endl;
            
            for(int i=0;i<nx;i++)
            {
                for(int j=0;j<inp_shape[1]-1;j++)
                {
                    outputfile<<to_string(y_pred(i,j))<<",";
                }
                outputfile<<to_string(y_pred(i,inp_shape[1]-1))<<endl;
            }
            outputfile.close();
        }
        {
            // 30 : z output
            auto z_pred = forward_result[30]->get_output();
        
            ofstream outputfile( "./test/vae1.z" );
            outputfile<<to_string(inp_shape[0])<<","<<to_string(2)<<endl;
        
            for(int k=0;k<inp_shape[0];k++)
            {
                outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
            }
            outputfile.close();
        }
        
    
    }
    