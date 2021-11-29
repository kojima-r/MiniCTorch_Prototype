
    //
    //  regr1_train
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
        
        fprec lr = 0.01;
        int epoch_num = 300;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
    
        ofstream outputfile("./test/regr1.out");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            train_mode = true;
            do_forward( forward_result, NL );
            
            auto o = forward_result[NL]->output;
            cout<<"epoch "<<epoch<<" - loss "<<o[0]<<endl;
            outputfile<<to_string(o[0])<<endl;
        
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        
        }
        outputfile.close();
        
        train_mode = false;
        inp_data.reshape( {1000,2} );
        {
            // 11 : aten::linear
            input_var.output = inp_data;
            auto inp_shape = inp_data.shape();
            int  nx = inp_shape[0];
            
            do_forward( forward_result, 11 );
            auto y_pred = forward_result[11]->get_output();
            
            ofstream outputfile( "./test/regr1.pred" );
            outputfile<<to_string(nx)<<",1"<<endl;
            for(int i=0;i<nx;i++)
            {
                outputfile<<to_string(y_pred(i,0))<<endl;
            }
            outputfile.close();
        }
    
    }
    