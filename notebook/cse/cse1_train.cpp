
    //
    //  cse1_train
    //
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    extern bool train_mode;
    
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
        int epoch_num = 200;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
            
        auto labels = forward_result[8]->output;
        auto lb_shape = labels.shape(); 
        
        ofstream outputfile("./cse/cse1.out");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            train_mode = true;
            do_forward( forward_result, NL );
            
            fprec o = forward_result[NL]->output[0];
          
            auto y = forward_result[7]->output;
            auto preds = xt::argmax( y, 1 );
            auto eq    = xt::equal( labels, preds );
            auto eq_t  = xt::sum( eq );     
            fprec acc = (fprec)eq_t[0] / (fprec)lb_shape[0];
            
            cout<<"epoch "<<epoch<<" - loss "<<o<<" - accuracy "<<acc<<endl;
            outputfile<<to_string(o)<<","<<to_string(acc)<<endl;
            
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        }
        outputfile.close();
        
        train_mode = false;
    
    }
    