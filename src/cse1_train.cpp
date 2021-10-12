
    //
    //  cse1_train
    //
    #include"minictorch.hpp"
        
    void cse_train_loop( vector<MCTNode*>& forward_result, int NL, fprec lr )
    {
        auto do_forward=[]( vector<MCTNode*> &op, int n ) 
        {
            //cout<<"### forward computation ..."<<endl;
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->forward();
            }
        };
        auto do_backward=[]( vector<MCTNode*> &op, int n ) 
        {
            //cout<<"### backward computation ..."<<endl;
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
        
        //fprec lr= 0.01;
        //int   NL = 12; // loss
        int   NS =  7;   // result
        
        auto labels = forward_result[8]->get_output();
        auto sh = labels.shape();
            
        ofstream outputfile("cse1.out");
        for(int epoch=0;epoch<300;epoch++)
        {
            do_forward( forward_result, NL );
            
            CrossEntropyLossOp *op = dynamic_cast<CrossEntropyLossOp*>(forward_result[NL]);
            
            auto  cls = op->get_classes();
            auto  eq  = xt::equal( labels, cls );
            auto  eq_t = xt::sum( eq );
            fprec acc = (fprec)eq_t[0] / (fprec)sh[0];
            
            fprec o = op->get_loss();
            cout<<"epoch "<<epoch<<" - loss "<<o<<" - accuracy "<<acc<<endl;
            outputfile<<to_string(o)<<","<<to_string(acc)<<endl;
            
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        }
        outputfile.close();
    
        //return 0;
    }
    
    /*  main program for cross entropy loss training loop
    
    extern void cse_train_loop( vector<MCTNode*>& forward_result, int NL, fprec lr );
    
    int main()
    {
        vector<MCTNode*> forward_result(14);
    
        // input data
        Tensor::shape_type shape = {112,4};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
        cse_train_loop( forward_result, 12, 0.01 );
        //do_train1( forward_result, input_var, 12 );
        
        return 0;
    }
    */