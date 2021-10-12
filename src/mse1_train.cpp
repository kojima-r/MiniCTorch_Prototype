
    //
    //  mse1_train
    //
    #include"minictorch.hpp"
    
    
    void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int NL, fprec lr )
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
    
        //fprec lr = 0.01;
        //int   NL = 18;
            
        ofstream outputfile("mse1.out");
        for(int epoch=0;epoch<500;epoch++)
        {
            do_forward( forward_result, NL );
            
            MseLossOp *op = dynamic_cast<MseLossOp*>(forward_result[NL]);
            
            auto o = op->get_loss();
            cout<<"epoch "<<epoch<<" - loss "<<o<<endl;
            outputfile<<to_string(o)<<endl;
        
            do_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            do_zerograd( forward_result, NL );
        }
        outputfile.close();
    
        // evaluate funnction from learning data
        {
           int  NS = 11;
           int  nx = 1000;
           auto xx = xt::linspace<fprec>(0.,10.,nx);
           auto x1 = xt::sin(xx);
           auto x2 = xt::exp(xx/5.0);
           auto x_pred = xt::stack(xtuple(x1,x2), 1);
           
           input_var.output = x_pred;
           do_forward( forward_result, NS );
           auto y_pred = forward_result[NS]->get_output();
           //cout<<"y_pred"<<y_pred<<endl;
           
           ofstream outputfile("mse1.pred");
           for(int i=0;i<nx;i++)
           {
                outputfile<<to_string(xx[i])<<","<<to_string(y_pred(i,0))<<endl;
           }
           outputfile.close();
           
        }
    }
    
/*  main program for learning loop
#ifdef _TRAIN_LOOP
    extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int NL, fprec lr );
#endif

    int main()
    {
        vector<MCTNode*> forward_result(20);
    
        // input data
        Tensor::shape_type shape = {100,2};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
      
#ifdef _TRAIN_LOOP
        do_train_loop( forward_result, input_var, 18, 0.01 );
#else
        do_train1( forward_result, input_var, 18 );
    #endif
        
        return 0;
    } 
*/