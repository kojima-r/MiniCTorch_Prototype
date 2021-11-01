    //
    //  vae1_train
    //
    #include "../../src/minictorch.hpp"
    
    // all input data
    extern Tensor indata;

    
    void do_train_loop( vector<MCTNode*> &forward_result, VariableTensor &input_var, int NL )
    {
        auto do_forward=[]( vector<MCTNode*> &op, int n, int out ) 
        {
            //cout<<"### forward computation ..."<<endl;
            for(int k=0;k<=n;k++) {
              if( out> 0 )  cout<<"id - "<<k<<endl;
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
        
        
        string project = "vae1";
        string folder  = "./src/";
        
        fprec lr = 0.01;
        //int   NL = 79;
        
        indata.reshape({1797,64});
        auto indata_s = indata.shape();
        
        int batch_size = 32;
        int n_batch = (int)indata_s[0] / batch_size;
        cout<<"batch1 "<<indata_s[0]<<","<<indata_s[1]<<endl;
        cout<<"batch2 "<<n_batch<<","<<batch_size<<endl;
        
        xt::random::seed(1);
        
        int epoch_num = 200;
        cout<<"epoch_num : "<<epoch_num<<endl;
        
        Tensor x_pred = xt::zeros<fprec>( { batch_size, (int)indata_s[1] } );
        
        string flname = folder + project + ".out";
        ofstream outputfile( flname );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            Tensor index = xt::arange( (int)indata_s[0] );
            xt::random::shuffle( index );
            
            forward_result[59]->set_output1( (fprec)batch_size );  // div size
            forward_result[75]->set_output1( (fprec)batch_size );  // div size
            
            fprec total_loss = 0.0;
            for(int jj=0;jj<n_batch;jj++)
            {
                int j1 = jj*batch_size;
                for(int k=0;k<batch_size;k++)
                {
                    xt::row(x_pred,k) = xt::flatten( xt::row(indata,index(j1+k)) );
                }
                input_var.output = x_pred;
            
                do_forward( forward_result, NL, 0 );
                
                auto o = forward_result[NL]->output;
                //cout<<"epoch "<<epoch<<","<<jj<<" - loss "<<o<<endl;
                total_loss += o[0];
        
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss "<<epoch<<" loss - "<<total_loss<<endl;
            
            
            input_var.output = indata;
            
            forward_result[59]->set_output1( (fprec)indata_s[0] );  // div size
            forward_result[75]->set_output1( (fprec)indata_s[0] );  // div size
            
            do_forward( forward_result, NL, 0 );
            auto o = forward_result[NL]->output;
            cout<<"epoch "<<epoch<<" - loss "<<o<<endl;
            outputfile<<to_string(o[0])<<endl;
            
        }
        outputfile.close();
        
        {
            // 38: sigmoid outpout
            auto y_pred = forward_result[38]->get_output();
           
            flname = folder + project + ".pred";
            ofstream outputfile( flname );
            outputfile<<to_string(n_img)<<","<<to_string(indata_s[1])<<endl;
            
            int n_img = 10;
            for(int k=0;k<n_img;k++)
            {
                for(int i=0;i<indata_s[1]-1;i++)
                {
                    outputfile<<to_string(y_pred(k,i))<<",";
                }
                outputfile<<to_string(y_pred(k,indata_s[1]-1))<<endl;
            }
            outputfile.close();
        }
        {
            // 30: z output
            auto z_pred = forward_result[30]->get_output();
            
            flname = folder + project + ".z";
            ofstream outputfile( flname );
            outputfile<<to_string(indata_s[0])<<","<<to_string(2)<<endl;
            
            for(int k=0;k<indata_s[0];k++)
            {
                outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
            }
            outputfile.close();
        }
    }