
    //
    //  clas_train
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
        int epoch_num = 200;
        cout<<"epoch_num : "<<epoch_num<<endl;
    
        inp_data.reshape( {112,4} );
        auto inp_shape = inp_data.shape();
    
        int batch_size = 16;
        int n_batch = (int)inp_shape[0] / batch_size;
        cout<<"indata shape   : "<<inp_shape[0]<<","<<inp_shape[1]<<endl;
        cout<<"batch  number  : "<<n_batch<<","<<batch_size<<endl;
        cout<<"learning ratio : "<<lr<<endl;
    
        
        Tensor x_tmp = xt::zeros<fprec>( { batch_size, (int)inp_shape[1] } );
        Tensor y_tmp = xt::zeros<fprec>( { batch_size } );
    
        ofstream outputfile("./classification/clas.out");
        
        do_zerograd( forward_result, NL );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            train_mode = true;
            
            xt::xarray<int> index = xt::arange( (int)inp_shape[0] );
            xt::random::shuffle( index );
            
            fprec total_loss = 0.0;
            int   total_corrects = 0;
            for(int j=0;j<n_batch;j++)
            {
                int jb = j * batch_size;
                for(int k=0;k<batch_size;k++)
                {
                    xt::row( x_tmp, k ) = xt::row( inp_data, index(jb+k) );
                    y_tmp( k ) = target_data( index(jb+k) );
                }
                
                input_var.output = x_tmp;
                forward_result[8]->output = y_tmp;
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                total_loss += o[0];
                
                auto y    = forward_result[7]->output;
                auto lbs  = xt::argmax( y, 1 );
                auto eq   = xt::equal( y_tmp, lbs );
                auto eq_t = xt::sum( eq );     
                total_corrects += (int)eq_t[0];
            
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            fprec total_acc = (fprec)total_corrects / (fprec)inp_shape[0];
            cout<<"total_loss : epoch "<<epoch<<" : loss "<<total_loss<<" : Acc "<<total_acc<<" "<<total_corrects<<endl;
            
            train_mode = false;
            
            input_var.output = inp_data;
            forward_result[8]->output = target_data;
            do_forward( forward_result, NL );
            
            auto  o = forward_result[NL]->output;
            auto  y = forward_result[7]->output;
            auto  lbs  = xt::argmax( y, 1 );
            auto  eq   = xt::equal( target_data, lbs );
            auto  eq_t = xt::sum( eq );     
            fprec acc = (fprec)eq_t[0] / (fprec)inp_shape[0];
            cout<<"total_loss : epoch "<<epoch<<" : loss "<<o[0]<<" : Acc "<<acc<<" "<<eq_t[0]<<endl;
            outputfile<<to_string(o[0])<<","<<to_string(acc)<<endl;
        }
        outputfile.close();
        
        train_mode = false;
    
    }
    