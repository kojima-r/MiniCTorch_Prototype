
    //
    //  test04-1
    //
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include "minictorch.hpp"
    
    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  Constant1;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [3, 3], 'constant_value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {3,3};
            Constant1.reshape( shape );
            forward_result[0] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'input/y', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [3], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {3};
            forward_result[1] = &input_var;
        }
        
        // {'name': 'Net/5', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [3], 'sorted_id': 2}
        {
            Tensor c = (fprec)1.0;
            forward_result[2] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/6', 'op': 'aten::add', 'in': [0, 1, 2], 'output_id': 0, 'shape': [3, 3], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {3,3};
            AddOp* op = new AddOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [3], 'output_id': 0, 'shape': [3, 3], 'out': [], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {3,3};
        }
        
    }
    
    void do_train1( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N )
    {
        cout<<"### forward computation ..."<<endl;
        for(int k=0;k<=N;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[N]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[N]->grad = xt::ones_like( forward_result[N]->output );
        for(int k=N;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    }
    
    
    #ifdef _TRAIN
    extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N );
    #endif
    
    int main()
    {
        vector<MCTNode*> forward_result(5);
    
        // input data
        Tensor::shape_type shape = {3};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 3 );
    #else
        do_train1( forward_result, input_var, 3 );
    #endif
        
        return 0;
    }
    