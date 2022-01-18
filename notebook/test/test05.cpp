
    //
    //  test05
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
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [2, 3, 4], 'constant_value': [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0, 9.0], 'out': [2], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {2,3,4};
            Constant1.reshape( shape );
            forward_result[0] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'input/y', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [4, 3], 'out': [2], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {4,3};
            forward_result[1] = &input_var;
        }
        
        // {'name': 'Net/5', 'op': 'aten::matmul', 'in': [0, 1], 'output_id': 0, 'shape': [2, 3, 3], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {2,3,3};
            MatMulOp* op = new MatMulOp();
            forward_result[2] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [2], 'output_id': 0, 'shape': [2, 3, 3], 'out': [], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {2,3,3};
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
        vector<MCTNode*> forward_result(4);
    
        // input data
        Tensor::shape_type shape = {4,3};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 2 );
    #else
        do_train1( forward_result, input_var, 2 );
    #endif
        
        return 0;
    }
    