
    //
    //  test01
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
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [2, 2], 'out': [3, 7, 2], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {2,2};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 10.0, 'out': [2], 'sorted_id': 1}
        {
            Tensor c = (fprec)10.0;
            forward_result[1] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/5', 'op': 'aten::mul', 'in': [0, 1], 'output_id': 0, 'shape': [2, 2], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[2] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
        }
        
        // {'name': 'Net/6', 'op': 'aten::mul', 'in': [2, 0], 'output_id': 0, 'shape': [2, 2], 'out': [5], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/7', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [2, 2], 'constant_value': [1.0, 2.0, 3.0, 4.0], 'out': [5], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {2,2};
            Constant1.reshape( shape );
            forward_result[4] = new VariableTensor( Constant1, 1 );
        }
        
        // {'name': 'Net/f1', 'op': 'aten::mul', 'in': [3, 4], 'output_id': 0, 'shape': [2, 2], 'out': [9], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[5] = op;
            
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
        }
        
        // {'name': 'Net/9', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 5.0, 'out': [7], 'sorted_id': 6}
        {
            Tensor c = (fprec)5.0;
            forward_result[6] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/f2', 'op': 'aten::mul', 'in': [0, 6], 'output_id': 0, 'shape': [2, 2], 'out': [9], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/11', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [9], 'sorted_id': 8}
        {
            Tensor c = (fprec)1.0;
            forward_result[8] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/12', 'op': 'aten::add', 'in': [5, 7, 8], 'output_id': 0, 'shape': [2, 2], 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {2,2};
            AddOp* op = new AddOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [9], 'output_id': 0, 'shape': [2, 2], 'out': [], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {2,2};
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
        vector<MCTNode*> forward_result(11);
    
        // input data
        Tensor::shape_type shape = {2,2};
        xin.reshape( shape );
        VariableTensor input_var( xin, 3 );
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 9 );
    #else
        do_train1( forward_result, input_var, 9 );
    #endif
        
        return 0;
    }
    