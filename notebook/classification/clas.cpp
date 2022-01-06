
    //
    //  clas
    //
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include "minictorch.hpp"
    
    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  fc1_weight;
    extern Tensor  fc1_bias;
    extern Tensor  fc2_weight;
    extern Tensor  fc2_bias;
    extern Tensor  Constant1;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [16, 4], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {16,4};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Model/Net[net]/Linear[fc1]/weight/weight.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {64,4};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'Model/Net[net]/Linear[fc1]/bias/bias.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {64};
            fc1_bias.reshape( shape );
            forward_result[2] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'Model/Net[net]/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'output_id': 0, 'shape': [16, 64], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {16,64};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Model/Net[net]/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [16, 64], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {16,64};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Model/Net[net]/Linear[fc2]/weight/weight', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {3,64};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_weight );
        }
        
        // {'name': 'Model/Net[net]/Linear[fc2]/bias/bias', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {3};
            fc2_bias.reshape( shape );
            forward_result[6] = new VariableTensor( fc2_bias );
        }
        
        // {'name': 'Model/Net[net]/Linear[fc2]/input', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [16, 3], 'out': [13], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {16,3};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Loss[loss]/60', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [16], 'constant_value': [0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0], 'out': [13], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {16};
            Constant1.reshape( shape );
            forward_result[8] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Model/Loss[loss]/59', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 9}
        {
            forward_result[9] = NULL;
        }
        
        // {'name': 'Model/Loss[loss]/58', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [13], 'sorted_id': 10}
        {
            Tensor c = (fprec)1.0;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'Model/Loss[loss]/57', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [13], 'sorted_id': 11}
        {
            Tensor c = (fprec)-100.0;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Model/Loss[loss]/56', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [13], 'sorted_id': 12}
        {
            Tensor c = (fprec)0.0;
            forward_result[12] = new VariableTensor( c, false );
        }
        
        // {'name': 'Model/Loss[loss]/61', 'op': 'aten::cross_entropy_loss', 'in': [7, 8, 9, 10, 11, 12], 'output_id': 0, 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            CrossEntropyLossOp* op = new CrossEntropyLossOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [13], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 14}
        {
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
        vector<MCTNode*> forward_result(15);
    
        // input data
        Tensor::shape_type shape = {16,4};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 13 );
    #else
        do_train1( forward_result, input_var, 13 );
    #endif
        
        return 0;
    }
    