
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  fc1_weight;
    extern Tensor  fc1_bias;
    extern Tensor  fc2_weight;
    extern Tensor  fc2_bias;
    extern Tensor  Constant1;
    
    int main()
    {
        vector<MCTNode*> forward_result(14);
    
        // input data
        Tensor::shape_type shape = {112,4};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [112, 4], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {112,4};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/35', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {64,4};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/34', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {64};
            forward_result[2] = new VariableTensor(fc1_bias);
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'shape': [112, 64], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {112,64};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [112, 64], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {112,64};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2]/weight/38', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {3,64};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_weight);
        }
        
        // {'name': 'Net/Linear[fc2]/bias/37', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {3};
            forward_result[6] = new VariableTensor(fc2_bias);
        }
        
        // {'name': 'Net/Linear[fc2]/input', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [112, 3], 'out': [12], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {112,3};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/17', 'op': 'prim::Constant', 'in': [], 'shape': [112], 'constant_value': [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0], 'out': [12], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {112};
            Constant1.reshape( shape );
            forward_result[8] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/18', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [12], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {};
            forward_result[9] = NULL;
        }
        
        // {'name': 'Net/19', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [12], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/20', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': -100.0, 'out': [12], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)-100.0;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/21', 'op': 'aten::cross_entropy_loss', 'in': [7, 8, 9, 10, 11], 'shape': [], 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {};
            CrossEntropyLossOp* op = new CrossEntropyLossOp();
            forward_result[12] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [12], 'shape': [], 'out': [], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[12]->forward();
        for(int k=0;k<=12;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[12]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[12]->grad = xt::ones_like( forward_result[12]->output );
        //forward_result[12]->backward();
        for(int k=12;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    