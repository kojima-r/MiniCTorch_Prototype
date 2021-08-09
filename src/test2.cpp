
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
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    
    int main()
    {
        vector<MCTNode*> forward_result(13);
    
        // input data
        Tensor::shape_type shape = {1,2};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [1, 2], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {1,2};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/43', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {32,2};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/42', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {32};
            forward_result[2] = new VariableTensor(fc1_bias);
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'shape': [1, 32], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {1,32};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [1, 32], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {1,32};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2]/weight/46', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {32,32};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_weight);
        }
        
        // {'name': 'Net/Linear[fc2]/bias/45', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {32};
            forward_result[6] = new VariableTensor(fc2_bias);
        }
        
        // {'name': 'Net/Linear[fc2]/input.5', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [1, 32], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {1,32};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [7], 'shape': [1, 32], 'out': [11], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {1,32};
            ReluOp* op = new ReluOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/49', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {3,32};
            fc3_weight.reshape( shape );
            forward_result[9] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/48', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {3};
            forward_result[10] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/50', 'op': 'aten::linear', 'in': [8, 9, 10], 'shape': [1, 3], 'out': [12], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {1,3};
            LinearOp* op = new LinearOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [11], 'shape': [1, 3], 'out': [], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {1,3};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[11]->forward();
        for(int k=0;k<=11;k++) {
           if( forward_result[k] )  forward_result[k]->forward();
        }
        auto o = forward_result[11]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[11]->grad = xt::ones_like( forward_result[11]->output );
        //forward_result[11]->backward();
        for(int k=11;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    