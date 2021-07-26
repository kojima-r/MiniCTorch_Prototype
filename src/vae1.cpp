
    #include <stdio.h>
    #include <iostream>
    #include <fstream>
    #include <string>
    #include <vector>
    #include "minictorch.hpp"

    using namespace std;
    
    extern void LoadParameter();
    
    extern Tensor  xin;
    extern Tensor  fc1_weight;
    extern Tensor  fc1_bias;
    extern Tensor  fc2_mean_weight;
    extern Tensor  fc2_mean_bias;
    extern Tensor  Constant1;
    extern Tensor  fc2_var_weight;
    extern Tensor  fc2_var_bias;
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    extern Tensor  fc4_weight;
    extern Tensor  fc4_bias;

    int main()
    {
        // load parameters
        LoadParameter();
        
        // input data
        VariableTensor input_var(xin);
        vector<MCTNode*> forward_result(27);
    
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [32, 64], 'out': [3], 'sorted_id': 0}
        {
        
            Tensor::shape_type shape = {32,64};
            forward_result[0]=&input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/75', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
        
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/74', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
        
            forward_result[2] = new VariableTensor(fc1_bias);
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'shape': [32, 16], 'out': [4], 'sorted_id': 3}
        {
        
            Tensor::shape_type shape = {32,16};
            LinearOp*  op = new LinearOp();
            forward_result[3]=op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [32, 16], 'out': [7, 11], 'sorted_id': 4}
        {
        
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4]=op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/78', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
        
            forward_result[5] = new VariableTensor(fc2_mean_weight);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/77', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
        
            forward_result[6] = new VariableTensor(fc2_mean_bias);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/79', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [32, 2], 'out': [17], 'sorted_id': 7}
        {
        
            Tensor::shape_type shape = {32,2};
            LinearOp*  op = new LinearOp();
            forward_result[7]=op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::Constant', 'in': [], 'shape': [32, 2], 'constant_value': [-1.5256, -0.7502, -0.654, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091, -0.7121, 0.3037, -0.7773, -0.2515, -0.2223, 1.6871, 0.2284, 0.4676, -0.697, -1.1608, 0.6995, 0.1991, 0.8657, 0.2444, -0.6629, 0.8073, 1.1017, -0.1759, -2.2456, -1.4465, 0.0612, -0.6177, -0.7981, -0.1316, 1.8793, -0.0721, 0.1578, -0.7735, 0.1991, 0.0457, 0.153, -0.4757, -0.111, 0.2927, -0.1578, -0.0288, 2.3571, -1.0373, 1.5748, -0.6298, -0.9274, 0.5451, 0.0663, -0.437, 0.7626, 0.4415, 1.1651, 2.0154, 0.1374, 0.9386, -0.186, -0.6446, 1.5392, -0.8696, -3.3312, -0.7479], 'out': [15], 'sorted_id': 8}
        {
        
            Tensor::shape_type shape = {32,2};
            forward_result[8] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/81', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
        {
        
            forward_result[9] = new VariableTensor(fc2_var_weight);
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/80', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
        {
        
            forward_result[10] = new VariableTensor(fc2_var_bias);
        }
        
        // {'name': 'Net/Linear[fc2_var]/82', 'op': 'aten::linear', 'in': [4, 9, 10], 'shape': [32, 2], 'out': [13], 'sorted_id': 11}
        {
        
            Tensor::shape_type shape = {32,2};
            LinearOp*  op = new LinearOp();
            forward_result[11]=op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [13], 'sorted_id': 12}
        {
        
            Tensor::shape_type shape = {};
            Tensor c=(float)0.5;
            forward_result[12]=new VariableTensor(c);
        }
        
        // {'name': 'Net/34', 'op': 'aten::mul', 'in': [11, 12], 'shape': [32, 2], 'out': [14], 'sorted_id': 13}
        {
        
            Tensor::shape_type shape = {32,2};
            MulOp* op=new MulOp();
            forward_result[13]=op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/35', 'op': 'aten::exp', 'in': [13], 'shape': [32, 2], 'out': [15], 'sorted_id': 14}
        {
        
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[14]=op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/37', 'op': 'aten::mul', 'in': [8, 14], 'shape': [32, 2], 'out': [17], 'sorted_id': 15}
        {
        
            Tensor::shape_type shape = {32,2};
            MulOp* op=new MulOp();
            forward_result[15]=op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/38', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [17], 'sorted_id': 16}
        {
        
            Tensor::shape_type shape = {};
            Tensor c=(float)1.0;
            forward_result[16]=new VariableTensor(c);
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [7, 15, 16], 'shape': [32, 2], 'out': [20], 'sorted_id': 17}
        {
        
            Tensor::shape_type shape = {32,2};
            AddOp* op=new AddOp();
            forward_result[17]=op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/84', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [20], 'sorted_id': 18}
        {
        
            forward_result[18] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/83', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [20], 'sorted_id': 19}
        {
        
            forward_result[19] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [17, 18, 19], 'shape': [32, 16], 'out': [21], 'sorted_id': 20}
        {
        
            Tensor::shape_type shape = {32,16};
            LinearOp*  op = new LinearOp();
            forward_result[20]=op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [20], 'shape': [32, 16], 'out': [24], 'sorted_id': 21}
        {
        
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[21]=op;
            
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/87', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 22}
        {
        
            forward_result[22] = new VariableTensor(fc4_weight);
        }
        
        // {'name': 'Net/Linear[fc4]/bias/86', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 23}
        {
        
            forward_result[23] = new VariableTensor(fc4_bias);
        }
        
        // {'name': 'Net/Linear[fc4]/88', 'op': 'aten::linear', 'in': [21, 22, 23], 'shape': [32, 64], 'out': [25], 'sorted_id': 24}
        {
        
            Tensor::shape_type shape = {32,64};
            LinearOp*  op = new LinearOp();
            forward_result[24]=op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/43', 'op': 'aten::sigmoid', 'in': [24], 'shape': [32, 64], 'out': [26], 'sorted_id': 25}
        {
        
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[25]=op;
            
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [25], 'shape': [32, 64], 'out': [], 'sorted_id': 26}
        {
        
            Tensor::shape_type shape = {32,64};
        }
        
        cout<<"### forward computation ..."<<endl;
        forward_result[25]->forward();
        auto o = forward_result[25]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[25]->grad=xt::ones_like(forward_result[25]->output); // 210702 mod mari
        forward_result[25]->backward();
        cout<<input_var.grad<<endl;
    
        return 0;
    }
    