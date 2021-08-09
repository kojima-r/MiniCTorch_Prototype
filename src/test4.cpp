
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
    extern Tensor  Constant1;
    
    int main()
    {
        vector<MCTNode*> forward_result(20);
    
        // input data
        Tensor::shape_type shape = {100,2};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [100, 2], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {100,2};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/105', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {64,2};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/104', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {64};
            forward_result[2] = new VariableTensor(fc1_bias);
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'shape': [100, 64], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {100,64};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [100, 64], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {100,64};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2]/weight/108', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {32,64};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_weight);
        }
        
        // {'name': 'Net/Linear[fc2]/bias/107', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {32};
            forward_result[6] = new VariableTensor(fc2_bias);
        }
        
        // {'name': 'Net/Linear[fc2]/input.5', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [100, 32], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {100,32};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/input.7', 'op': 'aten::relu', 'in': [7], 'shape': [100, 32], 'out': [11], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {100,32};
            ReluOp* op = new ReluOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/111', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {1,32};
            fc3_weight.reshape( shape );
            forward_result[9] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/110', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {1};
            forward_result[10] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/input.9', 'op': 'aten::linear', 'in': [8, 9, 10], 'shape': [100, 1], 'out': [13], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {100,1};
            LinearOp* op = new LinearOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/target.1', 'op': 'prim::Constant', 'in': [], 'shape': [100, 1], 'constant_value': [12.6442, 14.2724, 2.3541, 3.9799, 2.7917, 6.2946, 5.4144, 3.6112, 14.1766, 4.7809, 4.6936, 12.7688, 11.7691, 3.2024, 3.2638, 9.1611, 6.271, 4.5049, 5.3078, 3.2171, 13.3978, 5.5882, 4.32, 12.3751, 3.1452, 9.6706, 2.7055, 2.4756, 5.1311, 2.4733, 5.1901, 12.4477, 2.2895, 1.4738, 12.5678, 8.8092, 5.9747, 9.4119, 5.2347, 12.5353, 2.8599, 13.761, 11.8049, 14.1149, 4.6603, 4.8467, 7.3956, 3.8746, 5.8784, 2.454, 3.005, 3.8026, 5.1516, 5.8612, 4.4625, 3.2352, 3.885, 13.8736, 12.0878, 10.4887, 2.4796, 8.4232, 10.8639, 3.6631, 1.6655, 3.8742, 13.4705, 13.8999, 11.0951, 12.7526, 5.4187, 6.7698, 7.1659, 1.756, 4.0808, 8.9944, 12.381, 12.8833, 4.6455, 5.4636, 4.3245, 5.0018, 14.0947, 4.3089, 12.3858, 2.9765, 13.8632, 14.326, 2.1931, 5.8966, 12.2755, 2.9158, 2.3136, 4.9452, 12.3885, 4.2832, 14.1509, 2.685, 5.2271, 1.9709], 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {100,1};
            Constant1.reshape( shape );
            forward_result[12] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/39', 'op': 'prim::ListConstruct', 'in': [11, 12], 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {};
            ListConstructOp* op = new ListConstructOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/40', 'op': 'aten::broadcast_tensors', 'in': [13], 'shape': [], 'out': [16, 15], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "broadcast_tensors" );
            forward_result[14] = op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/input', 'op': 'prim::ListUnpack', 'in': [14], 'shape': [100, 1], 'out': [18], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {100,1};
            ListUnpackOp* op = new ListUnpackOp();
            forward_result[15] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/target', 'op': 'prim::ListUnpack', 'in': [14], 'shape': [100, 1], 'out': [18], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {100,1};
            ListUnpackOp* op = new ListUnpackOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/43', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [18], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[17] = new VariableTensor( c );
        }
        
        // {'name': 'Net/tensor', 'op': 'aten::mse_loss', 'in': [15, 16, 17], 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {};
            MseLossOp* op = new MseLossOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [18], 'shape': [], 'out': [], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[18]->forward();
        for(int k=0;k<=18;k++) {
           if( forward_result[k] )  forward_result[k]->forward();
        }
        auto o = forward_result[18]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[18]->grad = xt::ones_like( forward_result[18]->output );
        //forward_result[18]->backward();
        for(int k=18;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    