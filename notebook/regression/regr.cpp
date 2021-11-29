
    //
    //  regr
    //
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #ifdef _NOTEBOOK
    #include "../../src/minictorch.hpp"
    #else
    #include "minictorch.hpp"
    #endif
    
    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  fc1_weight;
    extern Tensor  fc1_bias;
    extern Tensor  fc2_weight;
    extern Tensor  fc2_bias;
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    extern Tensor  Constant1;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [63, 1], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {63,1};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc1]/weight/weight.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {32,1};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc1]/bias/bias.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {32};
            fc1_bias.reshape( shape );
            forward_result[2] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'output_id': 0, 'shape': [63, 32], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {63,32};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'MSE/Net[net]/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [63, 32], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {63,32};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc2]/weight/weight.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {16,32};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_weight );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc2]/bias/bias.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {16};
            fc2_bias.reshape( shape );
            forward_result[6] = new VariableTensor( fc2_bias );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc2]/input.5', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [63, 16], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {63,16};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'MSE/Net[net]/input.7', 'op': 'aten::relu', 'in': [7], 'output_id': 0, 'shape': [63, 16], 'out': [11], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {63,16};
            ReluOp* op = new ReluOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc3]/weight/weight', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [11], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {1,16};
            fc3_weight.reshape( shape );
            forward_result[9] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc3]/bias/bias', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {1};
            fc3_bias.reshape( shape );
            forward_result[10] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'MSE/Net[net]/Linear[fc3]/input.9', 'op': 'aten::linear', 'in': [8, 9, 10], 'output_id': 0, 'shape': [63, 1], 'out': [13], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {63,1};
            LinearOp* op = new LinearOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'MSE/Loss[loss]/target.1', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [63, 1], 'constant_value': [0.0, 0.0998, 0.1987, 0.2955, 0.3894, 0.4794, 0.5646, 0.6442, 0.7174, 0.7833, 0.8415, 0.8912, 0.932, 0.9636, 0.9854, 0.9975, 0.9996, 0.9917, 0.9738, 0.9463, 0.9093, 0.8632, 0.8085, 0.7457, 0.6755, 0.5985, 0.5155, 0.4274, 0.335, 0.2392, 0.1411, 0.0416, -0.0584, -0.1577, -0.2555, -0.3508, -0.4425, -0.5298, -0.6119, -0.6878, -0.7568, -0.8183, -0.8716, -0.9162, -0.9516, -0.9775, -0.9937, -0.9999, -0.9962, -0.9825, -0.9589, -0.9258, -0.8835, -0.8323, -0.7728, -0.7055, -0.6313, -0.5507, -0.4646, -0.3739, -0.2794, -0.1822, -0.0831], 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {63,1};
            Constant1.reshape( shape );
            forward_result[12] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'MSE/Loss[loss]/90', 'op': 'prim::ListConstruct', 'in': [11, 12], 'output_id': 0, 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'MSE/Loss[loss]/91', 'op': 'aten::broadcast_tensors', 'in': [13], 'output_id': 0, 'shape': [], 'out': [16, 15], 'sorted_id': 14}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'MSE/Loss[loss]/input', 'op': 'prim::ListUnpack', 'in': [14], 'output_id': 0, 'shape': [63, 1], 'out': [18], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {63,1};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[15] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'MSE/Loss[loss]/target', 'op': 'prim::ListUnpack', 'in': [14], 'output_id': 1, 'shape': [63, 1], 'out': [18], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {63,1};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[16] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'MSE/Loss[loss]/88', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [18], 'sorted_id': 17}
        {
            Tensor c = (fprec)1.0;
            forward_result[17] = new VariableTensor( c, false );
        }
        
        // {'name': 'MSE/Loss[loss]/94', 'op': 'aten::mse_loss', 'in': [15, 16, 17], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            MseLossOp* op = new MseLossOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [18], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 19}
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
        vector<MCTNode*> forward_result(20);
    
        // input data
        Tensor::shape_type shape = {63,1};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 18 );
    #else
        do_train1( forward_result, input_var, 18 );
    #endif
        
        return 0;
    }
    