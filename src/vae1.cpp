
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
    extern Tensor  fc2_mean_weight;
    extern Tensor  fc2_mean_bias;
    extern Tensor  fc2_var_weight;
    extern Tensor  fc2_var_bias;
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    extern Tensor  fc4_weight;
    extern Tensor  fc4_bias;
    
    int main()
    {
        vector<MCTNode*> forward_result(40);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [32, 64], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/88', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/87', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {16};
            forward_result[2] = new VariableTensor(fc1_bias);
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'shape': [32, 16], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [32, 16], 'out': [7, 24], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/91', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_mean_weight);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/90', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            forward_result[6] = new VariableTensor(fc2_mean_bias);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/92', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [32, 2], 'out': [13, 9, 30], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.0, 'out': [9], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)0.0;
            forward_result[8] = new VariableTensor( c );
        }
        
        // {'name': 'Net/34', 'op': 'aten::size', 'in': [7, 8], 'shape': [], 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {};
            SizeOp* op = new SizeOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
        }
        
        // {'name': 'Net/35', 'op': 'prim::NumToTensor', 'in': [9], 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[10] = op;
            
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'Net/39', 'op': 'aten::Int', 'in': [10], 'shape': [], 'out': [16], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "Int" );
            forward_result[11] = op;
            
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[12] = new VariableTensor( c );
        }
        
        // {'name': 'Net/37', 'op': 'aten::size', 'in': [7, 12], 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {};
            SizeOp* op = new SizeOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/38', 'op': 'prim::NumToTensor', 'in': [13], 'shape': [], 'out': [15], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[14] = op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/40', 'op': 'aten::Int', 'in': [14], 'shape': [], 'out': [16], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "Int" );
            forward_result[15] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/41', 'op': 'prim::ListConstruct', 'in': [11, 15], 'shape': [], 'out': [21], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {};
            ListConstructOp* op = new ListConstructOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/42', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 6.0, 'out': [21], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)6.0;
            forward_result[17] = new VariableTensor( c );
        }
        
        // {'name': 'Net/43', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [21], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {};
            forward_result[18] = NULL;
        }
        
        // {'name': 'Net/44', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [21], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {};
            forward_result[19] = NULL;
        }
        
        // {'name': 'Net/45', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.0, 'out': [21], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)0.0;
            forward_result[20] = new VariableTensor( c );
        }
        
        // {'name': 'Net/eps', 'op': 'aten::randn', 'in': [16, 17, 18, 19, 20], 'shape': [32, 2], 'out': [28], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {32,2};
            RandnOp* op = new RandnOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/94', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[22] = new VariableTensor(fc2_var_weight);
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/93', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {2};
            forward_result[23] = new VariableTensor(fc2_var_bias);
        }
        
        // {'name': 'Net/Linear[fc2_var]/95', 'op': 'aten::linear', 'in': [4, 22, 23], 'shape': [32, 2], 'out': [26], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/47', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [26], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)0.5;
            forward_result[25] = new VariableTensor( c );
        }
        
        // {'name': 'Net/48', 'op': 'aten::mul', 'in': [24, 25], 'shape': [32, 2], 'out': [27], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/49', 'op': 'aten::exp', 'in': [26], 'shape': [32, 2], 'out': [28], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'Net/50', 'op': 'aten::mul', 'in': [21, 27], 'shape': [32, 2], 'out': [30], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'Net/51', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [30], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[29] = new VariableTensor( c );
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [7, 28, 29], 'shape': [32, 2], 'out': [33], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/97', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [33], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[31] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/96', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {16};
            forward_result[32] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [30, 31, 32], 'shape': [32, 16], 'out': [34], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [33], 'shape': [32, 16], 'out': [37], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[33] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/100', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [37], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[35] = new VariableTensor(fc4_weight);
        }
        
        // {'name': 'Net/Linear[fc4]/bias/99', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {64};
            forward_result[36] = new VariableTensor(fc4_bias);
        }
        
        // {'name': 'Net/Linear[fc4]/101', 'op': 'aten::linear', 'in': [34, 35, 36], 'shape': [32, 64], 'out': [38], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'Net/56', 'op': 'aten::sigmoid', 'in': [37], 'shape': [32, 64], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [38], 'shape': [32, 64], 'out': [], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {32,64};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[38]->forward();
        for(int k=0;k<=38;k++) {
           if( forward_result[k] )  forward_result[k]->forward();
        }
        auto o = forward_result[38]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[38]->grad = xt::ones_like( forward_result[38]->output );
        //forward_result[38]->backward();
        for(int k=38;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    