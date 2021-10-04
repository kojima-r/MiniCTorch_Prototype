
    //
    //  vae3
    //
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
    extern Tensor  bn1_weight;
    extern Tensor  bn1_bias;
    extern Tensor  bn1_running_mean;
    extern Tensor  bn1_running_var;
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
        vector<MCTNode*> forward_result(111);
    
        // input data
        Tensor::shape_type shape = {1000,784};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x0', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [1000, 784], 'out': [71, 4], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {1000,784};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'VAE/43', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -1.0, 'out': [3], 'sorted_id': 1}
        {
            Tensor c = (fprec)-1.0;
            forward_result[1] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/44', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 784.0, 'out': [3], 'sorted_id': 2}
        {
            Tensor c = (fprec)784.0;
            forward_result[2] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/45', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [4], 'sorted_id': 3}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'VAE/input.1', 'op': 'aten::view', 'in': [0, 3], 'output_id': 0, 'shape': [1000, 784], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {1000,784};
            ViewOp* op = new ViewOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'VAE/Linear[fc1]/weight/211', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {20,784};
            fc1_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'VAE/Linear[fc1]/bias/210', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {20};
            forward_result[6] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'VAE/Linear[fc1]/input.3', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [1000, 20], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {1000,20};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'VAE/input.5', 'op': 'aten::relu', 'in': [7], 'output_id': 0, 'shape': [1000, 20], 'out': [17], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {1000,20};
            ReluOp* op = new ReluOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/weight/220', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {20};
            bn1_weight.reshape( shape );
            forward_result[9] = new VariableTensor( bn1_weight );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/bias/219', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {20};
            forward_result[10] = new VariableTensor( bn1_bias );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/running_mean/218', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {20};
            forward_result[11] = new VariableTensor( bn1_running_mean );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/running_var/217', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {20};
            forward_result[12] = new VariableTensor( bn1_running_var );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/216', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [17], 'sorted_id': 13}
        {
            Tensor c = (fprec)0.0;
            forward_result[13] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/215', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.1, 'out': [17], 'sorted_id': 14}
        {
            Tensor c = (fprec)0.1;
            forward_result[14] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/214', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-05, 'out': [17], 'sorted_id': 15}
        {
            Tensor c = (fprec)1e-05;
            forward_result[15] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/213', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [17], 'sorted_id': 16}
        {
            Tensor c = (fprec)1.0;
            forward_result[16] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/BatchNorm1d[bn1]/input.7', 'op': 'aten::batch_norm', 'in': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'output_id': 0, 'shape': [1000, 20], 'out': [23, 20], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {1000,20};
            BatchNormOp* op = new BatchNormOp();
            forward_result[17] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
        }
        
        // {'name': 'VAE/Linear[fc2_mean]/weight/223', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {10,20};
            fc2_mean_weight.reshape( shape );
            forward_result[18] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'VAE/Linear[fc2_mean]/bias/222', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {10};
            forward_result[19] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'VAE/Linear[fc2_mean]/224', 'op': 'aten::linear', 'in': [17, 18, 19], 'output_id': 0, 'shape': [1000, 10], 'out': [27], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {1000,10};
            LinearOp* op = new LinearOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'VAE/Linear[fc2_var]/weight/226', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [23], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {10,20};
            fc2_var_weight.reshape( shape );
            forward_result[21] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'VAE/Linear[fc2_var]/bias/225', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [23], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {10};
            forward_result[22] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'VAE/Linear[fc2_var]/227', 'op': 'aten::linear', 'in': [17, 21, 22], 'output_id': 0, 'shape': [1000, 10], 'out': [25], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {1000,10};
            LinearOp* op = new LinearOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'VAE/56', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [25], 'sorted_id': 24}
        {
            Tensor c = (fprec)0.5;
            forward_result[24] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/57', 'op': 'aten::mul', 'in': [23, 24], 'output_id': 0, 'shape': [1000, 10], 'out': [26], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {1000,10};
            MulOp* op = new MulOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'VAE/58', 'op': 'aten::exp', 'in': [25], 'output_id': 0, 'shape': [1000, 10], 'out': [27], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {1000,10};
            ExpOp* op = new ExpOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'VAE/59', 'op': 'prim::ListConstruct', 'in': [20, 26], 'output_id': 0, 'shape': [], 'out': [28], 'sorted_id': 27}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'VAE/60', 'op': 'aten::broadcast_tensors', 'in': [27], 'output_id': 0, 'shape': [], 'out': [29, 54], 'sorted_id': 28}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'VAE/loc.1', 'op': 'prim::ListUnpack', 'in': [28], 'output_id': 0, 'shape': [1000, 10], 'out': [35, 92, 57, 31, 77], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {1000,10};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'VAE/63', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [31], 'sorted_id': 30}
        {
            Tensor c = (fprec)0.0;
            forward_result[30] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/64', 'op': 'aten::size', 'in': [29, 30], 'output_id': 0, 'shape': [], 'out': [32], 'sorted_id': 31}
        {
            SizeOp* op = new SizeOp();
            forward_result[31] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[30] );
        }
        
        // {'name': 'VAE/65', 'op': 'prim::NumToTensor', 'in': [31], 'output_id': 0, 'shape': [], 'out': [44, 33], 'sorted_id': 32}
        {
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[32] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'VAE/74', 'op': 'aten::Int', 'in': [32], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 33}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[33] = op;
            
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'VAE/66', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [35], 'sorted_id': 34}
        {
            Tensor c = (fprec)1.0;
            forward_result[34] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/67', 'op': 'aten::size', 'in': [29, 34], 'output_id': 0, 'shape': [], 'out': [36], 'sorted_id': 35}
        {
            SizeOp* op = new SizeOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/68', 'op': 'prim::NumToTensor', 'in': [35], 'output_id': 0, 'shape': [], 'out': [37, 45], 'sorted_id': 36}
        {
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[36] = op;
            
            op->set_inputs( forward_result[35] );
        }
        
        // {'name': 'VAE/75', 'op': 'aten::Int', 'in': [36], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 37}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[37] = op;
            
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'VAE/76', 'op': 'prim::ListConstruct', 'in': [33, 37], 'output_id': 0, 'shape': [], 'out': [43], 'sorted_id': 38}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/77', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [43], 'sorted_id': 39}
        {
            Tensor c = (fprec)6.0;
            forward_result[39] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/78', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [43], 'sorted_id': 40}
        {
            forward_result[40] = NULL;
        }
        
        // {'name': 'VAE/79', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [43], 'sorted_id': 41}
        {
            forward_result[41] = NULL;
        }
        
        // {'name': 'VAE/80', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [43], 'sorted_id': 42}
        {
            Tensor c = (fprec)0.0;
            forward_result[42] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/81', 'op': 'aten::zeros', 'in': [38, 39, 40, 41, 42], 'output_id': 0, 'shape': [1000, 10], 'out': [53], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {1000,10};
            ZerosOp* op = new ZerosOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'VAE/82', 'op': 'aten::Int', 'in': [32], 'output_id': 0, 'shape': [], 'out': [46], 'sorted_id': 44}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[44] = op;
            
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'VAE/83', 'op': 'aten::Int', 'in': [36], 'output_id': 0, 'shape': [], 'out': [46], 'sorted_id': 45}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[45] = op;
            
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'VAE/84', 'op': 'prim::ListConstruct', 'in': [44, 45], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 46}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'VAE/85', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [51], 'sorted_id': 47}
        {
            Tensor c = (fprec)6.0;
            forward_result[47] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/86', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 48}
        {
            forward_result[48] = NULL;
        }
        
        // {'name': 'VAE/87', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 49}
        {
            forward_result[49] = NULL;
        }
        
        // {'name': 'VAE/88', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [51], 'sorted_id': 50}
        {
            Tensor c = (fprec)0.0;
            forward_result[50] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/89', 'op': 'aten::ones', 'in': [46, 47, 48, 49, 50], 'output_id': 0, 'shape': [1000, 10], 'out': [53], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {1000,10};
            OnesOp* op = new OnesOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'VAE/90', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [53], 'sorted_id': 52}
        {
            forward_result[52] = NULL;
        }
        
        // {'name': 'VAE/eps', 'op': 'aten::normal', 'in': [43, 51, 52], 'output_id': 0, 'shape': [1000, 10], 'out': [55], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {1000,10};
            NormalOp* op = new NormalOp();
            forward_result[53] = op;
            
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[52] );
        }
        
        // {'name': 'VAE/value.1', 'op': 'prim::ListUnpack', 'in': [28], 'output_id': 1, 'shape': [1000, 10], 'out': [87, 83, 55], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {1000,10};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[54] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'VAE/92', 'op': 'aten::mul', 'in': [53, 54], 'output_id': 0, 'shape': [1000, 10], 'out': [57], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {1000,10};
            MulOp* op = new MulOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'VAE/93', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [57], 'sorted_id': 56}
        {
            Tensor c = (fprec)1.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/input.9', 'op': 'aten::add', 'in': [29, 55, 56], 'output_id': 0, 'shape': [1000, 10], 'out': [60], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {1000,10};
            AddOp* op = new AddOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'VAE/Linear[fc3]/weight/229', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [60], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {20,10};
            fc3_weight.reshape( shape );
            forward_result[58] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'VAE/Linear[fc3]/bias/228', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [60], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {20};
            forward_result[59] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'VAE/Linear[fc3]/input.11', 'op': 'aten::linear', 'in': [57, 58, 59], 'output_id': 0, 'shape': [1000, 20], 'out': [61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {1000,20};
            LinearOp* op = new LinearOp();
            forward_result[60] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[59] );
        }
        
        // {'name': 'VAE/input.13', 'op': 'aten::relu', 'in': [60], 'output_id': 0, 'shape': [1000, 20], 'out': [64], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {1000,20};
            ReluOp* op = new ReluOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[60] );
        }
        
        // {'name': 'VAE/Dropout[drop1]/232', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.2, 'out': [64], 'sorted_id': 62}
        {
            Tensor c = (fprec)0.2;
            forward_result[62] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Dropout[drop1]/231', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [64], 'sorted_id': 63}
        {
            Tensor c = (fprec)0.0;
            forward_result[63] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Dropout[drop1]/input.15', 'op': 'aten::dropout', 'in': [61, 62, 63], 'output_id': 0, 'shape': [1000, 20], 'out': [67], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {1000,20};
            DropoutOp* op = new DropoutOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'VAE/Linear[fc4]/weight/235', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {784,20};
            fc4_weight.reshape( shape );
            forward_result[65] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'VAE/Linear[fc4]/bias/234', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {784};
            forward_result[66] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'VAE/Linear[fc4]/236', 'op': 'aten::linear', 'in': [64, 65, 66], 'output_id': 0, 'shape': [1000, 784], 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {1000,784};
            LinearOp* op = new LinearOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[66] );
        }
        
        // {'name': 'VAE/input', 'op': 'aten::sigmoid', 'in': [67], 'output_id': 0, 'shape': [1000, 784], 'out': [71], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {1000,784};
            SigmoidOp* op = new SigmoidOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'VAE/116', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [71], 'sorted_id': 69}
        {
            forward_result[69] = NULL;
        }
        
        // {'name': 'VAE/117', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [71], 'sorted_id': 70}
        {
            Tensor c = (fprec)1.0;
            forward_result[70] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/e1', 'op': 'aten::binary_cross_entropy', 'in': [68, 0, 69, 70], 'output_id': 0, 'shape': [], 'out': [109], 'sorted_id': 71}
        {
            BCELossOp* op = new BCELossOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[68] );
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'VAE/119', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [77], 'sorted_id': 72}
        {
            Tensor c = (fprec)6.0;
            forward_result[72] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/120', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [77], 'sorted_id': 73}
        {
            Tensor c = (fprec)0.0;
            forward_result[73] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/121', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [77], 'sorted_id': 74}
        {
            forward_result[74] = NULL;
        }
        
        // {'name': 'VAE/122', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [77], 'sorted_id': 75}
        {
            Tensor c = (fprec)0.0;
            forward_result[75] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/123', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [77], 'sorted_id': 76}
        {
            forward_result[76] = NULL;
        }
        
        // {'name': 'VAE/124', 'op': 'aten::zeros_like', 'in': [29, 72, 73, 74, 75, 76], 'output_id': 0, 'shape': [1000, 10], 'out': [84], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {1000,10};
            FullLikeOp* op = new FullLikeOp( 0.0 );
            forward_result[77] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[76] );
        }
        
        // {'name': 'VAE/125', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [83], 'sorted_id': 78}
        {
            Tensor c = (fprec)6.0;
            forward_result[78] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/126', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [83], 'sorted_id': 79}
        {
            Tensor c = (fprec)0.0;
            forward_result[79] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/127', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [83], 'sorted_id': 80}
        {
            forward_result[80] = NULL;
        }
        
        // {'name': 'VAE/128', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [83], 'sorted_id': 81}
        {
            Tensor c = (fprec)0.0;
            forward_result[81] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/129', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [83], 'sorted_id': 82}
        {
            forward_result[82] = NULL;
        }
        
        // {'name': 'VAE/130', 'op': 'aten::ones_like', 'in': [54, 78, 79, 80, 81, 82], 'output_id': 0, 'shape': [1000, 10], 'out': [84], 'sorted_id': 83}
        {
            Tensor::shape_type shape = {1000,10};
            FullLikeOp* op = new FullLikeOp( 1.0 );
            forward_result[83] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[78] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[80] );
            op->set_inputs( forward_result[81] );
            op->set_inputs( forward_result[82] );
        }
        
        // {'name': 'VAE/131', 'op': 'prim::ListConstruct', 'in': [77, 83], 'output_id': 0, 'shape': [], 'out': [85], 'sorted_id': 84}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[83] );
        }
        
        // {'name': 'VAE/132', 'op': 'aten::broadcast_tensors', 'in': [84], 'output_id': 0, 'shape': [], 'out': [90, 86], 'sorted_id': 85}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[84] );
        }
        
        // {'name': 'VAE/value', 'op': 'prim::ListUnpack', 'in': [85], 'output_id': 1, 'shape': [1000, 10], 'out': [87, 93], 'sorted_id': 86}
        {
            Tensor::shape_type shape = {1000,10};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[86] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'VAE/146', 'op': 'aten::div', 'in': [54, 86], 'output_id': 0, 'shape': [1000, 10], 'out': [89], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {1000,10};
            DivOp* op = new DivOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[86] );
        }
        
        // {'name': 'VAE/147', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [89], 'sorted_id': 88}
        {
            Tensor c = (fprec)2.0;
            forward_result[88] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/var_ratio', 'op': 'aten::pow', 'in': [87, 88], 'output_id': 0, 'shape': [1000, 10], 'out': [101, 97], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {1000,10};
            PowOp* op = new PowOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'VAE/loc', 'op': 'prim::ListUnpack', 'in': [85], 'output_id': 0, 'shape': [1000, 10], 'out': [92], 'sorted_id': 90}
        {
            Tensor::shape_type shape = {1000,10};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[90] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'VAE/149', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [92], 'sorted_id': 91}
        {
            Tensor c = (fprec)1.0;
            forward_result[91] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/150', 'op': 'aten::sub', 'in': [29, 90, 91], 'output_id': 0, 'shape': [1000, 10], 'out': [93], 'sorted_id': 92}
        {
            Tensor::shape_type shape = {1000,10};
            SubOp* op = new SubOp();
            forward_result[92] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[90] );
            op->set_inputs( forward_result[91] );
        }
        
        // {'name': 'VAE/151', 'op': 'aten::div', 'in': [92, 86], 'output_id': 0, 'shape': [1000, 10], 'out': [95], 'sorted_id': 93}
        {
            Tensor::shape_type shape = {1000,10};
            DivOp* op = new DivOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[92] );
            op->set_inputs( forward_result[86] );
        }
        
        // {'name': 'VAE/152', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [95], 'sorted_id': 94}
        {
            Tensor c = (fprec)2.0;
            forward_result[94] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/t1', 'op': 'aten::pow', 'in': [93, 94], 'output_id': 0, 'shape': [1000, 10], 'out': [97], 'sorted_id': 95}
        {
            Tensor::shape_type shape = {1000,10};
            PowOp* op = new PowOp();
            forward_result[95] = op;
            
            op->set_inputs( forward_result[93] );
            op->set_inputs( forward_result[94] );
        }
        
        // {'name': 'VAE/154', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [97], 'sorted_id': 96}
        {
            Tensor c = (fprec)1.0;
            forward_result[96] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/155', 'op': 'aten::add', 'in': [89, 95, 96], 'output_id': 0, 'shape': [1000, 10], 'out': [100], 'sorted_id': 97}
        {
            Tensor::shape_type shape = {1000,10};
            AddOp* op = new AddOp();
            forward_result[97] = op;
            
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[95] );
            op->set_inputs( forward_result[96] );
        }
        
        // {'name': 'VAE/156', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [100], 'sorted_id': 98}
        {
            Tensor c = (fprec)1.0;
            forward_result[98] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/157', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [100], 'sorted_id': 99}
        {
            Tensor c = (fprec)1.0;
            forward_result[99] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/158', 'op': 'aten::sub', 'in': [97, 98, 99], 'output_id': 0, 'shape': [1000, 10], 'out': [103], 'sorted_id': 100}
        {
            Tensor::shape_type shape = {1000,10};
            SubOp* op = new SubOp();
            forward_result[100] = op;
            
            op->set_inputs( forward_result[97] );
            op->set_inputs( forward_result[98] );
            op->set_inputs( forward_result[99] );
        }
        
        // {'name': 'VAE/159', 'op': 'aten::log', 'in': [89], 'output_id': 0, 'shape': [1000, 10], 'out': [103], 'sorted_id': 101}
        {
            Tensor::shape_type shape = {1000,10};
            LogOp* op = new LogOp();
            forward_result[101] = op;
            
            op->set_inputs( forward_result[89] );
        }
        
        // {'name': 'VAE/160', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [103], 'sorted_id': 102}
        {
            Tensor c = (fprec)1.0;
            forward_result[102] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/161', 'op': 'aten::sub', 'in': [100, 101, 102], 'output_id': 0, 'shape': [1000, 10], 'out': [105], 'sorted_id': 103}
        {
            Tensor::shape_type shape = {1000,10};
            SubOp* op = new SubOp();
            forward_result[103] = op;
            
            op->set_inputs( forward_result[100] );
            op->set_inputs( forward_result[101] );
            op->set_inputs( forward_result[102] );
        }
        
        // {'name': 'VAE/162', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [105], 'sorted_id': 104}
        {
            Tensor c = (fprec)0.5;
            forward_result[104] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/163', 'op': 'aten::mul', 'in': [103, 104], 'output_id': 0, 'shape': [1000, 10], 'out': [107], 'sorted_id': 105}
        {
            Tensor::shape_type shape = {1000,10};
            MulOp* op = new MulOp();
            forward_result[105] = op;
            
            op->set_inputs( forward_result[103] );
            op->set_inputs( forward_result[104] );
        }
        
        // {'name': 'VAE/164', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [107], 'sorted_id': 106}
        {
            forward_result[106] = NULL;
        }
        
        // {'name': 'VAE/e2', 'op': 'aten::sum', 'in': [105, 106], 'output_id': 0, 'shape': [], 'out': [109], 'sorted_id': 107}
        {
            SumOp*    op = new SumOp();
            forward_result[107] = op;
            
            op->set_inputs( forward_result[105] );
            op->set_inputs( forward_result[106] );
        }
        
        // {'name': 'VAE/166', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [109], 'sorted_id': 108}
        {
            Tensor c = (fprec)1.0;
            forward_result[108] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/167', 'op': 'aten::add', 'in': [71, 107, 108], 'output_id': 0, 'shape': [], 'out': [110], 'sorted_id': 109}
        {
            AddOp* op = new AddOp();
            forward_result[109] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[108] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [109], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 110}
        {
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[109]->forward();
        for(int k=0;k<=109;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[109]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[109]->grad = xt::ones_like( forward_result[109]->output );
        //forward_result[109]->backward();
        for(int k=109;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    