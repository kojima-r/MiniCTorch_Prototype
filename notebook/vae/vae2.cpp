
    //
    //  vae2
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
    extern Tensor  fc2_mean_weight;
    extern Tensor  fc2_mean_bias;
    extern Tensor  fc2_var_weight;
    extern Tensor  fc2_var_bias;
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    extern Tensor  fc4_weight;
    extern Tensor  fc4_bias;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [55, 3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/weight.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'Net/Linear[fc1]/bias/bias.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {16};
            fc1_bias.reshape( shape );
            forward_result[2] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'output_id': 0, 'shape': [32, 16], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [32, 16], 'out': [7, 10], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/weight.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/bias.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            fc2_mean_bias.reshape( shape );
            forward_result[6] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/187', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [32, 2], 'out': [14], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/weight.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [10], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[8] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/bias.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {2};
            fc2_var_bias.reshape( shape );
            forward_result[9] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'Net/Linear[fc2_var]/190', 'op': 'aten::linear', 'in': [4, 8, 9], 'output_id': 0, 'shape': [32, 2], 'out': [12], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[10] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [12], 'sorted_id': 11}
        {
            Tensor c = (fprec)0.5;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/34', 'op': 'aten::mul', 'in': [10, 11], 'output_id': 0, 'shape': [32, 2], 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[12] = op;
            
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
        }
        
        // {'name': 'Net/35', 'op': 'aten::exp', 'in': [12], 'output_id': 0, 'shape': [32, 2], 'out': [14], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::ListConstruct', 'in': [7, 13], 'output_id': 0, 'shape': [], 'out': [15], 'sorted_id': 14}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/37', 'op': 'aten::broadcast_tensors', 'in': [14], 'output_id': 0, 'shape': [], 'out': [16, 41], 'sorted_id': 15}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[15] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/loc.1', 'op': 'prim::ListUnpack', 'in': [15], 'output_id': 0, 'shape': [32, 2], 'out': [22, 76, 61, 44, 18], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[16] = op;
            
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/40', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [18], 'sorted_id': 17}
        {
            Tensor c = (fprec)0.0;
            forward_result[17] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/41', 'op': 'aten::size', 'in': [16, 17], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            SizeOp* op = new SizeOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
        }
        
        // {'name': 'Net/42', 'op': 'prim::NumToTensor', 'in': [18], 'output_id': 0, 'shape': [], 'out': [20, 31], 'sorted_id': 19}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'Net/56', 'op': 'aten::Int', 'in': [19], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 20}
        {
            IntOp* op = new IntOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/43', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [22], 'sorted_id': 21}
        {
            Tensor c = (fprec)1.0;
            forward_result[21] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/44', 'op': 'aten::size', 'in': [16, 21], 'output_id': 0, 'shape': [], 'out': [23], 'sorted_id': 22}
        {
            SizeOp* op = new SizeOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[21] );
        }
        
        // {'name': 'Net/45', 'op': 'prim::NumToTensor', 'in': [22], 'output_id': 0, 'shape': [], 'out': [32, 24], 'sorted_id': 23}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'Net/57', 'op': 'aten::Int', 'in': [23], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 24}
        {
            IntOp* op = new IntOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/58', 'op': 'prim::ListConstruct', 'in': [20, 24], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 25}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/59', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [30], 'sorted_id': 26}
        {
            Tensor c = (fprec)6.0;
            forward_result[26] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/60', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 27}
        {
            forward_result[27] = NULL;
        }
        
        // {'name': 'Net/61', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 28}
        {
            forward_result[28] = NULL;
        }
        
        // {'name': 'Net/62', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [30], 'sorted_id': 29}
        {
            Tensor c = (fprec)0.0;
            forward_result[29] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/63', 'op': 'aten::zeros', 'in': [25, 26, 27, 28, 29], 'output_id': 0, 'shape': [32, 2], 'out': [40], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {32,2};
            ZerosOp* op = new ZerosOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/64', 'op': 'aten::Int', 'in': [19], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 31}
        {
            IntOp* op = new IntOp();
            forward_result[31] = op;
            
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/65', 'op': 'aten::Int', 'in': [23], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            IntOp* op = new IntOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/66', 'op': 'prim::ListConstruct', 'in': [31, 32], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 33}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/67', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [38], 'sorted_id': 34}
        {
            Tensor c = (fprec)6.0;
            forward_result[34] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/68', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 35}
        {
            forward_result[35] = NULL;
        }
        
        // {'name': 'Net/69', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 36}
        {
            forward_result[36] = NULL;
        }
        
        // {'name': 'Net/70', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [38], 'sorted_id': 37}
        {
            Tensor c = (fprec)0.0;
            forward_result[37] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/71', 'op': 'aten::ones', 'in': [33, 34, 35, 36, 37], 'output_id': 0, 'shape': [32, 2], 'out': [40], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {32,2};
            OnesOp* op = new OnesOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/72', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [40], 'sorted_id': 39}
        {
            forward_result[39] = NULL;
        }
        
        // {'name': 'Net/eps', 'op': 'aten::normal', 'in': [30, 38, 39], 'output_id': 0, 'shape': [32, 2], 'out': [42], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {32,2};
            NormalOp* op = new NormalOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Net/value.1', 'op': 'prim::ListUnpack', 'in': [15], 'output_id': 1, 'shape': [32, 2], 'out': [71, 42, 67], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[41] = op;
            
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/74', 'op': 'aten::mul', 'in': [40, 41], 'output_id': 0, 'shape': [32, 2], 'out': [44], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[41] );
        }
        
        // {'name': 'Net/75', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [44], 'sorted_id': 43}
        {
            Tensor c = (fprec)1.0;
            forward_result[43] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [16, 42, 43], 'output_id': 0, 'shape': [32, 2], 'out': [47], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[42] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/weight.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[45] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'Net/Linear[fc3]/bias/bias.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {16};
            fc3_bias.reshape( shape );
            forward_result[46] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [44, 45, 46], 'output_id': 0, 'shape': [32, 16], 'out': [48], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/input.9', 'op': 'aten::relu', 'in': [47], 'output_id': 0, 'shape': [32, 16], 'out': [51], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[48] = op;
            
            op->set_inputs( forward_result[47] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/weight', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[49] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'Net/Linear[fc4]/bias/bias', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {64};
            fc4_bias.reshape( shape );
            forward_result[50] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'Net/Linear[fc4]/196', 'op': 'aten::linear', 'in': [48, 49, 50], 'output_id': 0, 'shape': [32, 64], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::sigmoid', 'in': [51], 'output_id': 0, 'shape': [32, 64], 'out': [55], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Net/95', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [55], 'sorted_id': 53}
        {
            forward_result[53] = NULL;
        }
        
        // {'name': 'Net/96', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [55], 'sorted_id': 54}
        {
            Tensor c = (fprec)2.0;
            forward_result[54] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e1', 'op': 'aten::binary_cross_entropy', 'in': [52, 0, 53, 54], 'output_id': 0, 'shape': [], 'out': [93], 'sorted_id': 55}
        {
            BCELossOp* op = new BCELossOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'Net/98', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [61], 'sorted_id': 56}
        {
            Tensor c = (fprec)6.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/99', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [61], 'sorted_id': 57}
        {
            Tensor c = (fprec)0.0;
            forward_result[57] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/100', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [61], 'sorted_id': 58}
        {
            forward_result[58] = NULL;
        }
        
        // {'name': 'Net/101', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [61], 'sorted_id': 59}
        {
            Tensor c = (fprec)0.0;
            forward_result[59] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/102', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [61], 'sorted_id': 60}
        {
            forward_result[60] = NULL;
        }
        
        // {'name': 'Net/103', 'op': 'aten::zeros_like', 'in': [16, 56, 57, 58, 59, 60], 'output_id': 0, 'shape': [32, 2], 'out': [68], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {32,2};
            FullLikeOp* op = new FullLikeOp( 0.0 );
            forward_result[61] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[60] );
        }
        
        // {'name': 'Net/104', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [67], 'sorted_id': 62}
        {
            Tensor c = (fprec)6.0;
            forward_result[62] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/105', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [67], 'sorted_id': 63}
        {
            Tensor c = (fprec)0.0;
            forward_result[63] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/106', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 64}
        {
            forward_result[64] = NULL;
        }
        
        // {'name': 'Net/107', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [67], 'sorted_id': 65}
        {
            Tensor c = (fprec)0.0;
            forward_result[65] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/108', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            forward_result[66] = NULL;
        }
        
        // {'name': 'Net/109', 'op': 'aten::ones_like', 'in': [41, 62, 63, 64, 65, 66], 'output_id': 0, 'shape': [32, 2], 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {32,2};
            FullLikeOp* op = new FullLikeOp( 1.0 );
            forward_result[67] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[66] );
        }
        
        // {'name': 'Net/110', 'op': 'prim::ListConstruct', 'in': [61, 67], 'output_id': 0, 'shape': [], 'out': [69], 'sorted_id': 68}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'Net/111', 'op': 'aten::broadcast_tensors', 'in': [68], 'output_id': 0, 'shape': [], 'out': [74, 70], 'sorted_id': 69}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'Net/value', 'op': 'prim::ListUnpack', 'in': [69], 'output_id': 1, 'shape': [32, 2], 'out': [71, 77], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[70] = op;
            
            op->set_inputs( forward_result[69] );
        }
        
        // {'name': 'Net/130', 'op': 'aten::div', 'in': [41, 70], 'output_id': 0, 'shape': [32, 2], 'out': [73], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/131', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [73], 'sorted_id': 72}
        {
            Tensor c = (fprec)2.0;
            forward_result[72] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/var_ratio', 'op': 'aten::pow', 'in': [71, 72], 'output_id': 0, 'shape': [32, 2], 'out': [85, 81], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/loc', 'op': 'prim::ListUnpack', 'in': [69], 'output_id': 0, 'shape': [32, 2], 'out': [76], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[74] = op;
            
            op->set_inputs( forward_result[69] );
        }
        
        // {'name': 'Net/133', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [76], 'sorted_id': 75}
        {
            Tensor c = (fprec)1.0;
            forward_result[75] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/134', 'op': 'aten::sub', 'in': [16, 74, 75], 'output_id': 0, 'shape': [32, 2], 'out': [77], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
        }
        
        // {'name': 'Net/135', 'op': 'aten::div', 'in': [76, 70], 'output_id': 0, 'shape': [32, 2], 'out': [79], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/136', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [79], 'sorted_id': 78}
        {
            Tensor c = (fprec)2.0;
            forward_result[78] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/t1', 'op': 'aten::pow', 'in': [77, 78], 'output_id': 0, 'shape': [32, 2], 'out': [81], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[78] );
        }
        
        // {'name': 'Net/138', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [81], 'sorted_id': 80}
        {
            Tensor c = (fprec)1.0;
            forward_result[80] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/139', 'op': 'aten::add', 'in': [73, 79, 80], 'output_id': 0, 'shape': [32, 2], 'out': [84], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/140', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [84], 'sorted_id': 82}
        {
            Tensor c = (fprec)1.0;
            forward_result[82] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/141', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [84], 'sorted_id': 83}
        {
            Tensor c = (fprec)1.0;
            forward_result[83] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/142', 'op': 'aten::sub', 'in': [81, 82, 83], 'output_id': 0, 'shape': [32, 2], 'out': [87], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[81] );
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[83] );
        }
        
        // {'name': 'Net/143', 'op': 'aten::log', 'in': [73], 'output_id': 0, 'shape': [32, 2], 'out': [87], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {32,2};
            LogOp* op = new LogOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Net/144', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [87], 'sorted_id': 86}
        {
            Tensor c = (fprec)1.0;
            forward_result[86] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/145', 'op': 'aten::sub', 'in': [84, 85, 86], 'output_id': 0, 'shape': [32, 2], 'out': [89], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[86] );
        }
        
        // {'name': 'Net/146', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [89], 'sorted_id': 88}
        {
            Tensor c = (fprec)0.5;
            forward_result[88] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/147', 'op': 'aten::mul', 'in': [87, 88], 'output_id': 0, 'shape': [32, 2], 'out': [91], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'Net/148', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [91], 'sorted_id': 90}
        {
            forward_result[90] = NULL;
        }
        
        // {'name': 'Net/e2', 'op': 'aten::sum', 'in': [89, 90], 'output_id': 0, 'shape': [], 'out': [93], 'sorted_id': 91}
        {
            SumOp*    op = new SumOp();
            forward_result[91] = op;
            
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[90] );
        }
        
        // {'name': 'Net/150', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [93], 'sorted_id': 92}
        {
            Tensor c = (fprec)1.0;
            forward_result[92] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/151', 'op': 'aten::add', 'in': [55, 91, 92], 'output_id': 0, 'shape': [], 'out': [94], 'sorted_id': 93}
        {
            AddOp* op = new AddOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[91] );
            op->set_inputs( forward_result[92] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [93], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 94}
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
        vector<MCTNode*> forward_result(95);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 93 );
    #else
        do_train1( forward_result, input_var, 93 );
    #endif
        
        return 0;
    }
    