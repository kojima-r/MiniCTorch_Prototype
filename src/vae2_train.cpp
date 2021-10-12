    //
    //  vae2_train.cpp
    //
    /*#include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>*/
    #include"minictorch.hpp"

    /*
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
        vector<MCTNode*> forward_result(95);
        
        forward_result[0]->print_shape("xin",xin);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [55, 3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/173', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'Net/Linear[fc1]/bias/172', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {16};
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
        
        // {'name': 'Net/Linear[fc2_mean]/weight/176', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/175', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            forward_result[6] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/177', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [32, 2], 'out': [14], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/179', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [10], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[8] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/178', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {2};
            forward_result[9] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'Net/Linear[fc2_var]/180', 'op': 'aten::linear', 'in': [4, 8, 9], 'output_id': 0, 'shape': [32, 2], 'out': [12], 'sorted_id': 10}
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
            Tensor c = (float)0.5;
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
        
        // {'name': 'Net/loc.1', 'op': 'prim::ListUnpack', 'in': [15], 'output_id': 0, 'shape': [32, 2], 'out': [44, 18, 76, 22, 61], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[16] = op;
            
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/40', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [18], 'sorted_id': 17}
        {
            Tensor c = (float)0.0;
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
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[19] = op;
            
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'Net/51', 'op': 'aten::Int', 'in': [19], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 20}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[20] = op;
            
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/43', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [22], 'sorted_id': 21}
        {
            Tensor c = (float)1.0;
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
            MoveOp* op = new MoveOp( "NumToTensor" );
            forward_result[23] = op;
            
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'Net/52', 'op': 'aten::Int', 'in': [23], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 24}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[24] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/53', 'op': 'prim::ListConstruct', 'in': [20, 24], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 25}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/54', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [30], 'sorted_id': 26}
        {
            Tensor c = (float)6.0;
            forward_result[26] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/55', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 27}
        {
            forward_result[27] = NULL;
        }
        
        // {'name': 'Net/56', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 28}
        {
            forward_result[28] = NULL;
        }
        
        // {'name': 'Net/57', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [30], 'sorted_id': 29}
        {
            Tensor c = (float)0.0;
            forward_result[29] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/58', 'op': 'aten::zeros', 'in': [25, 26, 27, 28, 29], 'output_id': 0, 'shape': [32, 2], 'out': [40], 'sorted_id': 30}
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
        
        // {'name': 'Net/59', 'op': 'aten::Int', 'in': [19], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 31}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[31] = op;
            
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/60', 'op': 'aten::Int', 'in': [23], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            MoveOp* op = new MoveOp( "Int" );
            forward_result[32] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/61', 'op': 'prim::ListConstruct', 'in': [31, 32], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 33}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/62', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [38], 'sorted_id': 34}
        {
            Tensor c = (float)6.0;
            forward_result[34] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/63', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 35}
        {
            forward_result[35] = NULL;
        }
        
        // {'name': 'Net/64', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [38], 'sorted_id': 36}
        {
            forward_result[36] = NULL;
        }
        
        // {'name': 'Net/65', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [38], 'sorted_id': 37}
        {
            Tensor c = (float)0.0;
            forward_result[37] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/66', 'op': 'aten::ones', 'in': [33, 34, 35, 36, 37], 'output_id': 0, 'shape': [32, 2], 'out': [40], 'sorted_id': 38}
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
        
        // {'name': 'Net/67', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [40], 'sorted_id': 39}
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
        
        // {'name': 'Net/value.1', 'op': 'prim::ListUnpack', 'in': [15], 'output_id': 1, 'shape': [32, 2], 'out': [42, 71, 67], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[41] = op;
            
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/69', 'op': 'aten::mul', 'in': [40, 41], 'output_id': 0, 'shape': [32, 2], 'out': [44], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[41] );
        }
        
        // {'name': 'Net/70', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [44], 'sorted_id': 43}
        {
            Tensor c = (float)1.0;
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
        
        // {'name': 'Net/Linear[fc3]/weight/182', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[45] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'Net/Linear[fc3]/bias/181', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {16};
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
        
        // {'name': 'Net/Linear[fc4]/weight/185', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[49] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'Net/Linear[fc4]/bias/184', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [51], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {64};
            forward_result[50] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'Net/Linear[fc4]/186', 'op': 'aten::linear', 'in': [48, 49, 50], 'output_id': 0, 'shape': [32, 64], 'out': [52], 'sorted_id': 51}
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
        
        // {'name': 'Net/90', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [55], 'sorted_id': 53}
        {
            forward_result[53] = NULL;
        }
        
        // {'name': 'Net/91', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [55], 'sorted_id': 54}
        {
            Tensor c = (float)1.0;
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
        
        // {'name': 'Net/93', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [61], 'sorted_id': 56}
        {
            Tensor c = (float)6.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/94', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [61], 'sorted_id': 57}
        {
            Tensor c = (float)0.0;
            forward_result[57] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/95', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [61], 'sorted_id': 58}
        {
            forward_result[58] = NULL;
        }
        
        // {'name': 'Net/96', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [61], 'sorted_id': 59}
        {
            Tensor c = (float)0.0;
            forward_result[59] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/97', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [61], 'sorted_id': 60}
        {
            forward_result[60] = NULL;
        }
        
        // {'name': 'Net/98', 'op': 'aten::zeros_like', 'in': [16, 56, 57, 58, 59, 60], 'output_id': 0, 'shape': [32, 2], 'out': [68], 'sorted_id': 61}
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
        
        // {'name': 'Net/99', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [67], 'sorted_id': 62}
        {
            Tensor c = (float)6.0;
            forward_result[62] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/100', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [67], 'sorted_id': 63}
        {
            Tensor c = (float)0.0;
            forward_result[63] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/101', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 64}
        {
            forward_result[64] = NULL;
        }
        
        // {'name': 'Net/102', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [67], 'sorted_id': 65}
        {
            Tensor c = (float)0.0;
            forward_result[65] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/103', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            forward_result[66] = NULL;
        }
        
        // {'name': 'Net/104', 'op': 'aten::ones_like', 'in': [41, 62, 63, 64, 65, 66], 'output_id': 0, 'shape': [32, 2], 'out': [68], 'sorted_id': 67}
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
        
        // {'name': 'Net/105', 'op': 'prim::ListConstruct', 'in': [61, 67], 'output_id': 0, 'shape': [], 'out': [69], 'sorted_id': 68}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'Net/106', 'op': 'aten::broadcast_tensors', 'in': [68], 'output_id': 0, 'shape': [], 'out': [74, 70], 'sorted_id': 69}
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
        
        // {'name': 'Net/120', 'op': 'aten::div', 'in': [41, 70], 'output_id': 0, 'shape': [32, 2], 'out': [73], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/121', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [73], 'sorted_id': 72}
        {
            Tensor c = (float)2.0;
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
        
        // {'name': 'Net/123', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [76], 'sorted_id': 75}
        {
            Tensor c = (float)1.0;
            forward_result[75] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/124', 'op': 'aten::sub', 'in': [16, 74, 75], 'output_id': 0, 'shape': [32, 2], 'out': [77], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
        }
        
        // {'name': 'Net/125', 'op': 'aten::div', 'in': [76, 70], 'output_id': 0, 'shape': [32, 2], 'out': [79], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/126', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [79], 'sorted_id': 78}
        {
            Tensor c = (float)2.0;
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
        
        // {'name': 'Net/128', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [81], 'sorted_id': 80}
        {
            Tensor c = (float)1.0;
            forward_result[80] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/129', 'op': 'aten::add', 'in': [73, 79, 80], 'output_id': 0, 'shape': [32, 2], 'out': [84], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/130', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [84], 'sorted_id': 82}
        {
            Tensor c = (float)1.0;
            forward_result[82] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/131', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [84], 'sorted_id': 83}
        {
            Tensor c = (float)1.0;
            forward_result[83] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/132', 'op': 'aten::sub', 'in': [81, 82, 83], 'output_id': 0, 'shape': [32, 2], 'out': [87], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[81] );
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[83] );
        }
        
        // {'name': 'Net/133', 'op': 'aten::log', 'in': [73], 'output_id': 0, 'shape': [32, 2], 'out': [87], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {32,2};
            LogOp* op = new LogOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Net/134', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [87], 'sorted_id': 86}
        {
            Tensor c = (float)1.0;
            forward_result[86] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/135', 'op': 'aten::sub', 'in': [84, 85, 86], 'output_id': 0, 'shape': [32, 2], 'out': [89], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[86] );
        }
        
        // {'name': 'Net/136', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [89], 'sorted_id': 88}
        {
            Tensor c = (float)0.5;
            forward_result[88] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/137', 'op': 'aten::mul', 'in': [87, 88], 'output_id': 0, 'shape': [32, 2], 'out': [91], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'Net/138', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [91], 'sorted_id': 90}
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
        
        // {'name': 'Net/140', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [93], 'sorted_id': 92}
        {
            Tensor c = (float)1.0;
            forward_result[92] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/141', 'op': 'aten::add', 'in': [55, 91, 92], 'output_id': 0, 'shape': [], 'out': [94], 'sorted_id': 93}
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
        */
/*
        cout<<"### forward computation ..."<<endl;
        //forward_result[93]->forward();
        for(int k=0;k<=93;k++) {
          if( forward_result[k] ){
            forward_result[k]->set_id( k );
            forward_result[k]->forward();
            forward_result[k]->zerograd();
          }
        }
        auto o = forward_result[93]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[93]->grad=xt::ones_like(forward_result[93]->output);
        //forward_result[93]->backward();
        for(int k=93;k>=0;k--) {
          if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
*/

    // all input data
    extern Tensor indata;

    void do_train_loop( vector<MCTNode*> &forward_result, VariableTensor &input_var, int NL, fprec lr )
    {

        auto do_forward=[]( vector<MCTNode*> &op, int n ) 
        {
            //cout<<"### forward computation ..."<<endl;
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->forward();
            }
        };
        auto do_backward=[]( vector<MCTNode*> &op, int n ) 
        {
            //cout<<"### backward computation ..."<<endl;
            op[n]->grad = xt::ones_like( op[n]->output );
            for(int k=n;k>=0;k--) {
              if( op[k] )  op[k]->backward();
            }
        };
        auto do_zerograd=[]( vector<MCTNode*> &op, int n ) 
        {
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->zerograd();
            }
        };
        auto update_params=[]( vector<MCTNode*> &op, int n, fprec lr=0.01 ) 
        {
            for(int k=0;k<=n;k++) {
              if( op[k] )  op[k]->update( lr );
            }
        };
        
        string project = "vae2";
        
        //fprec lr = 0.01;
        //int   NL = 93;
        
        // all input data
        extern Tensor indata;
        
        indata.reshape({1797,64});
        auto indata_s = indata.shape();
        
        int batch_size = 32;
        int n_batch = (int)indata_s[0] / batch_size;
        cout<<"indata "<<indata_s[0]<<","<<indata_s[1]<<endl;
        cout<<"batch  "<<n_batch<<","<<batch_size<<endl;
        
        xt::random::seed(1);
        
        int epoch_num = 200;
        cout<<"epoch_num : "<<epoch_num<<endl;
        
        Tensor x_pred = xt::zeros<fprec>( { batch_size, (int)indata_s[1] } );

        string flname = project + ".out";
        ofstream outputfile( flname );
        for(int epoch=0;epoch<epoch_num;epoch++)
        {
            Tensor index = xt::arange( (int)indata_s[0] );
            xt::random::shuffle( index );
           
            //forward_result[ 8]->set_output1( (fprec)batch_size );
            //forward_result[53]->set_output1( (fprec)batch_size );
            //forward_result[69]->set_output1( (fprec)batch_size );
    
            fprec total_loss = 0.0;
            //n_batch = 2;
            for(int jj=0;jj<n_batch;jj++)
            {
                int j1 = jj*batch_size;
                for(int k=0;k<batch_size;k++)
                {
                    xt::row(x_pred,k) = xt::flatten( xt::row(indata,index(j1+k)) );
                }
                
                input_var.output = x_pred;
            
                do_forward( forward_result, NL );
                
                auto o = forward_result[NL]->output;
                //auto o1 = forward_result[55]->output;
                //auto o2 = forward_result[91]->output;
                //cout<<"epoch "<<epoch<<","<<jj<<" - loss "<<o[0]<<","<<o1<<","<<o2<<endl;
                //outputfile<<to_string(o[0])<<endl;
                
                total_loss += o[0];
        
                do_backward( forward_result, NL );
                update_params( forward_result, NL, lr );
                do_zerograd( forward_result, NL );
            }
            cout<<"total_loss "<<epoch<<" loss-"<<total_loss<<endl;
            
            
            input_var.output = indata;
            //forward_result[ 8]->set_output1( (fprec)indata_s[0] );  // randn
            //forward_result[53]->set_output1( (fprec)indata_s[0] );  // div
            //forward_result[69]->set_output1( (fprec)indata_s[0] );  // div
            
            do_forward( forward_result, NL );
            auto o  = forward_result[NL]->output;
            auto o1 = forward_result[55]->output;
            auto o2 = forward_result[91]->output;
            cout<<"epoch "<<epoch<<" - loss "<<o<<" ( "<<o1<<" , "<<o2<<" ) "<<endl;
            outputfile<<to_string(o[0])<<endl;
            
        }
        outputfile.close();
        
        {
            int n_img = 10;
            
            // 52: sigmoid outpout
            auto y_pred = forward_result[52]->get_output();
           
            flname = project + ".pred";
            ofstream outputfile( flname );
            outputfile<<to_string(n_img)<<","<<to_string(indata_s[1])<<endl;
        
            for(int k=0;k<n_img;k++)
            {
                for(int i=0;i<indata_s[1]-1;i++)
                {
                    outputfile<<to_string(y_pred(k,i))<<",";
                }
                outputfile<<to_string(y_pred(k,indata_s[1]-1))<<endl;
            }
            outputfile.close();
        }
        {
            // 44: z output
            auto z_pred = forward_result[44]->get_output();
            
            flname = project + ".z";
            ofstream outputfile( flname );
            outputfile<<to_string(indata_s[0])<<","<<to_string(2)<<endl;
            
            for(int k=0;k<indata_s[0];k++)
            {
                outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
            }
            outputfile.close();
        }
    }
    