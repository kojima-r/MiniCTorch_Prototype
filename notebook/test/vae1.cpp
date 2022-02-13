
    //
    //  vae1
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
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [43, 46, 3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/weight.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight, 2 );
        }
        
        // {'name': 'Net/Linear[fc1]/bias/bias.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {16};
            fc1_bias.reshape( shape );
            forward_result[2] = new VariableTensor( fc1_bias, 2 );
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
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [32, 16], 'out': [24, 7], 'sorted_id': 4}
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
            forward_result[5] = new VariableTensor( fc2_mean_weight, 2 );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/bias.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            fc2_mean_bias.reshape( shape );
            forward_result[6] = new VariableTensor( fc2_mean_bias, 2 );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/130', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [32, 2], 'out': [30, 13, 9, 64], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [9], 'sorted_id': 8}
        {
            Tensor c = (fprec)0.0;
            forward_result[8] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/34', 'op': 'aten::size', 'in': [7, 8], 'output_id': 0, 'shape': [], 'out': [10], 'sorted_id': 9}
        {
            SizeOp* op = new SizeOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
        }
        
        // {'name': 'Net/35', 'op': 'prim::NumToTensor', 'in': [9], 'output_id': 0, 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[10] = op;
            
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'Net/39', 'op': 'aten::Int', 'in': [10], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 11}
        {
            IntOp* op = new IntOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [13], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/37', 'op': 'aten::size', 'in': [7, 12], 'output_id': 0, 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            SizeOp* op = new SizeOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/38', 'op': 'prim::NumToTensor', 'in': [13], 'output_id': 0, 'shape': [], 'out': [15], 'sorted_id': 14}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/40', 'op': 'aten::Int', 'in': [14], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 15}
        {
            IntOp* op = new IntOp();
            forward_result[15] = op;
            
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/41', 'op': 'prim::ListConstruct', 'in': [11, 15], 'output_id': 0, 'shape': [], 'out': [21], 'sorted_id': 16}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/42', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [21], 'sorted_id': 17}
        {
            Tensor c = (fprec)6.0;
            forward_result[17] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/43', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [21], 'sorted_id': 18}
        {
            forward_result[18] = NULL;
        }
        
        // {'name': 'Net/44', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [21], 'sorted_id': 19}
        {
            forward_result[19] = NULL;
        }
        
        // {'name': 'Net/45', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [21], 'sorted_id': 20}
        {
            Tensor c = (fprec)0.0;
            forward_result[20] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/eps', 'op': 'aten::randn', 'in': [16, 17, 18, 19, 20], 'output_id': 0, 'shape': [32, 2], 'out': [28], 'sorted_id': 21}
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
        
        // {'name': 'Net/Linear[fc2_var]/weight/weight.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [24], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[22] = new VariableTensor( fc2_var_weight, 2 );
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/bias.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [24], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {2};
            fc2_var_bias.reshape( shape );
            forward_result[23] = new VariableTensor( fc2_var_bias, 2 );
        }
        
        // {'name': 'Net/Linear[fc2_var]/log_var', 'op': 'aten::linear', 'in': [4, 22, 23], 'output_id': 0, 'shape': [32, 2], 'out': [62, 67, 26], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/47', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [26], 'sorted_id': 25}
        {
            Tensor c = (fprec)0.5;
            forward_result[25] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/48', 'op': 'aten::mul', 'in': [24, 25], 'output_id': 0, 'shape': [32, 2], 'out': [27], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/49', 'op': 'aten::exp', 'in': [26], 'output_id': 0, 'shape': [32, 2], 'out': [28], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'Net/50', 'op': 'aten::mul', 'in': [21, 27], 'output_id': 0, 'shape': [32, 2], 'out': [30], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'Net/51', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [30], 'sorted_id': 29}
        {
            Tensor c = (fprec)1.0;
            forward_result[29] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [7, 28, 29], 'output_id': 0, 'shape': [32, 2], 'out': [33], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/weight.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[31] = new VariableTensor( fc3_weight, 2 );
        }
        
        // {'name': 'Net/Linear[fc3]/bias/bias.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {16};
            fc3_bias.reshape( shape );
            forward_result[32] = new VariableTensor( fc3_bias, 2 );
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [30, 31, 32], 'output_id': 0, 'shape': [32, 16], 'out': [34], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [33], 'output_id': 0, 'shape': [32, 16], 'out': [37], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[33] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/weight', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [37], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[35] = new VariableTensor( fc4_weight, 2 );
        }
        
        // {'name': 'Net/Linear[fc4]/bias/bias', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {64};
            fc4_bias.reshape( shape );
            forward_result[36] = new VariableTensor( fc4_bias, 2 );
        }
        
        // {'name': 'Net/Linear[fc4]/139', 'op': 'aten::linear', 'in': [34, 35, 36], 'output_id': 0, 'shape': [32, 64], 'out': [38], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'Net/y.1', 'op': 'aten::sigmoid', 'in': [37], 'output_id': 0, 'shape': [32, 64], 'out': [49, 41], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/57', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-07, 'out': [41], 'sorted_id': 39}
        {
            Tensor c = (fprec)1e-07;
            forward_result[39] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/58', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [41], 'sorted_id': 40}
        {
            Tensor c = (fprec)1.0;
            forward_result[40] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/59', 'op': 'aten::add', 'in': [38, 39, 40], 'output_id': 0, 'shape': [32, 64], 'out': [42], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[41] = op;
            
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[40] );
        }
        
        // {'name': 'Net/60', 'op': 'aten::log', 'in': [41], 'output_id': 0, 'shape': [32, 64], 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[41] );
        }
        
        // {'name': 'Net/61', 'op': 'aten::mul', 'in': [0, 42], 'output_id': 0, 'shape': [32, 64], 'out': [56], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/62', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [46], 'sorted_id': 44}
        {
            Tensor c = (fprec)1.0;
            forward_result[44] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/63', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [46], 'sorted_id': 45}
        {
            Tensor c = (fprec)1.0;
            forward_result[45] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/64', 'op': 'aten::rsub', 'in': [0, 44, 45], 'output_id': 0, 'shape': [32, 64], 'out': [54], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Net/65', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [49], 'sorted_id': 47}
        {
            Tensor c = (fprec)1.0;
            forward_result[47] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/66', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [49], 'sorted_id': 48}
        {
            Tensor c = (fprec)1.0;
            forward_result[48] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/67', 'op': 'aten::rsub', 'in': [38, 47, 48], 'output_id': 0, 'shape': [32, 64], 'out': [52], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/68', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-07, 'out': [52], 'sorted_id': 50}
        {
            Tensor c = (fprec)1e-07;
            forward_result[50] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/69', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [52], 'sorted_id': 51}
        {
            Tensor c = (fprec)1.0;
            forward_result[51] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/70', 'op': 'aten::add', 'in': [49, 50, 51], 'output_id': 0, 'shape': [32, 64], 'out': [53], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[49] );
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Net/71', 'op': 'aten::log', 'in': [52], 'output_id': 0, 'shape': [32, 64], 'out': [54], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[53] = op;
            
            op->set_inputs( forward_result[52] );
        }
        
        // {'name': 'Net/72', 'op': 'aten::mul', 'in': [46, 53], 'output_id': 0, 'shape': [32, 64], 'out': [56], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/73', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [56], 'sorted_id': 55}
        {
            Tensor c = (fprec)1.0;
            forward_result[55] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/e', 'op': 'aten::add', 'in': [43, 54, 55], 'output_id': 0, 'shape': [32, 64], 'out': [58], 'sorted_id': 56}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[56] = op;
            
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[55] );
        }
        
        // {'name': 'Net/75', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [58], 'sorted_id': 57}
        {
            forward_result[57] = NULL;
        }
        
        // {'name': 'Net/z', 'op': 'aten::sum', 'in': [56, 57], 'output_id': 0, 'shape': [], 'out': [59], 'sorted_id': 58}
        {
            SumOp* op = new SumOp();
            forward_result[58] = op;
            
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[57] );
        }
        
        // {'name': 'Net/e1', 'op': 'aten::neg', 'in': [58], 'output_id': 0, 'shape': [], 'out': [76], 'sorted_id': 59}
        {
            NegOp* op = new NegOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Net/78', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [62], 'sorted_id': 60}
        {
            Tensor c = (fprec)1.0;
            forward_result[60] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/79', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [62], 'sorted_id': 61}
        {
            Tensor c = (fprec)1.0;
            forward_result[61] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/80', 'op': 'aten::add', 'in': [24, 60, 61], 'output_id': 0, 'shape': [32, 2], 'out': [66], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/81', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [64], 'sorted_id': 63}
        {
            Tensor c = (fprec)2.0;
            forward_result[63] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/82', 'op': 'aten::pow', 'in': [7, 63], 'output_id': 0, 'shape': [32, 2], 'out': [66], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'Net/83', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [66], 'sorted_id': 65}
        {
            Tensor c = (fprec)1.0;
            forward_result[65] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/84', 'op': 'aten::sub', 'in': [62, 64, 65], 'output_id': 0, 'shape': [32, 2], 'out': [69], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
        }
        
        // {'name': 'Net/85', 'op': 'aten::exp', 'in': [24], 'output_id': 0, 'shape': [32, 2], 'out': [69], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/86', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [69], 'sorted_id': 68}
        {
            Tensor c = (fprec)1.0;
            forward_result[68] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/g', 'op': 'aten::sub', 'in': [66, 67, 68], 'output_id': 0, 'shape': [32, 2], 'out': [71], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[67] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'Net/88', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [71], 'sorted_id': 70}
        {
            forward_result[70] = NULL;
        }
        
        // {'name': 'Net/89', 'op': 'aten::sum', 'in': [69, 70], 'output_id': 0, 'shape': [], 'out': [73], 'sorted_id': 71}
        {
            SumOp* op = new SumOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/90', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [73], 'sorted_id': 72}
        {
            Tensor c = (fprec)0.5;
            forward_result[72] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/y', 'op': 'aten::mul', 'in': [71, 72], 'output_id': 0, 'shape': [], 'out': [74], 'sorted_id': 73}
        {
            MulOp* op = new MulOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/e2', 'op': 'aten::neg', 'in': [73], 'output_id': 0, 'shape': [], 'out': [76], 'sorted_id': 74}
        {
            NegOp* op = new NegOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Net/93', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [76], 'sorted_id': 75}
        {
            Tensor c = (fprec)1.0;
            forward_result[75] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Net/94', 'op': 'aten::add', 'in': [59, 74, 75], 'output_id': 0, 'shape': [], 'out': [77], 'sorted_id': 76}
        {
            AddOp* op = new AddOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [76], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 77}
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
        vector<MCTNode*> forward_result(78);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var( xin, 3 );
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 76 );
    #else
        do_train1( forward_result, input_var, 76 );
    #endif
        
        return 0;
    }
    