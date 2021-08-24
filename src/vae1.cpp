
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
    extern Tensor  Constant1;
    extern Tensor  fc2_var_weight;
    extern Tensor  fc2_var_bias;
    extern Tensor  fc3_weight;
    extern Tensor  fc3_bias;
    extern Tensor  fc4_weight;
    extern Tensor  fc4_bias;
    
    int main()
    {
        vector<MCTNode*> forward_result(68);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [32, 64], 'out': [3, 30, 33], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/128', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/127', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
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
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [32, 16], 'out': [7, 11], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/131', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_mean_weight);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/130', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            forward_result[6] = new VariableTensor(fc2_mean_bias);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/mean', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [32, 2], 'out': [17, 52], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::Constant', 'in': [], 'shape': [32, 2], 'constant_value': [-1.5256, -0.7502, -0.654, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091, -0.7121, 0.3037, -0.7773, -0.2515, -0.2223, 1.6871, 0.2284, 0.4676, -0.697, -1.1608, 0.6995, 0.1991, 0.8657, 0.2444, -0.6629, 0.8073, 1.1017, -0.1759, -2.2456, -1.4465, 0.0612, -0.6177, -0.7981, -0.1316, 1.8793, -0.0721, 0.1578, -0.7735, 0.1991, 0.0457, 0.153, -0.4757, -0.111, 0.2927, -0.1578, -0.0288, 2.3571, -1.0373, 1.5748, -0.6298, -0.9274, 0.5451, 0.0663, -0.437, 0.7626, 0.4415, 1.1651, 2.0154, 0.1374, 0.9386, -0.186, -0.6446, 1.5392, -0.8696, -3.3312, -0.7479], 'out': [15], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {32,2};
            Constant1.reshape( shape );
            forward_result[8] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/134', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[9] = new VariableTensor(fc2_var_weight);
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/133', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {2};
            forward_result[10] = new VariableTensor(fc2_var_bias);
        }
        
        // {'name': 'Net/Linear[fc2_var]/log_var', 'op': 'aten::linear', 'in': [4, 9, 10], 'shape': [32, 2], 'out': [55, 13, 50], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [13], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)0.5;
            forward_result[12] = new VariableTensor( c );
        }
        
        // {'name': 'Net/34', 'op': 'aten::mul', 'in': [11, 12], 'shape': [32, 2], 'out': [14], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/35', 'op': 'aten::exp', 'in': [13], 'shape': [32, 2], 'out': [15], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/37', 'op': 'aten::mul', 'in': [8, 14], 'shape': [32, 2], 'out': [17], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[15] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/38', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [17], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[16] = new VariableTensor( c );
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [7, 15, 16], 'shape': [32, 2], 'out': [20], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[17] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/137', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [20], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[18] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/136', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [20], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {16};
            forward_result[19] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [17, 18, 19], 'shape': [32, 16], 'out': [21], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [20], 'shape': [32, 16], 'out': [24], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/140', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[22] = new VariableTensor(fc4_weight);
        }
        
        // {'name': 'Net/Linear[fc4]/bias/139', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [24], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {64};
            forward_result[23] = new VariableTensor(fc4_bias);
        }
        
        // {'name': 'Net/Linear[fc4]/141', 'op': 'aten::linear', 'in': [21, 22, 23], 'shape': [32, 64], 'out': [25], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/y', 'op': 'aten::sigmoid', 'in': [24], 'shape': [32, 64], 'out': [28, 36], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/44', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1e-07, 'out': [28], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1e-07;
            forward_result[26] = new VariableTensor( c );
        }
        
        // {'name': 'Net/45', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [28], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[27] = new VariableTensor( c );
        }
        
        // {'name': 'Net/46', 'op': 'aten::add', 'in': [25, 26, 27], 'shape': [32, 64], 'out': [29], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'Net/47', 'op': 'aten::log', 'in': [28], 'shape': [32, 64], 'out': [30], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Net/48', 'op': 'aten::mul', 'in': [0, 29], 'shape': [32, 64], 'out': [43], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/49', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [33], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[31] = new VariableTensor( c );
        }
        
        // {'name': 'Net/50', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [33], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[32] = new VariableTensor( c );
        }
        
        // {'name': 'Net/51', 'op': 'aten::rsub', 'in': [0, 31, 32], 'shape': [32, 64], 'out': [41], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/52', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [36], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[34] = new VariableTensor( c );
        }
        
        // {'name': 'Net/53', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [36], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[35] = new VariableTensor( c );
        }
        
        // {'name': 'Net/54', 'op': 'aten::rsub', 'in': [25, 34, 35], 'shape': [32, 64], 'out': [39], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[35] );
        }
        
        // {'name': 'Net/55', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1e-07, 'out': [39], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1e-07;
            forward_result[37] = new VariableTensor( c );
        }
        
        // {'name': 'Net/56', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[38] = new VariableTensor( c );
        }
        
        // {'name': 'Net/57', 'op': 'aten::add', 'in': [36, 37, 38], 'shape': [32, 64], 'out': [40], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'Net/58', 'op': 'aten::log', 'in': [39], 'shape': [32, 64], 'out': [41], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Net/59', 'op': 'aten::mul', 'in': [33, 40], 'shape': [32, 64], 'out': [43], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[41] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[40] );
        }
        
        // {'name': 'Net/60', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[42] = new VariableTensor( c );
        }
        
        // {'name': 'Net/e', 'op': 'aten::add', 'in': [30, 41, 42], 'shape': [32, 64], 'out': [45], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/62', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [45], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {};
            forward_result[44] = NULL;
        }
        
        // {'name': 'Net/63', 'op': 'aten::sum', 'in': [43, 44], 'shape': [], 'out': [47], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {};
            SumOp*    op = new SumOp();
            forward_result[45] = op;
            
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[44] );
        }
        
        // {'name': 'Net/70', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 32.0, 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)32.0;
            forward_result[46] = new VariableTensor( c );
        }
        
        // {'name': 'Net/e1', 'op': 'aten::div', 'in': [45, 46], 'shape': [], 'out': [65], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {};
            DivOp* op = new DivOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/72', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [50], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[48] = new VariableTensor( c );
        }
        
        // {'name': 'Net/73', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [50], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[49] = new VariableTensor( c );
        }
        
        // {'name': 'Net/74', 'op': 'aten::add', 'in': [11, 48, 49], 'shape': [32, 2], 'out': [54], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Net/75', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 2.0, 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)2.0;
            forward_result[51] = new VariableTensor( c );
        }
        
        // {'name': 'Net/76', 'op': 'aten::pow', 'in': [7, 51], 'shape': [32, 2], 'out': [54], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Net/77', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [54], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[53] = new VariableTensor( c );
        }
        
        // {'name': 'Net/78', 'op': 'aten::sub', 'in': [50, 52, 53], 'shape': [32, 2], 'out': [57], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/79', 'op': 'aten::exp', 'in': [11], 'shape': [32, 2], 'out': [57], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[11] );
        }
        
        // {'name': 'Net/80', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [57], 'sorted_id': 56}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[56] = new VariableTensor( c );
        }
        
        // {'name': 'Net/g', 'op': 'aten::sub', 'in': [54, 55, 56], 'shape': [32, 2], 'out': [59], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/82', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [59], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {};
            forward_result[58] = NULL;
        }
        
        // {'name': 'Net/83', 'op': 'aten::sum', 'in': [57, 58], 'shape': [], 'out': [61], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {};
            SumOp*    op = new SumOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Net/84', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)0.5;
            forward_result[60] = new VariableTensor( c );
        }
        
        // {'name': 'Net/85', 'op': 'aten::mul', 'in': [59, 60], 'shape': [], 'out': [63], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {};
            MulOp* op = new MulOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[60] );
        }
        
        // {'name': 'Net/92', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 32.0, 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)32.0;
            forward_result[62] = new VariableTensor( c );
        }
        
        // {'name': 'Net/e2', 'op': 'aten::div', 'in': [61, 62], 'shape': [], 'out': [65], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {};
            DivOp* op = new DivOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'Net/94', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [65], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[64] = new VariableTensor( c );
        }
        
        // {'name': 'Net/95', 'op': 'aten::add', 'in': [47, 63, 64], 'shape': [], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {};
            AddOp* op = new AddOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'Net/96', 'op': 'aten::neg', 'in': [65], 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {};
            NegOp* op = new NegOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[65] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [66], 'shape': [], 'out': [], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[66]->forward();
        for(int k=0;k<=66;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[66]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[66]->grad = xt::ones_like( forward_result[66]->output );
        //forward_result[66]->backward();
        for(int k=66;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    