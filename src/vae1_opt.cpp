
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
        vector<MCTNode*> forward_result(75);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [32, 64], 'out': [3, 40, 37], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/135', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor(fc1_weight);
        }
        
        // {'name': 'Net/Linear[fc1]/bias/134', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
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
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'shape': [32, 16], 'out': [18, 7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/138', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor(fc2_mean_weight);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/137', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            forward_result[6] = new VariableTensor(fc2_mean_bias);
        }
        
        // {'name': 'Net/Linear[fc2_mean]/mean', 'op': 'aten::linear', 'in': [4, 5, 6], 'shape': [32, 2], 'out': [24, 59], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/33', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 32.0, 'out': [10], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)32.0;
            forward_result[8] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/34', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 2.0, 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)2.0;
            forward_result[9] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/35', 'op': 'prim::ListConstruct', 'in': [8, 9], 'shape': [], 'out': [15], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {};
            ListConstructOp* op = new ListConstructOp();
            forward_result[10] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'Net/36', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 6.0, 'out': [15], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)6.0;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/37', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [15], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {};
            forward_result[12] = NULL;
        }
        
        // {'name': 'Net/38', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [15], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {};
            forward_result[13] = NULL;
        }
        
        // {'name': 'Net/39', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.0, 'out': [15], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)0.0;
            forward_result[14] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/eps', 'op': 'aten::randn', 'in': [10, 11, 12, 13, 14], 'shape': [32, 2], 'out': [22], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {32,2};
            RandnOp* op = new RandnOp();
            forward_result[15] = op;
            
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[14] );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/141', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [18], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[16] = new VariableTensor(fc2_var_weight);
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/140', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [18], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {2};
            forward_result[17] = new VariableTensor(fc2_var_bias);
        }
        
        // {'name': 'Net/Linear[fc2_var]/log_var', 'op': 'aten::linear', 'in': [4, 16, 17], 'shape': [32, 2], 'out': [20, 62, 57], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
        }
        
        // {'name': 'Net/41', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [20], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)0.5;
            forward_result[19] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/42', 'op': 'aten::mul', 'in': [18, 19], 'shape': [32, 2], 'out': [21], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/43', 'op': 'aten::exp', 'in': [20], 'shape': [32, 2], 'out': [22], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/44', 'op': 'aten::mul', 'in': [15, 21], 'shape': [32, 2], 'out': [24], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[21] );
        }
        
        // {'name': 'Net/45', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [24], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[23] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/input.5', 'op': 'aten::add', 'in': [7, 22, 23], 'shape': [32, 2], 'out': [27], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/144', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [27], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[25] = new VariableTensor(fc3_weight);
        }
        
        // {'name': 'Net/Linear[fc3]/bias/143', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [27], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {16};
            forward_result[26] = new VariableTensor(fc3_bias);
        }
        
        // {'name': 'Net/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [24, 25, 26], 'shape': [32, 16], 'out': [28], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [27], 'shape': [32, 16], 'out': [31], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/147', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [31], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[29] = new VariableTensor(fc4_weight);
        }
        
        // {'name': 'Net/Linear[fc4]/bias/146', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [31], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {64};
            forward_result[30] = new VariableTensor(fc4_bias);
        }
        
        // {'name': 'Net/Linear[fc4]/148', 'op': 'aten::linear', 'in': [28, 29, 30], 'shape': [32, 64], 'out': [32], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[31] = op;
            
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[30] );
        }
        
        // {'name': 'Net/y', 'op': 'aten::sigmoid', 'in': [31], 'shape': [32, 64], 'out': [43, 35], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'Net/51', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1e-07, 'out': [35], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1e-07;
            forward_result[33] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/52', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [35], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[34] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/53', 'op': 'aten::add', 'in': [32, 33, 34], 'shape': [32, 64], 'out': [36], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'Net/54', 'op': 'aten::log', 'in': [35], 'shape': [32, 64], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[35] );
        }
        
        // {'name': 'Net/55', 'op': 'aten::mul', 'in': [0, 36], 'shape': [32, 64], 'out': [50], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'Net/56', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [40], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[38] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/57', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [40], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[39] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/58', 'op': 'aten::rsub', 'in': [0, 38, 39], 'shape': [32, 64], 'out': [48], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Net/59', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [43], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[41] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/60', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[42] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/61', 'op': 'aten::rsub', 'in': [32, 41, 42], 'shape': [32, 64], 'out': [46], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/62', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1e-07, 'out': [46], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1e-07;
            forward_result[44] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/63', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [46], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[45] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/64', 'op': 'aten::add', 'in': [43, 44, 45], 'shape': [32, 64], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Net/65', 'op': 'aten::log', 'in': [46], 'shape': [32, 64], 'out': [48], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/66', 'op': 'aten::mul', 'in': [40, 47], 'shape': [32, 64], 'out': [50], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[48] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[47] );
        }
        
        // {'name': 'Net/67', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [50], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[49] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e', 'op': 'aten::add', 'in': [37, 48, 49], 'shape': [32, 64], 'out': [52], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Net/69', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {};
            forward_result[51] = NULL;
        }
        
        // {'name': 'Net/70', 'op': 'aten::sum', 'in': [50, 51], 'shape': [], 'out': [54], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {};
            SumOp*    op = new SumOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Net/77', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 32.0, 'out': [54], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)32.0;
            forward_result[53] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e1', 'op': 'aten::div', 'in': [52, 53], 'shape': [], 'out': [72], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {};
            DivOp* op = new DivOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/79', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [57], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[55] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/80', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [57], 'sorted_id': 56}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/81', 'op': 'aten::add', 'in': [18, 55, 56], 'shape': [32, 2], 'out': [61], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/82', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 2.0, 'out': [59], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)2.0;
            forward_result[58] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/83', 'op': 'aten::pow', 'in': [7, 58], 'shape': [32, 2], 'out': [61], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Net/84', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[60] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/85', 'op': 'aten::sub', 'in': [57, 59, 60], 'shape': [32, 2], 'out': [64], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[60] );
        }
        
        // {'name': 'Net/86', 'op': 'aten::exp', 'in': [18], 'shape': [32, 2], 'out': [64], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'Net/87', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [64], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[63] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/g', 'op': 'aten::sub', 'in': [61, 62, 63], 'shape': [32, 2], 'out': [66], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'Net/89', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {};
            forward_result[65] = NULL;
        }
        
        // {'name': 'Net/90', 'op': 'aten::sum', 'in': [64, 65], 'shape': [], 'out': [68], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {};
            SumOp*    op = new SumOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
        }
        
        // {'name': 'Net/91', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 0.5, 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)0.5;
            forward_result[67] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/92', 'op': 'aten::mul', 'in': [66, 67], 'shape': [], 'out': [70], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {};
            MulOp* op = new MulOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'Net/99', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 32.0, 'out': [70], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)32.0;
            forward_result[69] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e2', 'op': 'aten::div', 'in': [68, 69], 'shape': [], 'out': [72], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {};
            DivOp* op = new DivOp();
            forward_result[70] = op;
            
            op->set_inputs( forward_result[68] );
            op->set_inputs( forward_result[69] );
        }
        
        // {'name': 'Net/101', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [72], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {};
            Tensor c = (fprec)1.0;
            forward_result[71] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/102', 'op': 'aten::add', 'in': [54, 70, 71], 'shape': [], 'out': [73], 'sorted_id': 72}
        {
            Tensor::shape_type shape = {};
            AddOp* op = new AddOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[71] );
        }
        
        // {'name': 'Net/103', 'op': 'aten::neg', 'in': [72], 'shape': [], 'out': [74], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {};
            NegOp* op = new NegOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [73], 'shape': [], 'out': [], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {};
        }
        
    /* 210905 mod
        cout<<"### forward computation ..."<<endl;
        //forward_result[73]->forward();
        for(int k=0;k<=73;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[73]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[73]->grad = xt::ones_like( forward_result[73]->output );
        //forward_result[73]->backward();
        for(int k=73;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    */
    
    // optimization  210905 add
    auto exec_forward=[]( vector<MCTNode*> &op, int n, int out ) 
    {
        //cout<<"### forward computation ..."<<endl;
        //op[n]->forward();
        for(int k=0;k<=n;k++) {
          if( out> 0 )  cout<<"id - "<<k<<endl;
          if( op[k] )  op[k]->forward();
        }
    };
    auto exec_backward=[]( vector<MCTNode*> &op, int n ) 
    {
        //cout<<"### backward computation ..."<<endl;
        op[n]->grad = xt::ones_like( op[n]->output );
        //op[n]->backward();
        for(int k=n;k>=0;k--) {
          if( op[k] )  op[k]->backward();
        }
    };
    auto exec_zerograd=[]( vector<MCTNode*> &op, int n ) 
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
    
    fprec lr = 0.01;
    int   NL = 73;
    
    // all input data
    extern Tensor indata;
    
    indata.reshape({1797,64});
    auto indata_s = indata.shape();
    
    int batch_size = 32;
    int n_batch = (int)indata_s[0] / batch_size;
    cout<<"batch1 "<<indata_s[0]<<","<<indata_s[1]<<endl;
    cout<<"batch2 "<<n_batch<<","<<batch_size<<endl;
    
    xt::random::seed(1);
    
    int epoch_num = 200;
    cout<<"epoch_num : "<<epoch_num<<endl;
    
    Tensor x_pred = xt::zeros<fprec>( { batch_size, (int)indata_s[1] } );
    
    
    ofstream outputfile("vae1.out");
    for(int epoch=0;epoch<epoch_num;epoch++)
    {
        Tensor index = xt::arange( (int)indata_s[0] );
        xt::random::shuffle( index );
        
        forward_result[ 8]->set_output1( (fprec)batch_size );
        forward_result[53]->set_output1( (fprec)batch_size );
        forward_result[69]->set_output1( (fprec)batch_size );
        
        //Tensor x_pred=xt::zeros<fprec>( { batch_size, (int)indata_s[1] } );

        fprec total_loss = 0.0;
        for(int jj=0;jj<n_batch;jj++)
        {
            int j1 = jj*batch_size;
            //int j2 = (jj+1)*batch_size;
            
            //cout<<"index";
            //for(int jj=j1;jj<j2;jj++)  cout<<index(jj)<<",";
            //cout<<endl;
            
            /*int k=0;
            for(int k=0;k<batch_size;k++) {
                int j = index(j1+k);
                for(int i=0;i<(int)indata_s[1];i++)  x_pred(k,i) = indata(j,i);
            }*/
            for(int k=0;k<batch_size;k++)
            {
                xt::row(x_pred,k) = xt::flatten( xt::row(indata,index(j1+k)) );
            }
            
            input_var.output = x_pred;
        
            exec_forward( forward_result, NL, 0 );
            
            auto o = forward_result[NL]->output;
            //cout<<"epoch "<<epoch<<","<<jj<<" - loss "<<o<<endl;
            //outputfile<<to_string(o[0])<<endl;
            
            total_loss += o[0];
    
            exec_backward( forward_result, NL );
            update_params( forward_result, NL, lr );
            exec_zerograd( forward_result, NL );
        }
        cout<<"total_loss "<<epoch<<" loss - "<<total_loss<<endl;
        
        
        input_var.output = indata;
        forward_result[ 8]->set_output1( (fprec)indata_s[0] );
        forward_result[53]->set_output1( (fprec)indata_s[0] );
        forward_result[69]->set_output1( (fprec)indata_s[0] );
        
        
        exec_forward( forward_result, NL, 0 );
        auto o = forward_result[NL]->output;
        cout<<"epoch "<<epoch<<" - loss "<<o<<endl;
        outputfile<<to_string(o[0])<<endl;
        
    }
    outputfile.close();
    
    {
        int n_img = 10;
        
        // 32: sigmoid outpout
        auto y_pred = forward_result[32]->get_output();
        //cout<<"y_pred"<<y_pred<<endl;
       
        ofstream outputfile("vae1.pred");
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
        // 24: z output
        auto z_pred = forward_result[24]->get_output();
        
        ofstream outputfile("vae1.z");
        outputfile<<to_string(indata_s[0])<<","<<to_string(2)<<endl;
        
        for(int k=0;k<indata_s[0];k++)
        {
            outputfile<<to_string(z_pred(k,0))<<","<<to_string(z_pred(k,1))<<endl;
        }
        outputfile.close();
    }
    
    
        return 0;
    }
    