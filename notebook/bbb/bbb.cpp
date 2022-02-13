
    //
    //  bbb
    //
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include "minictorch.hpp"
    
    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  l1_weight_mu;
    extern Tensor  l1_weight_rho;
    extern Tensor  l1_bias_mu;
    extern Tensor  l1_bias_rho;
    extern Tensor  Constant1;
    extern Tensor  Constant2;
    extern Tensor  Constant3;
    extern Tensor  l2_weight_mu;
    extern Tensor  l2_weight_rho;
    extern Tensor  l2_bias_mu;
    extern Tensor  l2_bias_rho;
    extern Tensor  l3_weight_mu;
    extern Tensor  l3_weight_rho;
    extern Tensor  l3_bias_mu;
    extern Tensor  l3_bias_rho;
    extern Tensor  Constant4;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [4, 1, 28, 28], 'out': [4], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {4,1,28,28};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Model/Net[net]/961', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -1.0, 'out': [3], 'sorted_id': 1}
        {
            Tensor c = (fprec)-1.0;
            forward_result[1] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/960', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 784.0, 'out': [3], 'sorted_id': 2}
        {
            Tensor c = (fprec)784.0;
            forward_result[2] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/965', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [4], 'sorted_id': 3}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Model/Net[net]/input.1', 'op': 'aten::view', 'in': [0, 3], 'output_id': 0, 'shape': [4, 784], 'out': [40], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_mu/weight_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [26, 104], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[5] = new VariableTensor( l1_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_rho/weight_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [11, 106, 99, 13, 7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[6] = new VariableTensor( l1_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/979', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/980', 'op': 'aten::log1p', 'in': [7], 'output_id': 0, 'shape': [400, 784], 'out': [25], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/946', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [16, 259, 150, 137, 246, 33], 'sorted_id': 9}
        {
            Tensor c = (fprec)0.0;
            forward_result[9] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/944', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [24, 11, 263, 148, 366, 358, 31, 250, 134, 154, 37, 349, 243, 257, 350, 141], 'sorted_id': 10}
        {
            Tensor c = (fprec)0.0;
            forward_result[10] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/971', 'op': 'aten::size', 'in': [6, 10], 'output_id': 0, 'shape': [], 'out': [14, 18], 'sorted_id': 11}
        {
            SizeOp* op = new SizeOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/945', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [84, 233, 354, 310, 265, 289, 191, 42, 212, 194, 314, 267, 286, 278, 344, 202, 220, 169, 58, 300, 70, 116, 183, 362, 342, 201, 117, 92, 91, 95, 98, 126, 208, 252, 73, 158, 190, 244, 285, 347, 80, 299, 54, 235, 213, 177, 143, 322, 205, 67, 52, 66, 166, 364, 165, 274, 334, 124, 329, 135, 226, 156, 81, 103, 39, 111, 180, 225, 356, 303, 104, 335, 176, 311, 292, 13, 317, 26, 275, 321], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/972', 'op': 'aten::size', 'in': [6, 12], 'output_id': 0, 'shape': [], 'out': [14, 18], 'sorted_id': 13}
        {
            SizeOp* op = new SizeOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/973', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/947', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [24, 19, 250, 154, 261, 141, 16, 248, 152, 263, 139, 35, 351, 259, 150, 137, 37, 246, 33], 'sorted_id': 15}
        {
            Tensor c = (fprec)0.0;
            forward_result[15] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/974', 'op': 'aten::expand', 'in': [9, 14, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/948', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [248, 19, 139, 35, 261, 152], 'sorted_id': 17}
        {
            Tensor c = (fprec)1.0;
            forward_result[17] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/975', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/976', 'op': 'aten::expand', 'in': [17, 18, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/949', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [24, 112, 250, 249, 262, 36, 154, 367, 140, 141, 316, 182, 221, 291, 21, 125, 72, 330, 207, 97, 263, 359, 153, 351, 234, 343, 37, 347], 'sorted_id': 20}
        {
            forward_result[20] = NULL;
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/977', 'op': 'aten::normal', 'in': [16, 19, 20], 'output_id': 0, 'shape': [400, 784], 'out': [24], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/950', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [24, 263, 250, 154, 37, 141], 'sorted_id': 22}
        {
            Tensor c = (fprec)6.0;
            forward_result[22] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/951', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [24, 263, 250, 154, 37, 141], 'sorted_id': 23}
        {
            forward_result[23] = NULL;
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.1', 'op': 'aten::to', 'in': [21, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 784], 'out': [25], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {400,784};
            ToOp* op = new ToOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/981', 'op': 'aten::mul', 'in': [8, 24], 'output_id': 0, 'shape': [400, 784], 'out': [26], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.1', 'op': 'aten::add', 'in': [5, 25, 12], 'output_id': 0, 'shape': [400, 784], 'out': [40, 58, 104, 42], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_mu/bias_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [117, 39], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[27] = new VariableTensor( l1_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_rho/bias_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [31, 119, 113, 29], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[28] = new VariableTensor( l1_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/990', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [30], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/991', 'op': 'aten::log1p', 'in': [29], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/983', 'op': 'aten::size', 'in': [28, 10], 'output_id': 0, 'shape': [], 'out': [32, 34], 'sorted_id': 31}
        {
            SizeOp* op = new SizeOp();
            forward_result[31] = op;
            
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/984', 'op': 'prim::ListConstruct', 'in': [31], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/985', 'op': 'aten::expand', 'in': [9, 32, 15], 'output_id': 0, 'shape': [400], 'out': [36], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/986', 'op': 'prim::ListConstruct', 'in': [31], 'output_id': 0, 'shape': [], 'out': [35], 'sorted_id': 34}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/987', 'op': 'aten::expand', 'in': [17, 34, 15], 'output_id': 0, 'shape': [400], 'out': [36], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/988', 'op': 'aten::normal', 'in': [33, 35, 20], 'output_id': 0, 'shape': [400], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.3', 'op': 'aten::to', 'in': [36, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/992', 'op': 'aten::mul', 'in': [30, 37], 'output_id': 0, 'shape': [400], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.3', 'op': 'aten::add', 'in': [27, 38, 12], 'output_id': 0, 'shape': [400], 'out': [84, 40, 73, 117], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/input.3', 'op': 'aten::linear', 'in': [4, 26, 39], 'output_id': 0, 'shape': [4, 400], 'out': [127], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/954', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [84, 183, 42, 194, 158, 292, 278, 303, 169, 58, 73, 267], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {1};
            Constant1.reshape( shape );
            forward_result[41] = new VariableTensor( Constant1, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/996', 'op': 'aten::sub', 'in': [26, 41, 12], 'output_id': 0, 'shape': [400, 784], 'out': [44], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/953', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [59, 279, 62, 74, 270, 339, 323, 281, 159, 295, 304, 186, 161, 214, 184, 293, 76, 170, 227, 217, 326, 172, 47, 87, 108, 118, 336, 197, 195, 105, 85, 306, 44, 121, 230, 268], 'sorted_id': 43}
        {
            Tensor c = (fprec)2.0;
            forward_result[43] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/997', 'op': 'aten::pow', 'in': [42, 43], 'output_id': 0, 'shape': [400, 784], 'out': [45], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[42] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/998', 'op': 'aten::neg', 'in': [44], 'output_id': 0, 'shape': [400, 784], 'out': [50], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[45] = op;
            
            op->set_inputs( forward_result[44] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/952', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [186, 161, 51, 273, 298, 79, 76, 47, 270, 164, 295, 189], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {1};
            Constant2.reshape( shape );
            forward_result[46] = new VariableTensor( Constant2, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.1', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [49], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/955', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [198, 271, 327, 49, 296, 109, 122, 162, 218, 231, 307, 282, 77, 187, 88, 340, 173, 63], 'sorted_id': 48}
        {
            Tensor c = (fprec)2.0;
            forward_result[48] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/999', 'op': 'aten::mul', 'in': [47, 48], 'output_id': 0, 'shape': [1], 'out': [50], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1000', 'op': 'aten::div', 'in': [45, 49], 'output_id': 0, 'shape': [400, 784], 'out': [52], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.1', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1001', 'op': 'aten::sub', 'in': [50, 51, 12], 'output_id': 0, 'shape': [400, 784], 'out': [54], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/956', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.9189, 'out': [166, 54, 81, 177, 311, 92, 286, 202, 275, 191, 67, 300], 'sorted_id': 53}
        {
            Tensor c = (fprec)0.9189;
            forward_result[53] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1002', 'op': 'aten::sub', 'in': [52, 53, 12], 'output_id': 0, 'shape': [400, 784], 'out': [55], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.1', 'op': 'aten::exp', 'in': [54], 'output_id': 0, 'shape': [400, 784], 'out': [57], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/958', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [193, 313, 83, 302, 57, 277, 69, 168, 94, 204, 288, 179], 'sorted_id': 56}
        {
            Tensor c = (fprec)0.5;
            forward_result[56] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1014', 'op': 'aten::mul', 'in': [55, 56], 'output_id': 0, 'shape': [400, 784], 'out': [70], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1006', 'op': 'aten::sub', 'in': [26, 41, 12], 'output_id': 0, 'shape': [400, 784], 'out': [59], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[58] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1007', 'op': 'aten::pow', 'in': [58, 43], 'output_id': 0, 'shape': [400, 784], 'out': [60], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1008', 'op': 'aten::neg', 'in': [59], 'output_id': 0, 'shape': [400, 784], 'out': [64], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[60] = op;
            
            op->set_inputs( forward_result[59] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/957', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [65, 172, 62, 87, 306, 175, 90, 200, 281, 309, 197, 284], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {1};
            Constant3.reshape( shape );
            forward_result[61] = new VariableTensor( Constant3, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.3', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1009', 'op': 'aten::mul', 'in': [62, 48], 'output_id': 0, 'shape': [1], 'out': [64], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1010', 'op': 'aten::div', 'in': [60, 63], 'output_id': 0, 'shape': [400, 784], 'out': [66], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.3', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1011', 'op': 'aten::sub', 'in': [64, 65, 12], 'output_id': 0, 'shape': [400, 784], 'out': [67], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1012', 'op': 'aten::sub', 'in': [66, 53, 12], 'output_id': 0, 'shape': [400, 784], 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.1', 'op': 'aten::exp', 'in': [67], 'output_id': 0, 'shape': [400, 784], 'out': [69], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1015', 'op': 'aten::mul', 'in': [68, 56], 'output_id': 0, 'shape': [400, 784], 'out': [70], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[68] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1016', 'op': 'aten::add', 'in': [57, 69, 12], 'output_id': 0, 'shape': [400, 784], 'out': [71], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[70] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1017', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [400, 784], 'out': [72], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1018', 'op': 'aten::sum', 'in': [71, 20], 'output_id': 0, 'shape': [], 'out': [98], 'sorted_id': 72}
        {
            SumOp* op = new SumOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1021', 'op': 'aten::sub', 'in': [39, 41, 12], 'output_id': 0, 'shape': [400], 'out': [74], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1022', 'op': 'aten::pow', 'in': [73, 43], 'output_id': 0, 'shape': [400], 'out': [75], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1023', 'op': 'aten::neg', 'in': [74], 'output_id': 0, 'shape': [400], 'out': [78], 'sorted_id': 75}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[75] = op;
            
            op->set_inputs( forward_result[74] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.5', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [77], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1024', 'op': 'aten::mul', 'in': [76, 48], 'output_id': 0, 'shape': [1], 'out': [78], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1025', 'op': 'aten::div', 'in': [75, 77], 'output_id': 0, 'shape': [400], 'out': [80], 'sorted_id': 78}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[78] = op;
            
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[77] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.5', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [80], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1026', 'op': 'aten::sub', 'in': [78, 79, 12], 'output_id': 0, 'shape': [400], 'out': [81], 'sorted_id': 80}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[80] = op;
            
            op->set_inputs( forward_result[78] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1027', 'op': 'aten::sub', 'in': [80, 53, 12], 'output_id': 0, 'shape': [400], 'out': [82], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[80] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.3', 'op': 'aten::exp', 'in': [81], 'output_id': 0, 'shape': [400], 'out': [83], 'sorted_id': 82}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[82] = op;
            
            op->set_inputs( forward_result[81] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1039', 'op': 'aten::mul', 'in': [82, 56], 'output_id': 0, 'shape': [400], 'out': [95], 'sorted_id': 83}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[83] = op;
            
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1031', 'op': 'aten::sub', 'in': [39, 41, 12], 'output_id': 0, 'shape': [400], 'out': [85], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1032', 'op': 'aten::pow', 'in': [84, 43], 'output_id': 0, 'shape': [400], 'out': [86], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1033', 'op': 'aten::neg', 'in': [85], 'output_id': 0, 'shape': [400], 'out': [89], 'sorted_id': 86}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[86] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.7', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [88], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1034', 'op': 'aten::mul', 'in': [87, 48], 'output_id': 0, 'shape': [1], 'out': [89], 'sorted_id': 88}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[88] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1035', 'op': 'aten::div', 'in': [86, 88], 'output_id': 0, 'shape': [400], 'out': [91], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[86] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.7', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [91], 'sorted_id': 90}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[90] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1036', 'op': 'aten::sub', 'in': [89, 90, 12], 'output_id': 0, 'shape': [400], 'out': [92], 'sorted_id': 91}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[91] = op;
            
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[90] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1037', 'op': 'aten::sub', 'in': [91, 53, 12], 'output_id': 0, 'shape': [400], 'out': [93], 'sorted_id': 92}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[92] = op;
            
            op->set_inputs( forward_result[91] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.3', 'op': 'aten::exp', 'in': [92], 'output_id': 0, 'shape': [400], 'out': [94], 'sorted_id': 93}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[92] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1040', 'op': 'aten::mul', 'in': [93, 56], 'output_id': 0, 'shape': [400], 'out': [95], 'sorted_id': 94}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[94] = op;
            
            op->set_inputs( forward_result[93] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1041', 'op': 'aten::add', 'in': [83, 94, 12], 'output_id': 0, 'shape': [400], 'out': [96], 'sorted_id': 95}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[95] = op;
            
            op->set_inputs( forward_result[83] );
            op->set_inputs( forward_result[94] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1042', 'op': 'aten::log', 'in': [95], 'output_id': 0, 'shape': [400], 'out': [97], 'sorted_id': 96}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[96] = op;
            
            op->set_inputs( forward_result[95] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1043', 'op': 'aten::sum', 'in': [96, 20], 'output_id': 0, 'shape': [], 'out': [98], 'sorted_id': 97}
        {
            SumOp* op = new SumOp();
            forward_result[97] = op;
            
            op->set_inputs( forward_result[96] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1044', 'op': 'aten::add', 'in': [72, 97, 12], 'output_id': 0, 'shape': [], 'out': [127], 'sorted_id': 98}
        {
            AddOp* op = new AddOp();
            forward_result[98] = op;
            
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[97] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1045', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [100], 'sorted_id': 99}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[99] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1046', 'op': 'aten::log1p', 'in': [99], 'output_id': 0, 'shape': [400, 784], 'out': [101], 'sorted_id': 100}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[100] = op;
            
            op->set_inputs( forward_result[99] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1047', 'op': 'aten::log', 'in': [100], 'output_id': 0, 'shape': [400, 784], 'out': [103], 'sorted_id': 101}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[101] = op;
            
            op->set_inputs( forward_result[100] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/959', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -0.9189385332046727, 'out': [334, 225, 212, 103, 321, 116], 'sorted_id': 102}
        {
            Tensor c = (fprec)-0.9189385332046727;
            forward_result[102] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1048', 'op': 'aten::rsub', 'in': [101, 102, 12], 'output_id': 0, 'shape': [400, 784], 'out': [111], 'sorted_id': 103}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[103] = op;
            
            op->set_inputs( forward_result[101] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1049', 'op': 'aten::sub', 'in': [26, 5, 12], 'output_id': 0, 'shape': [400, 784], 'out': [105], 'sorted_id': 104}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[104] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1050', 'op': 'aten::pow', 'in': [104, 43], 'output_id': 0, 'shape': [400, 784], 'out': [110], 'sorted_id': 105}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[105] = op;
            
            op->set_inputs( forward_result[104] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1051', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [107], 'sorted_id': 106}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[106] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1052', 'op': 'aten::log1p', 'in': [106], 'output_id': 0, 'shape': [400, 784], 'out': [108], 'sorted_id': 107}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[107] = op;
            
            op->set_inputs( forward_result[106] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1053', 'op': 'aten::pow', 'in': [107, 43], 'output_id': 0, 'shape': [400, 784], 'out': [109], 'sorted_id': 108}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[108] = op;
            
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1054', 'op': 'aten::mul', 'in': [108, 48], 'output_id': 0, 'shape': [400, 784], 'out': [110], 'sorted_id': 109}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[109] = op;
            
            op->set_inputs( forward_result[108] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1055', 'op': 'aten::div', 'in': [105, 109], 'output_id': 0, 'shape': [400, 784], 'out': [111], 'sorted_id': 110}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[110] = op;
            
            op->set_inputs( forward_result[105] );
            op->set_inputs( forward_result[109] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1056', 'op': 'aten::sub', 'in': [103, 110, 12], 'output_id': 0, 'shape': [400, 784], 'out': [112], 'sorted_id': 111}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[111] = op;
            
            op->set_inputs( forward_result[103] );
            op->set_inputs( forward_result[110] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1057', 'op': 'aten::sum', 'in': [111, 20], 'output_id': 0, 'shape': [], 'out': [126], 'sorted_id': 112}
        {
            SumOp* op = new SumOp();
            forward_result[112] = op;
            
            op->set_inputs( forward_result[111] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1058', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [114], 'sorted_id': 113}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[113] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1059', 'op': 'aten::log1p', 'in': [113], 'output_id': 0, 'shape': [400], 'out': [115], 'sorted_id': 114}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[114] = op;
            
            op->set_inputs( forward_result[113] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1060', 'op': 'aten::log', 'in': [114], 'output_id': 0, 'shape': [400], 'out': [116], 'sorted_id': 115}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[115] = op;
            
            op->set_inputs( forward_result[114] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1061', 'op': 'aten::rsub', 'in': [115, 102, 12], 'output_id': 0, 'shape': [400], 'out': [124], 'sorted_id': 116}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[116] = op;
            
            op->set_inputs( forward_result[115] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1062', 'op': 'aten::sub', 'in': [39, 27, 12], 'output_id': 0, 'shape': [400], 'out': [118], 'sorted_id': 117}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[117] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1063', 'op': 'aten::pow', 'in': [117, 43], 'output_id': 0, 'shape': [400], 'out': [123], 'sorted_id': 118}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[118] = op;
            
            op->set_inputs( forward_result[117] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1064', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [120], 'sorted_id': 119}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[119] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1065', 'op': 'aten::log1p', 'in': [119], 'output_id': 0, 'shape': [400], 'out': [121], 'sorted_id': 120}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[120] = op;
            
            op->set_inputs( forward_result[119] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1066', 'op': 'aten::pow', 'in': [120, 43], 'output_id': 0, 'shape': [400], 'out': [122], 'sorted_id': 121}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[121] = op;
            
            op->set_inputs( forward_result[120] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1067', 'op': 'aten::mul', 'in': [121, 48], 'output_id': 0, 'shape': [400], 'out': [123], 'sorted_id': 122}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[122] = op;
            
            op->set_inputs( forward_result[121] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1068', 'op': 'aten::div', 'in': [118, 122], 'output_id': 0, 'shape': [400], 'out': [124], 'sorted_id': 123}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[123] = op;
            
            op->set_inputs( forward_result[118] );
            op->set_inputs( forward_result[122] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1069', 'op': 'aten::sub', 'in': [116, 123, 12], 'output_id': 0, 'shape': [400], 'out': [125], 'sorted_id': 124}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[124] = op;
            
            op->set_inputs( forward_result[116] );
            op->set_inputs( forward_result[123] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1070', 'op': 'aten::sum', 'in': [124, 20], 'output_id': 0, 'shape': [], 'out': [126], 'sorted_id': 125}
        {
            SumOp* op = new SumOp();
            forward_result[125] = op;
            
            op->set_inputs( forward_result[124] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/1071', 'op': 'aten::add', 'in': [112, 125, 12], 'output_id': 0, 'shape': [], 'out': [127], 'sorted_id': 126}
        {
            AddOp* op = new AddOp();
            forward_result[126] = op;
            
            op->set_inputs( forward_result[112] );
            op->set_inputs( forward_result[125] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/1073', 'op': 'prim::TupleConstruct', 'in': [40, 98, 126], 'output_id': 0, 'shape': [], 'out': [128, 360, 352], 'sorted_id': 127}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[127] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[98] );
            op->set_inputs( forward_result[126] );
        }
        
        // {'name': 'Model/1074', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 0, 'shape': [4, 400], 'out': [129], 'sorted_id': 128}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[128] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Model/Net[net]/input.5', 'op': 'aten::relu', 'in': [128], 'output_id': 0, 'shape': [4, 400], 'out': [157], 'sorted_id': 129}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[129] = op;
            
            op->set_inputs( forward_result[128] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_mu/weight_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [213, 143], 'sorted_id': 130}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[130] = new VariableTensor( l2_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_rho/weight_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [209, 132, 134, 135, 215], 'sorted_id': 131}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[131] = new VariableTensor( l2_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1090', 'op': 'aten::exp', 'in': [131], 'output_id': 0, 'shape': [400, 400], 'out': [133], 'sorted_id': 132}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[132] = op;
            
            op->set_inputs( forward_result[131] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1091', 'op': 'aten::log1p', 'in': [132], 'output_id': 0, 'shape': [400, 400], 'out': [142], 'sorted_id': 133}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[133] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1082', 'op': 'aten::size', 'in': [131, 10], 'output_id': 0, 'shape': [], 'out': [136, 138], 'sorted_id': 134}
        {
            SizeOp* op = new SizeOp();
            forward_result[134] = op;
            
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1083', 'op': 'aten::size', 'in': [131, 12], 'output_id': 0, 'shape': [], 'out': [136, 138], 'sorted_id': 135}
        {
            SizeOp* op = new SizeOp();
            forward_result[135] = op;
            
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1084', 'op': 'prim::ListConstruct', 'in': [134, 135], 'output_id': 0, 'shape': [], 'out': [137], 'sorted_id': 136}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[136] = op;
            
            op->set_inputs( forward_result[134] );
            op->set_inputs( forward_result[135] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1085', 'op': 'aten::expand', 'in': [9, 136, 15], 'output_id': 0, 'shape': [400, 400], 'out': [140], 'sorted_id': 137}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[137] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[136] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1086', 'op': 'prim::ListConstruct', 'in': [134, 135], 'output_id': 0, 'shape': [], 'out': [139], 'sorted_id': 138}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[138] = op;
            
            op->set_inputs( forward_result[134] );
            op->set_inputs( forward_result[135] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1087', 'op': 'aten::expand', 'in': [17, 138, 15], 'output_id': 0, 'shape': [400, 400], 'out': [140], 'sorted_id': 139}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[139] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[138] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1088', 'op': 'aten::normal', 'in': [137, 139, 20], 'output_id': 0, 'shape': [400, 400], 'out': [141], 'sorted_id': 140}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[140] = op;
            
            op->set_inputs( forward_result[137] );
            op->set_inputs( forward_result[139] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.5', 'op': 'aten::to', 'in': [140, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 400], 'out': [142], 'sorted_id': 141}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[141] = op;
            
            op->set_inputs( forward_result[140] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1092', 'op': 'aten::mul', 'in': [133, 141], 'output_id': 0, 'shape': [400, 400], 'out': [143], 'sorted_id': 142}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[142] = op;
            
            op->set_inputs( forward_result[133] );
            op->set_inputs( forward_result[141] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.5', 'op': 'aten::add', 'in': [130, 142, 12], 'output_id': 0, 'shape': [400, 400], 'out': [213, 157, 158, 169], 'sorted_id': 143}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[143] = op;
            
            op->set_inputs( forward_result[130] );
            op->set_inputs( forward_result[142] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_mu/bias_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [226, 156], 'sorted_id': 144}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[144] = new VariableTensor( l2_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_rho/bias_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [222, 146, 148, 228], 'sorted_id': 145}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[145] = new VariableTensor( l2_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1101', 'op': 'aten::exp', 'in': [145], 'output_id': 0, 'shape': [400], 'out': [147], 'sorted_id': 146}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[146] = op;
            
            op->set_inputs( forward_result[145] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1102', 'op': 'aten::log1p', 'in': [146], 'output_id': 0, 'shape': [400], 'out': [155], 'sorted_id': 147}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[147] = op;
            
            op->set_inputs( forward_result[146] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1094', 'op': 'aten::size', 'in': [145, 10], 'output_id': 0, 'shape': [], 'out': [151, 149], 'sorted_id': 148}
        {
            SizeOp* op = new SizeOp();
            forward_result[148] = op;
            
            op->set_inputs( forward_result[145] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1095', 'op': 'prim::ListConstruct', 'in': [148], 'output_id': 0, 'shape': [], 'out': [150], 'sorted_id': 149}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[149] = op;
            
            op->set_inputs( forward_result[148] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1096', 'op': 'aten::expand', 'in': [9, 149, 15], 'output_id': 0, 'shape': [400], 'out': [153], 'sorted_id': 150}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[150] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[149] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1097', 'op': 'prim::ListConstruct', 'in': [148], 'output_id': 0, 'shape': [], 'out': [152], 'sorted_id': 151}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[151] = op;
            
            op->set_inputs( forward_result[148] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1098', 'op': 'aten::expand', 'in': [17, 151, 15], 'output_id': 0, 'shape': [400], 'out': [153], 'sorted_id': 152}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[152] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[151] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1099', 'op': 'aten::normal', 'in': [150, 152, 20], 'output_id': 0, 'shape': [400], 'out': [154], 'sorted_id': 153}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[153] = op;
            
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[152] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.7', 'op': 'aten::to', 'in': [153, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [155], 'sorted_id': 154}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[154] = op;
            
            op->set_inputs( forward_result[153] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1103', 'op': 'aten::mul', 'in': [147, 154], 'output_id': 0, 'shape': [400], 'out': [156], 'sorted_id': 155}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[155] = op;
            
            op->set_inputs( forward_result[147] );
            op->set_inputs( forward_result[154] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.7', 'op': 'aten::add', 'in': [144, 155, 12], 'output_id': 0, 'shape': [400], 'out': [226, 183, 157, 194], 'sorted_id': 156}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[156] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[155] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/input.7', 'op': 'aten::linear', 'in': [129, 143, 156], 'output_id': 0, 'shape': [4, 400], 'out': [236], 'sorted_id': 157}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[157] = op;
            
            op->set_inputs( forward_result[129] );
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[156] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1107', 'op': 'aten::sub', 'in': [143, 41, 12], 'output_id': 0, 'shape': [400, 400], 'out': [159], 'sorted_id': 158}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[158] = op;
            
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1108', 'op': 'aten::pow', 'in': [158, 43], 'output_id': 0, 'shape': [400, 400], 'out': [160], 'sorted_id': 159}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[159] = op;
            
            op->set_inputs( forward_result[158] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1109', 'op': 'aten::neg', 'in': [159], 'output_id': 0, 'shape': [400, 400], 'out': [163], 'sorted_id': 160}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[160] = op;
            
            op->set_inputs( forward_result[159] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.9', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [162], 'sorted_id': 161}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[161] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1110', 'op': 'aten::mul', 'in': [161, 48], 'output_id': 0, 'shape': [1], 'out': [163], 'sorted_id': 162}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[162] = op;
            
            op->set_inputs( forward_result[161] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1111', 'op': 'aten::div', 'in': [160, 162], 'output_id': 0, 'shape': [400, 400], 'out': [165], 'sorted_id': 163}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[163] = op;
            
            op->set_inputs( forward_result[160] );
            op->set_inputs( forward_result[162] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.9', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [165], 'sorted_id': 164}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[164] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1112', 'op': 'aten::sub', 'in': [163, 164, 12], 'output_id': 0, 'shape': [400, 400], 'out': [166], 'sorted_id': 165}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[165] = op;
            
            op->set_inputs( forward_result[163] );
            op->set_inputs( forward_result[164] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1113', 'op': 'aten::sub', 'in': [165, 53, 12], 'output_id': 0, 'shape': [400, 400], 'out': [167], 'sorted_id': 166}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[166] = op;
            
            op->set_inputs( forward_result[165] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.5', 'op': 'aten::exp', 'in': [166], 'output_id': 0, 'shape': [400, 400], 'out': [168], 'sorted_id': 167}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[167] = op;
            
            op->set_inputs( forward_result[166] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1125', 'op': 'aten::mul', 'in': [167, 56], 'output_id': 0, 'shape': [400, 400], 'out': [180], 'sorted_id': 168}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[168] = op;
            
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1117', 'op': 'aten::sub', 'in': [143, 41, 12], 'output_id': 0, 'shape': [400, 400], 'out': [170], 'sorted_id': 169}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[169] = op;
            
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1118', 'op': 'aten::pow', 'in': [169, 43], 'output_id': 0, 'shape': [400, 400], 'out': [171], 'sorted_id': 170}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[170] = op;
            
            op->set_inputs( forward_result[169] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1119', 'op': 'aten::neg', 'in': [170], 'output_id': 0, 'shape': [400, 400], 'out': [174], 'sorted_id': 171}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[171] = op;
            
            op->set_inputs( forward_result[170] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.11', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [173], 'sorted_id': 172}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[172] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1120', 'op': 'aten::mul', 'in': [172, 48], 'output_id': 0, 'shape': [1], 'out': [174], 'sorted_id': 173}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[173] = op;
            
            op->set_inputs( forward_result[172] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1121', 'op': 'aten::div', 'in': [171, 173], 'output_id': 0, 'shape': [400, 400], 'out': [176], 'sorted_id': 174}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[174] = op;
            
            op->set_inputs( forward_result[171] );
            op->set_inputs( forward_result[173] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.11', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [176], 'sorted_id': 175}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[175] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1122', 'op': 'aten::sub', 'in': [174, 175, 12], 'output_id': 0, 'shape': [400, 400], 'out': [177], 'sorted_id': 176}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[176] = op;
            
            op->set_inputs( forward_result[174] );
            op->set_inputs( forward_result[175] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1123', 'op': 'aten::sub', 'in': [176, 53, 12], 'output_id': 0, 'shape': [400, 400], 'out': [178], 'sorted_id': 177}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[177] = op;
            
            op->set_inputs( forward_result[176] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.5', 'op': 'aten::exp', 'in': [177], 'output_id': 0, 'shape': [400, 400], 'out': [179], 'sorted_id': 178}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[178] = op;
            
            op->set_inputs( forward_result[177] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1126', 'op': 'aten::mul', 'in': [178, 56], 'output_id': 0, 'shape': [400, 400], 'out': [180], 'sorted_id': 179}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[179] = op;
            
            op->set_inputs( forward_result[178] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1127', 'op': 'aten::add', 'in': [168, 179, 12], 'output_id': 0, 'shape': [400, 400], 'out': [181], 'sorted_id': 180}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[180] = op;
            
            op->set_inputs( forward_result[168] );
            op->set_inputs( forward_result[179] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1128', 'op': 'aten::log', 'in': [180], 'output_id': 0, 'shape': [400, 400], 'out': [182], 'sorted_id': 181}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[181] = op;
            
            op->set_inputs( forward_result[180] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1129', 'op': 'aten::sum', 'in': [181, 20], 'output_id': 0, 'shape': [], 'out': [208], 'sorted_id': 182}
        {
            SumOp* op = new SumOp();
            forward_result[182] = op;
            
            op->set_inputs( forward_result[181] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1132', 'op': 'aten::sub', 'in': [156, 41, 12], 'output_id': 0, 'shape': [400], 'out': [184], 'sorted_id': 183}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[183] = op;
            
            op->set_inputs( forward_result[156] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1133', 'op': 'aten::pow', 'in': [183, 43], 'output_id': 0, 'shape': [400], 'out': [185], 'sorted_id': 184}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[184] = op;
            
            op->set_inputs( forward_result[183] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1134', 'op': 'aten::neg', 'in': [184], 'output_id': 0, 'shape': [400], 'out': [188], 'sorted_id': 185}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[185] = op;
            
            op->set_inputs( forward_result[184] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.13', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [187], 'sorted_id': 186}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[186] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1135', 'op': 'aten::mul', 'in': [186, 48], 'output_id': 0, 'shape': [1], 'out': [188], 'sorted_id': 187}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[187] = op;
            
            op->set_inputs( forward_result[186] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1136', 'op': 'aten::div', 'in': [185, 187], 'output_id': 0, 'shape': [400], 'out': [190], 'sorted_id': 188}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[188] = op;
            
            op->set_inputs( forward_result[185] );
            op->set_inputs( forward_result[187] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.13', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [190], 'sorted_id': 189}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[189] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1137', 'op': 'aten::sub', 'in': [188, 189, 12], 'output_id': 0, 'shape': [400], 'out': [191], 'sorted_id': 190}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[190] = op;
            
            op->set_inputs( forward_result[188] );
            op->set_inputs( forward_result[189] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1138', 'op': 'aten::sub', 'in': [190, 53, 12], 'output_id': 0, 'shape': [400], 'out': [192], 'sorted_id': 191}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[191] = op;
            
            op->set_inputs( forward_result[190] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.7', 'op': 'aten::exp', 'in': [191], 'output_id': 0, 'shape': [400], 'out': [193], 'sorted_id': 192}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[192] = op;
            
            op->set_inputs( forward_result[191] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1150', 'op': 'aten::mul', 'in': [192, 56], 'output_id': 0, 'shape': [400], 'out': [205], 'sorted_id': 193}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[193] = op;
            
            op->set_inputs( forward_result[192] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1142', 'op': 'aten::sub', 'in': [156, 41, 12], 'output_id': 0, 'shape': [400], 'out': [195], 'sorted_id': 194}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[194] = op;
            
            op->set_inputs( forward_result[156] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1143', 'op': 'aten::pow', 'in': [194, 43], 'output_id': 0, 'shape': [400], 'out': [196], 'sorted_id': 195}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[195] = op;
            
            op->set_inputs( forward_result[194] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1144', 'op': 'aten::neg', 'in': [195], 'output_id': 0, 'shape': [400], 'out': [199], 'sorted_id': 196}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[196] = op;
            
            op->set_inputs( forward_result[195] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.15', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [198], 'sorted_id': 197}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[197] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1145', 'op': 'aten::mul', 'in': [197, 48], 'output_id': 0, 'shape': [1], 'out': [199], 'sorted_id': 198}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[198] = op;
            
            op->set_inputs( forward_result[197] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1146', 'op': 'aten::div', 'in': [196, 198], 'output_id': 0, 'shape': [400], 'out': [201], 'sorted_id': 199}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[199] = op;
            
            op->set_inputs( forward_result[196] );
            op->set_inputs( forward_result[198] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.15', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [201], 'sorted_id': 200}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[200] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1147', 'op': 'aten::sub', 'in': [199, 200, 12], 'output_id': 0, 'shape': [400], 'out': [202], 'sorted_id': 201}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[201] = op;
            
            op->set_inputs( forward_result[199] );
            op->set_inputs( forward_result[200] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1148', 'op': 'aten::sub', 'in': [201, 53, 12], 'output_id': 0, 'shape': [400], 'out': [203], 'sorted_id': 202}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[202] = op;
            
            op->set_inputs( forward_result[201] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.7', 'op': 'aten::exp', 'in': [202], 'output_id': 0, 'shape': [400], 'out': [204], 'sorted_id': 203}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[203] = op;
            
            op->set_inputs( forward_result[202] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1151', 'op': 'aten::mul', 'in': [203, 56], 'output_id': 0, 'shape': [400], 'out': [205], 'sorted_id': 204}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[204] = op;
            
            op->set_inputs( forward_result[203] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1152', 'op': 'aten::add', 'in': [193, 204, 12], 'output_id': 0, 'shape': [400], 'out': [206], 'sorted_id': 205}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[205] = op;
            
            op->set_inputs( forward_result[193] );
            op->set_inputs( forward_result[204] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1153', 'op': 'aten::log', 'in': [205], 'output_id': 0, 'shape': [400], 'out': [207], 'sorted_id': 206}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[206] = op;
            
            op->set_inputs( forward_result[205] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1154', 'op': 'aten::sum', 'in': [206, 20], 'output_id': 0, 'shape': [], 'out': [208], 'sorted_id': 207}
        {
            SumOp* op = new SumOp();
            forward_result[207] = op;
            
            op->set_inputs( forward_result[206] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1155', 'op': 'aten::add', 'in': [182, 207, 12], 'output_id': 0, 'shape': [], 'out': [236], 'sorted_id': 208}
        {
            AddOp* op = new AddOp();
            forward_result[208] = op;
            
            op->set_inputs( forward_result[182] );
            op->set_inputs( forward_result[207] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1156', 'op': 'aten::exp', 'in': [131], 'output_id': 0, 'shape': [400, 400], 'out': [210], 'sorted_id': 209}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[209] = op;
            
            op->set_inputs( forward_result[131] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1157', 'op': 'aten::log1p', 'in': [209], 'output_id': 0, 'shape': [400, 400], 'out': [211], 'sorted_id': 210}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[210] = op;
            
            op->set_inputs( forward_result[209] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1158', 'op': 'aten::log', 'in': [210], 'output_id': 0, 'shape': [400, 400], 'out': [212], 'sorted_id': 211}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[211] = op;
            
            op->set_inputs( forward_result[210] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1159', 'op': 'aten::rsub', 'in': [211, 102, 12], 'output_id': 0, 'shape': [400, 400], 'out': [220], 'sorted_id': 212}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[212] = op;
            
            op->set_inputs( forward_result[211] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1160', 'op': 'aten::sub', 'in': [143, 130, 12], 'output_id': 0, 'shape': [400, 400], 'out': [214], 'sorted_id': 213}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[213] = op;
            
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[130] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1161', 'op': 'aten::pow', 'in': [213, 43], 'output_id': 0, 'shape': [400, 400], 'out': [219], 'sorted_id': 214}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[214] = op;
            
            op->set_inputs( forward_result[213] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1162', 'op': 'aten::exp', 'in': [131], 'output_id': 0, 'shape': [400, 400], 'out': [216], 'sorted_id': 215}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[215] = op;
            
            op->set_inputs( forward_result[131] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1163', 'op': 'aten::log1p', 'in': [215], 'output_id': 0, 'shape': [400, 400], 'out': [217], 'sorted_id': 216}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[216] = op;
            
            op->set_inputs( forward_result[215] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1164', 'op': 'aten::pow', 'in': [216, 43], 'output_id': 0, 'shape': [400, 400], 'out': [218], 'sorted_id': 217}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[217] = op;
            
            op->set_inputs( forward_result[216] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1165', 'op': 'aten::mul', 'in': [217, 48], 'output_id': 0, 'shape': [400, 400], 'out': [219], 'sorted_id': 218}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[218] = op;
            
            op->set_inputs( forward_result[217] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1166', 'op': 'aten::div', 'in': [214, 218], 'output_id': 0, 'shape': [400, 400], 'out': [220], 'sorted_id': 219}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[219] = op;
            
            op->set_inputs( forward_result[214] );
            op->set_inputs( forward_result[218] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1167', 'op': 'aten::sub', 'in': [212, 219, 12], 'output_id': 0, 'shape': [400, 400], 'out': [221], 'sorted_id': 220}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[220] = op;
            
            op->set_inputs( forward_result[212] );
            op->set_inputs( forward_result[219] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1168', 'op': 'aten::sum', 'in': [220, 20], 'output_id': 0, 'shape': [], 'out': [235], 'sorted_id': 221}
        {
            SumOp* op = new SumOp();
            forward_result[221] = op;
            
            op->set_inputs( forward_result[220] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1169', 'op': 'aten::exp', 'in': [145], 'output_id': 0, 'shape': [400], 'out': [223], 'sorted_id': 222}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[222] = op;
            
            op->set_inputs( forward_result[145] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1170', 'op': 'aten::log1p', 'in': [222], 'output_id': 0, 'shape': [400], 'out': [224], 'sorted_id': 223}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[223] = op;
            
            op->set_inputs( forward_result[222] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1171', 'op': 'aten::log', 'in': [223], 'output_id': 0, 'shape': [400], 'out': [225], 'sorted_id': 224}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[224] = op;
            
            op->set_inputs( forward_result[223] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1172', 'op': 'aten::rsub', 'in': [224, 102, 12], 'output_id': 0, 'shape': [400], 'out': [233], 'sorted_id': 225}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[225] = op;
            
            op->set_inputs( forward_result[224] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1173', 'op': 'aten::sub', 'in': [156, 144, 12], 'output_id': 0, 'shape': [400], 'out': [227], 'sorted_id': 226}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[226] = op;
            
            op->set_inputs( forward_result[156] );
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1174', 'op': 'aten::pow', 'in': [226, 43], 'output_id': 0, 'shape': [400], 'out': [232], 'sorted_id': 227}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[227] = op;
            
            op->set_inputs( forward_result[226] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1175', 'op': 'aten::exp', 'in': [145], 'output_id': 0, 'shape': [400], 'out': [229], 'sorted_id': 228}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[228] = op;
            
            op->set_inputs( forward_result[145] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1176', 'op': 'aten::log1p', 'in': [228], 'output_id': 0, 'shape': [400], 'out': [230], 'sorted_id': 229}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[229] = op;
            
            op->set_inputs( forward_result[228] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1177', 'op': 'aten::pow', 'in': [229, 43], 'output_id': 0, 'shape': [400], 'out': [231], 'sorted_id': 230}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[230] = op;
            
            op->set_inputs( forward_result[229] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1178', 'op': 'aten::mul', 'in': [230, 48], 'output_id': 0, 'shape': [400], 'out': [232], 'sorted_id': 231}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[231] = op;
            
            op->set_inputs( forward_result[230] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1179', 'op': 'aten::div', 'in': [227, 231], 'output_id': 0, 'shape': [400], 'out': [233], 'sorted_id': 232}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[232] = op;
            
            op->set_inputs( forward_result[227] );
            op->set_inputs( forward_result[231] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1180', 'op': 'aten::sub', 'in': [225, 232, 12], 'output_id': 0, 'shape': [400], 'out': [234], 'sorted_id': 233}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[233] = op;
            
            op->set_inputs( forward_result[225] );
            op->set_inputs( forward_result[232] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1181', 'op': 'aten::sum', 'in': [233, 20], 'output_id': 0, 'shape': [], 'out': [235], 'sorted_id': 234}
        {
            SumOp* op = new SumOp();
            forward_result[234] = op;
            
            op->set_inputs( forward_result[233] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/1182', 'op': 'aten::add', 'in': [221, 234, 12], 'output_id': 0, 'shape': [], 'out': [236], 'sorted_id': 235}
        {
            AddOp* op = new AddOp();
            forward_result[235] = op;
            
            op->set_inputs( forward_result[221] );
            op->set_inputs( forward_result[234] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/1184', 'op': 'prim::TupleConstruct', 'in': [157, 208, 235], 'output_id': 0, 'shape': [], 'out': [237, 361, 353], 'sorted_id': 236}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[236] = op;
            
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[208] );
            op->set_inputs( forward_result[235] );
        }
        
        // {'name': 'Model/1185', 'op': 'prim::TupleUnpack', 'in': [236], 'output_id': 0, 'shape': [4, 400], 'out': [238], 'sorted_id': 237}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[237] = op;
            
            op->set_inputs( forward_result[236] );
        }
        
        // {'name': 'Model/Net[net]/input.9', 'op': 'aten::relu', 'in': [237], 'output_id': 0, 'shape': [4, 400], 'out': [266], 'sorted_id': 238}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[238] = op;
            
            op->set_inputs( forward_result[237] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_mu/weight_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [322, 252], 'sorted_id': 239}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[239] = new VariableTensor( l3_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_rho/weight_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [241, 244, 243, 318, 324], 'sorted_id': 240}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[240] = new VariableTensor( l3_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1201', 'op': 'aten::exp', 'in': [240], 'output_id': 0, 'shape': [10, 400], 'out': [242], 'sorted_id': 241}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[241] = op;
            
            op->set_inputs( forward_result[240] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1202', 'op': 'aten::log1p', 'in': [241], 'output_id': 0, 'shape': [10, 400], 'out': [251], 'sorted_id': 242}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[242] = op;
            
            op->set_inputs( forward_result[241] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1193', 'op': 'aten::size', 'in': [240, 10], 'output_id': 0, 'shape': [], 'out': [247, 245], 'sorted_id': 243}
        {
            SizeOp* op = new SizeOp();
            forward_result[243] = op;
            
            op->set_inputs( forward_result[240] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1194', 'op': 'aten::size', 'in': [240, 12], 'output_id': 0, 'shape': [], 'out': [247, 245], 'sorted_id': 244}
        {
            SizeOp* op = new SizeOp();
            forward_result[244] = op;
            
            op->set_inputs( forward_result[240] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1195', 'op': 'prim::ListConstruct', 'in': [243, 244], 'output_id': 0, 'shape': [], 'out': [246], 'sorted_id': 245}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[245] = op;
            
            op->set_inputs( forward_result[243] );
            op->set_inputs( forward_result[244] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1196', 'op': 'aten::expand', 'in': [9, 245, 15], 'output_id': 0, 'shape': [10, 400], 'out': [249], 'sorted_id': 246}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[246] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[245] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1197', 'op': 'prim::ListConstruct', 'in': [243, 244], 'output_id': 0, 'shape': [], 'out': [248], 'sorted_id': 247}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[247] = op;
            
            op->set_inputs( forward_result[243] );
            op->set_inputs( forward_result[244] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1198', 'op': 'aten::expand', 'in': [17, 247, 15], 'output_id': 0, 'shape': [10, 400], 'out': [249], 'sorted_id': 248}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[248] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[247] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1199', 'op': 'aten::normal', 'in': [246, 248, 20], 'output_id': 0, 'shape': [10, 400], 'out': [250], 'sorted_id': 249}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[249] = op;
            
            op->set_inputs( forward_result[246] );
            op->set_inputs( forward_result[248] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.9', 'op': 'aten::to', 'in': [249, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10, 400], 'out': [251], 'sorted_id': 250}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[250] = op;
            
            op->set_inputs( forward_result[249] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1203', 'op': 'aten::mul', 'in': [242, 250], 'output_id': 0, 'shape': [10, 400], 'out': [252], 'sorted_id': 251}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[251] = op;
            
            op->set_inputs( forward_result[242] );
            op->set_inputs( forward_result[250] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.9', 'op': 'aten::add', 'in': [239, 251, 12], 'output_id': 0, 'shape': [10, 400], 'out': [266, 278, 322, 267], 'sorted_id': 252}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[252] = op;
            
            op->set_inputs( forward_result[239] );
            op->set_inputs( forward_result[251] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_mu/bias_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [265, 335], 'sorted_id': 253}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[253] = new VariableTensor( l3_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_rho/bias_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [337, 257, 255, 331], 'sorted_id': 254}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[254] = new VariableTensor( l3_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1212', 'op': 'aten::exp', 'in': [254], 'output_id': 0, 'shape': [10], 'out': [256], 'sorted_id': 255}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[255] = op;
            
            op->set_inputs( forward_result[254] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1213', 'op': 'aten::log1p', 'in': [255], 'output_id': 0, 'shape': [10], 'out': [264], 'sorted_id': 256}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[256] = op;
            
            op->set_inputs( forward_result[255] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1205', 'op': 'aten::size', 'in': [254, 10], 'output_id': 0, 'shape': [], 'out': [258, 260], 'sorted_id': 257}
        {
            SizeOp* op = new SizeOp();
            forward_result[257] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1206', 'op': 'prim::ListConstruct', 'in': [257], 'output_id': 0, 'shape': [], 'out': [259], 'sorted_id': 258}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[258] = op;
            
            op->set_inputs( forward_result[257] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1207', 'op': 'aten::expand', 'in': [9, 258, 15], 'output_id': 0, 'shape': [10], 'out': [262], 'sorted_id': 259}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[259] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[258] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1208', 'op': 'prim::ListConstruct', 'in': [257], 'output_id': 0, 'shape': [], 'out': [261], 'sorted_id': 260}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[260] = op;
            
            op->set_inputs( forward_result[257] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1209', 'op': 'aten::expand', 'in': [17, 260, 15], 'output_id': 0, 'shape': [10], 'out': [262], 'sorted_id': 261}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[261] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[260] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1210', 'op': 'aten::normal', 'in': [259, 261, 20], 'output_id': 0, 'shape': [10], 'out': [263], 'sorted_id': 262}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[262] = op;
            
            op->set_inputs( forward_result[259] );
            op->set_inputs( forward_result[261] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon', 'op': 'aten::to', 'in': [262, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10], 'out': [264], 'sorted_id': 263}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[263] = op;
            
            op->set_inputs( forward_result[262] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1214', 'op': 'aten::mul', 'in': [256, 263], 'output_id': 0, 'shape': [10], 'out': [265], 'sorted_id': 264}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[264] = op;
            
            op->set_inputs( forward_result[256] );
            op->set_inputs( forward_result[263] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value', 'op': 'aten::add', 'in': [253, 264, 12], 'output_id': 0, 'shape': [10], 'out': [266, 303, 292, 335], 'sorted_id': 265}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[265] = op;
            
            op->set_inputs( forward_result[253] );
            op->set_inputs( forward_result[264] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/input.11', 'op': 'aten::linear', 'in': [238, 252, 265], 'output_id': 0, 'shape': [4, 10], 'out': [345], 'sorted_id': 266}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[266] = op;
            
            op->set_inputs( forward_result[238] );
            op->set_inputs( forward_result[252] );
            op->set_inputs( forward_result[265] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1218', 'op': 'aten::sub', 'in': [252, 41, 12], 'output_id': 0, 'shape': [10, 400], 'out': [268], 'sorted_id': 267}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[267] = op;
            
            op->set_inputs( forward_result[252] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1219', 'op': 'aten::pow', 'in': [267, 43], 'output_id': 0, 'shape': [10, 400], 'out': [269], 'sorted_id': 268}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[268] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1220', 'op': 'aten::neg', 'in': [268], 'output_id': 0, 'shape': [10, 400], 'out': [272], 'sorted_id': 269}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[269] = op;
            
            op->set_inputs( forward_result[268] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.17', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [271], 'sorted_id': 270}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[270] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1221', 'op': 'aten::mul', 'in': [270, 48], 'output_id': 0, 'shape': [1], 'out': [272], 'sorted_id': 271}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[271] = op;
            
            op->set_inputs( forward_result[270] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1222', 'op': 'aten::div', 'in': [269, 271], 'output_id': 0, 'shape': [10, 400], 'out': [274], 'sorted_id': 272}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[272] = op;
            
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[271] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.17', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [274], 'sorted_id': 273}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[273] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1223', 'op': 'aten::sub', 'in': [272, 273, 12], 'output_id': 0, 'shape': [10, 400], 'out': [275], 'sorted_id': 274}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[274] = op;
            
            op->set_inputs( forward_result[272] );
            op->set_inputs( forward_result[273] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1224', 'op': 'aten::sub', 'in': [274, 53, 12], 'output_id': 0, 'shape': [10, 400], 'out': [276], 'sorted_id': 275}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[275] = op;
            
            op->set_inputs( forward_result[274] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.9', 'op': 'aten::exp', 'in': [275], 'output_id': 0, 'shape': [10, 400], 'out': [277], 'sorted_id': 276}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[276] = op;
            
            op->set_inputs( forward_result[275] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1236', 'op': 'aten::mul', 'in': [276, 56], 'output_id': 0, 'shape': [10, 400], 'out': [289], 'sorted_id': 277}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[277] = op;
            
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1228', 'op': 'aten::sub', 'in': [252, 41, 12], 'output_id': 0, 'shape': [10, 400], 'out': [279], 'sorted_id': 278}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[278] = op;
            
            op->set_inputs( forward_result[252] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1229', 'op': 'aten::pow', 'in': [278, 43], 'output_id': 0, 'shape': [10, 400], 'out': [280], 'sorted_id': 279}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[279] = op;
            
            op->set_inputs( forward_result[278] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1230', 'op': 'aten::neg', 'in': [279], 'output_id': 0, 'shape': [10, 400], 'out': [283], 'sorted_id': 280}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[280] = op;
            
            op->set_inputs( forward_result[279] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.19', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [282], 'sorted_id': 281}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[281] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1231', 'op': 'aten::mul', 'in': [281, 48], 'output_id': 0, 'shape': [1], 'out': [283], 'sorted_id': 282}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[282] = op;
            
            op->set_inputs( forward_result[281] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1232', 'op': 'aten::div', 'in': [280, 282], 'output_id': 0, 'shape': [10, 400], 'out': [285], 'sorted_id': 283}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[283] = op;
            
            op->set_inputs( forward_result[280] );
            op->set_inputs( forward_result[282] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.19', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [285], 'sorted_id': 284}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[284] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1233', 'op': 'aten::sub', 'in': [283, 284, 12], 'output_id': 0, 'shape': [10, 400], 'out': [286], 'sorted_id': 285}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[285] = op;
            
            op->set_inputs( forward_result[283] );
            op->set_inputs( forward_result[284] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1234', 'op': 'aten::sub', 'in': [285, 53, 12], 'output_id': 0, 'shape': [10, 400], 'out': [287], 'sorted_id': 286}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[286] = op;
            
            op->set_inputs( forward_result[285] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.9', 'op': 'aten::exp', 'in': [286], 'output_id': 0, 'shape': [10, 400], 'out': [288], 'sorted_id': 287}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[287] = op;
            
            op->set_inputs( forward_result[286] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1237', 'op': 'aten::mul', 'in': [287, 56], 'output_id': 0, 'shape': [10, 400], 'out': [289], 'sorted_id': 288}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[288] = op;
            
            op->set_inputs( forward_result[287] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1238', 'op': 'aten::add', 'in': [277, 288, 12], 'output_id': 0, 'shape': [10, 400], 'out': [290], 'sorted_id': 289}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[289] = op;
            
            op->set_inputs( forward_result[277] );
            op->set_inputs( forward_result[288] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1239', 'op': 'aten::log', 'in': [289], 'output_id': 0, 'shape': [10, 400], 'out': [291], 'sorted_id': 290}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[290] = op;
            
            op->set_inputs( forward_result[289] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1240', 'op': 'aten::sum', 'in': [290, 20], 'output_id': 0, 'shape': [], 'out': [317], 'sorted_id': 291}
        {
            SumOp* op = new SumOp();
            forward_result[291] = op;
            
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1243', 'op': 'aten::sub', 'in': [265, 41, 12], 'output_id': 0, 'shape': [10], 'out': [293], 'sorted_id': 292}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[292] = op;
            
            op->set_inputs( forward_result[265] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1244', 'op': 'aten::pow', 'in': [292, 43], 'output_id': 0, 'shape': [10], 'out': [294], 'sorted_id': 293}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[293] = op;
            
            op->set_inputs( forward_result[292] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1245', 'op': 'aten::neg', 'in': [293], 'output_id': 0, 'shape': [10], 'out': [297], 'sorted_id': 294}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[294] = op;
            
            op->set_inputs( forward_result[293] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.21', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [296], 'sorted_id': 295}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[295] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1246', 'op': 'aten::mul', 'in': [295, 48], 'output_id': 0, 'shape': [1], 'out': [297], 'sorted_id': 296}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[296] = op;
            
            op->set_inputs( forward_result[295] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1247', 'op': 'aten::div', 'in': [294, 296], 'output_id': 0, 'shape': [10], 'out': [299], 'sorted_id': 297}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[297] = op;
            
            op->set_inputs( forward_result[294] );
            op->set_inputs( forward_result[296] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.21', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [299], 'sorted_id': 298}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[298] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1248', 'op': 'aten::sub', 'in': [297, 298, 12], 'output_id': 0, 'shape': [10], 'out': [300], 'sorted_id': 299}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[299] = op;
            
            op->set_inputs( forward_result[297] );
            op->set_inputs( forward_result[298] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1249', 'op': 'aten::sub', 'in': [299, 53, 12], 'output_id': 0, 'shape': [10], 'out': [301], 'sorted_id': 300}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[300] = op;
            
            op->set_inputs( forward_result[299] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1', 'op': 'aten::exp', 'in': [300], 'output_id': 0, 'shape': [10], 'out': [302], 'sorted_id': 301}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[301] = op;
            
            op->set_inputs( forward_result[300] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1261', 'op': 'aten::mul', 'in': [301, 56], 'output_id': 0, 'shape': [10], 'out': [314], 'sorted_id': 302}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[302] = op;
            
            op->set_inputs( forward_result[301] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1253', 'op': 'aten::sub', 'in': [265, 41, 12], 'output_id': 0, 'shape': [10], 'out': [304], 'sorted_id': 303}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[303] = op;
            
            op->set_inputs( forward_result[265] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1254', 'op': 'aten::pow', 'in': [303, 43], 'output_id': 0, 'shape': [10], 'out': [305], 'sorted_id': 304}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[304] = op;
            
            op->set_inputs( forward_result[303] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1255', 'op': 'aten::neg', 'in': [304], 'output_id': 0, 'shape': [10], 'out': [308], 'sorted_id': 305}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[305] = op;
            
            op->set_inputs( forward_result[304] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [307], 'sorted_id': 306}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[306] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1256', 'op': 'aten::mul', 'in': [306, 48], 'output_id': 0, 'shape': [1], 'out': [308], 'sorted_id': 307}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[307] = op;
            
            op->set_inputs( forward_result[306] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1257', 'op': 'aten::div', 'in': [305, 307], 'output_id': 0, 'shape': [10], 'out': [310], 'sorted_id': 308}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[308] = op;
            
            op->set_inputs( forward_result[305] );
            op->set_inputs( forward_result[307] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [310], 'sorted_id': 309}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[309] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1258', 'op': 'aten::sub', 'in': [308, 309, 12], 'output_id': 0, 'shape': [10], 'out': [311], 'sorted_id': 310}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[310] = op;
            
            op->set_inputs( forward_result[308] );
            op->set_inputs( forward_result[309] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1259', 'op': 'aten::sub', 'in': [310, 53, 12], 'output_id': 0, 'shape': [10], 'out': [312], 'sorted_id': 311}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[311] = op;
            
            op->set_inputs( forward_result[310] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2', 'op': 'aten::exp', 'in': [311], 'output_id': 0, 'shape': [10], 'out': [313], 'sorted_id': 312}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[312] = op;
            
            op->set_inputs( forward_result[311] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1262', 'op': 'aten::mul', 'in': [312, 56], 'output_id': 0, 'shape': [10], 'out': [314], 'sorted_id': 313}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[313] = op;
            
            op->set_inputs( forward_result[312] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1263', 'op': 'aten::add', 'in': [302, 313, 12], 'output_id': 0, 'shape': [10], 'out': [315], 'sorted_id': 314}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[314] = op;
            
            op->set_inputs( forward_result[302] );
            op->set_inputs( forward_result[313] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1264', 'op': 'aten::log', 'in': [314], 'output_id': 0, 'shape': [10], 'out': [316], 'sorted_id': 315}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[315] = op;
            
            op->set_inputs( forward_result[314] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1265', 'op': 'aten::sum', 'in': [315, 20], 'output_id': 0, 'shape': [], 'out': [317], 'sorted_id': 316}
        {
            SumOp* op = new SumOp();
            forward_result[316] = op;
            
            op->set_inputs( forward_result[315] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1266', 'op': 'aten::add', 'in': [291, 316, 12], 'output_id': 0, 'shape': [], 'out': [345], 'sorted_id': 317}
        {
            AddOp* op = new AddOp();
            forward_result[317] = op;
            
            op->set_inputs( forward_result[291] );
            op->set_inputs( forward_result[316] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1267', 'op': 'aten::exp', 'in': [240], 'output_id': 0, 'shape': [10, 400], 'out': [319], 'sorted_id': 318}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[318] = op;
            
            op->set_inputs( forward_result[240] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1268', 'op': 'aten::log1p', 'in': [318], 'output_id': 0, 'shape': [10, 400], 'out': [320], 'sorted_id': 319}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[319] = op;
            
            op->set_inputs( forward_result[318] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1269', 'op': 'aten::log', 'in': [319], 'output_id': 0, 'shape': [10, 400], 'out': [321], 'sorted_id': 320}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[320] = op;
            
            op->set_inputs( forward_result[319] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1270', 'op': 'aten::rsub', 'in': [320, 102, 12], 'output_id': 0, 'shape': [10, 400], 'out': [329], 'sorted_id': 321}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[321] = op;
            
            op->set_inputs( forward_result[320] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1271', 'op': 'aten::sub', 'in': [252, 239, 12], 'output_id': 0, 'shape': [10, 400], 'out': [323], 'sorted_id': 322}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[322] = op;
            
            op->set_inputs( forward_result[252] );
            op->set_inputs( forward_result[239] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1272', 'op': 'aten::pow', 'in': [322, 43], 'output_id': 0, 'shape': [10, 400], 'out': [328], 'sorted_id': 323}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[323] = op;
            
            op->set_inputs( forward_result[322] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1273', 'op': 'aten::exp', 'in': [240], 'output_id': 0, 'shape': [10, 400], 'out': [325], 'sorted_id': 324}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[324] = op;
            
            op->set_inputs( forward_result[240] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1274', 'op': 'aten::log1p', 'in': [324], 'output_id': 0, 'shape': [10, 400], 'out': [326], 'sorted_id': 325}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[325] = op;
            
            op->set_inputs( forward_result[324] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1275', 'op': 'aten::pow', 'in': [325, 43], 'output_id': 0, 'shape': [10, 400], 'out': [327], 'sorted_id': 326}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[326] = op;
            
            op->set_inputs( forward_result[325] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1276', 'op': 'aten::mul', 'in': [326, 48], 'output_id': 0, 'shape': [10, 400], 'out': [328], 'sorted_id': 327}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[327] = op;
            
            op->set_inputs( forward_result[326] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1277', 'op': 'aten::div', 'in': [323, 327], 'output_id': 0, 'shape': [10, 400], 'out': [329], 'sorted_id': 328}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[328] = op;
            
            op->set_inputs( forward_result[323] );
            op->set_inputs( forward_result[327] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1278', 'op': 'aten::sub', 'in': [321, 328, 12], 'output_id': 0, 'shape': [10, 400], 'out': [330], 'sorted_id': 329}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[329] = op;
            
            op->set_inputs( forward_result[321] );
            op->set_inputs( forward_result[328] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1279', 'op': 'aten::sum', 'in': [329, 20], 'output_id': 0, 'shape': [], 'out': [344], 'sorted_id': 330}
        {
            SumOp* op = new SumOp();
            forward_result[330] = op;
            
            op->set_inputs( forward_result[329] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1280', 'op': 'aten::exp', 'in': [254], 'output_id': 0, 'shape': [10], 'out': [332], 'sorted_id': 331}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[331] = op;
            
            op->set_inputs( forward_result[254] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1281', 'op': 'aten::log1p', 'in': [331], 'output_id': 0, 'shape': [10], 'out': [333], 'sorted_id': 332}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[332] = op;
            
            op->set_inputs( forward_result[331] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1282', 'op': 'aten::log', 'in': [332], 'output_id': 0, 'shape': [10], 'out': [334], 'sorted_id': 333}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[333] = op;
            
            op->set_inputs( forward_result[332] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1283', 'op': 'aten::rsub', 'in': [333, 102, 12], 'output_id': 0, 'shape': [10], 'out': [342], 'sorted_id': 334}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[334] = op;
            
            op->set_inputs( forward_result[333] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1284', 'op': 'aten::sub', 'in': [265, 253, 12], 'output_id': 0, 'shape': [10], 'out': [336], 'sorted_id': 335}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[335] = op;
            
            op->set_inputs( forward_result[265] );
            op->set_inputs( forward_result[253] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1285', 'op': 'aten::pow', 'in': [335, 43], 'output_id': 0, 'shape': [10], 'out': [341], 'sorted_id': 336}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[336] = op;
            
            op->set_inputs( forward_result[335] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1286', 'op': 'aten::exp', 'in': [254], 'output_id': 0, 'shape': [10], 'out': [338], 'sorted_id': 337}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[337] = op;
            
            op->set_inputs( forward_result[254] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1287', 'op': 'aten::log1p', 'in': [337], 'output_id': 0, 'shape': [10], 'out': [339], 'sorted_id': 338}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[338] = op;
            
            op->set_inputs( forward_result[337] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1288', 'op': 'aten::pow', 'in': [338, 43], 'output_id': 0, 'shape': [10], 'out': [340], 'sorted_id': 339}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[339] = op;
            
            op->set_inputs( forward_result[338] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1289', 'op': 'aten::mul', 'in': [339, 48], 'output_id': 0, 'shape': [10], 'out': [341], 'sorted_id': 340}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[340] = op;
            
            op->set_inputs( forward_result[339] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1290', 'op': 'aten::div', 'in': [336, 340], 'output_id': 0, 'shape': [10], 'out': [342], 'sorted_id': 341}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[341] = op;
            
            op->set_inputs( forward_result[336] );
            op->set_inputs( forward_result[340] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1291', 'op': 'aten::sub', 'in': [334, 341, 12], 'output_id': 0, 'shape': [10], 'out': [343], 'sorted_id': 342}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[342] = op;
            
            op->set_inputs( forward_result[334] );
            op->set_inputs( forward_result[341] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1292', 'op': 'aten::sum', 'in': [342, 20], 'output_id': 0, 'shape': [], 'out': [344], 'sorted_id': 343}
        {
            SumOp* op = new SumOp();
            forward_result[343] = op;
            
            op->set_inputs( forward_result[342] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/1293', 'op': 'aten::add', 'in': [330, 343, 12], 'output_id': 0, 'shape': [], 'out': [345], 'sorted_id': 344}
        {
            AddOp* op = new AddOp();
            forward_result[344] = op;
            
            op->set_inputs( forward_result[330] );
            op->set_inputs( forward_result[343] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/1295', 'op': 'prim::TupleConstruct', 'in': [266, 317, 344], 'output_id': 0, 'shape': [], 'out': [363, 355, 346], 'sorted_id': 345}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[345] = op;
            
            op->set_inputs( forward_result[266] );
            op->set_inputs( forward_result[317] );
            op->set_inputs( forward_result[344] );
        }
        
        // {'name': 'Model/1296', 'op': 'prim::TupleUnpack', 'in': [345], 'output_id': 0, 'shape': [4, 10], 'out': [347], 'sorted_id': 346}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[346] = op;
            
            op->set_inputs( forward_result[345] );
        }
        
        // {'name': 'Model/Net[net]/1299', 'op': 'aten::log_softmax', 'in': [346, 12, 20], 'output_id': 0, 'shape': [4, 10], 'out': [348], 'sorted_id': 347}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[347] = op;
            
            op->set_inputs( forward_result[346] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/1304', 'op': 'prim::ListConstruct', 'in': [347], 'output_id': 0, 'shape': [], 'out': [349], 'sorted_id': 348}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[348] = op;
            
            op->set_inputs( forward_result[347] );
        }
        
        // {'name': 'Model/Net[net]/outputs', 'op': 'aten::stack', 'in': [348, 10], 'output_id': 0, 'shape': [1, 4, 10], 'out': [351], 'sorted_id': 349}
        {
            Tensor::shape_type shape = {1,4,10};
            StackOp* op = new StackOp();
            forward_result[349] = op;
            
            op->set_inputs( forward_result[348] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/1310', 'op': 'prim::ListConstruct', 'in': [10], 'output_id': 0, 'shape': [], 'out': [351], 'sorted_id': 350}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[350] = op;
            
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/input', 'op': 'aten::mean', 'in': [349, 350, 15, 20], 'output_id': 0, 'shape': [4, 10], 'out': [368], 'sorted_id': 351}
        {
            Tensor::shape_type shape = {4,10};
            MeanOp* op = new MeanOp();
            forward_result[351] = op;
            
            op->set_inputs( forward_result[349] );
            op->set_inputs( forward_result[350] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/1076', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 2, 'shape': [], 'out': [354], 'sorted_id': 352}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[352] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Model/1187', 'op': 'prim::TupleUnpack', 'in': [236], 'output_id': 2, 'shape': [], 'out': [354], 'sorted_id': 353}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[353] = op;
            
            op->set_inputs( forward_result[236] );
        }
        
        // {'name': 'Model/Net[net]/1302', 'op': 'aten::add', 'in': [352, 353, 12], 'output_id': 0, 'shape': [], 'out': [356], 'sorted_id': 354}
        {
            AddOp* op = new AddOp();
            forward_result[354] = op;
            
            op->set_inputs( forward_result[352] );
            op->set_inputs( forward_result[353] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/1298', 'op': 'prim::TupleUnpack', 'in': [345], 'output_id': 2, 'shape': [], 'out': [356], 'sorted_id': 355}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[355] = op;
            
            op->set_inputs( forward_result[345] );
        }
        
        // {'name': 'Model/Net[net]/1303', 'op': 'aten::add', 'in': [354, 355, 12], 'output_id': 0, 'shape': [], 'out': [357], 'sorted_id': 356}
        {
            AddOp* op = new AddOp();
            forward_result[356] = op;
            
            op->set_inputs( forward_result[354] );
            op->set_inputs( forward_result[355] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/1308', 'op': 'prim::ListConstruct', 'in': [356], 'output_id': 0, 'shape': [], 'out': [358], 'sorted_id': 357}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[357] = op;
            
            op->set_inputs( forward_result[356] );
        }
        
        // {'name': 'Model/Net[net]/log_qs', 'op': 'aten::stack', 'in': [357, 10], 'output_id': 0, 'shape': [1], 'out': [359], 'sorted_id': 358}
        {
            Tensor::shape_type shape = {1};
            StackOp* op = new StackOp();
            forward_result[358] = op;
            
            op->set_inputs( forward_result[357] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/log_q', 'op': 'aten::mean', 'in': [358, 20], 'output_id': 0, 'shape': [], 'out': [368], 'sorted_id': 359}
        {
            MeanOp* op = new MeanOp();
            forward_result[359] = op;
            
            op->set_inputs( forward_result[358] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/1075', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 1, 'shape': [], 'out': [362], 'sorted_id': 360}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[360] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Model/1186', 'op': 'prim::TupleUnpack', 'in': [236], 'output_id': 1, 'shape': [], 'out': [362], 'sorted_id': 361}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[361] = op;
            
            op->set_inputs( forward_result[236] );
        }
        
        // {'name': 'Model/Net[net]/1300', 'op': 'aten::add', 'in': [360, 361, 12], 'output_id': 0, 'shape': [], 'out': [364], 'sorted_id': 362}
        {
            AddOp* op = new AddOp();
            forward_result[362] = op;
            
            op->set_inputs( forward_result[360] );
            op->set_inputs( forward_result[361] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/1297', 'op': 'prim::TupleUnpack', 'in': [345], 'output_id': 1, 'shape': [], 'out': [364], 'sorted_id': 363}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[363] = op;
            
            op->set_inputs( forward_result[345] );
        }
        
        // {'name': 'Model/Net[net]/1301', 'op': 'aten::add', 'in': [362, 363, 12], 'output_id': 0, 'shape': [], 'out': [365], 'sorted_id': 364}
        {
            AddOp* op = new AddOp();
            forward_result[364] = op;
            
            op->set_inputs( forward_result[362] );
            op->set_inputs( forward_result[363] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/1306', 'op': 'prim::ListConstruct', 'in': [364], 'output_id': 0, 'shape': [], 'out': [366], 'sorted_id': 365}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[365] = op;
            
            op->set_inputs( forward_result[364] );
        }
        
        // {'name': 'Model/Net[net]/log_ps', 'op': 'aten::stack', 'in': [365, 10], 'output_id': 0, 'shape': [1], 'out': [367], 'sorted_id': 366}
        {
            Tensor::shape_type shape = {1};
            StackOp* op = new StackOp();
            forward_result[366] = op;
            
            op->set_inputs( forward_result[365] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/log_p', 'op': 'aten::mean', 'in': [366, 20], 'output_id': 0, 'shape': [], 'out': [368], 'sorted_id': 367}
        {
            MeanOp* op = new MeanOp();
            forward_result[367] = op;
            
            op->set_inputs( forward_result[366] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/1314', 'op': 'prim::TupleConstruct', 'in': [351, 359, 367], 'output_id': 0, 'shape': [], 'out': [375, 370, 369], 'sorted_id': 368}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[368] = op;
            
            op->set_inputs( forward_result[351] );
            op->set_inputs( forward_result[359] );
            op->set_inputs( forward_result[367] );
        }
        
        // {'name': 'Model/940', 'op': 'prim::TupleUnpack', 'in': [368], 'output_id': 1, 'shape': [], 'out': [372], 'sorted_id': 369}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[369] = op;
            
            op->set_inputs( forward_result[368] );
        }
        
        // {'name': 'Model/941', 'op': 'prim::TupleUnpack', 'in': [368], 'output_id': 2, 'shape': [], 'out': [372], 'sorted_id': 370}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[370] = op;
            
            op->set_inputs( forward_result[368] );
        }
        
        // {'name': 'Model/Loss[loss]/1316', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [372, 381], 'sorted_id': 371}
        {
            Tensor c = (fprec)1.0;
            forward_result[371] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/1322', 'op': 'aten::sub', 'in': [369, 370, 371], 'output_id': 0, 'shape': [], 'out': [374], 'sorted_id': 372}
        {
            SubOp* op = new SubOp();
            forward_result[372] = op;
            
            op->set_inputs( forward_result[369] );
            op->set_inputs( forward_result[370] );
            op->set_inputs( forward_result[371] );
        }
        
        // {'name': 'Model/Loss[loss]/1315', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 15000.0, 'out': [374], 'sorted_id': 373}
        {
            Tensor c = (fprec)15000.0;
            forward_result[373] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/1323', 'op': 'aten::div', 'in': [372, 373], 'output_id': 0, 'shape': [], 'out': [381], 'sorted_id': 374}
        {
            DivOp* op = new DivOp();
            forward_result[374] = op;
            
            op->set_inputs( forward_result[372] );
            op->set_inputs( forward_result[373] );
        }
        
        // {'name': 'Model/939', 'op': 'prim::TupleUnpack', 'in': [368], 'output_id': 0, 'shape': [4, 10], 'out': [380], 'sorted_id': 375}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[375] = op;
            
            op->set_inputs( forward_result[368] );
        }
        
        // {'name': 'Model/Loss[loss]/1320', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [4], 'constant_value': [5.0, 4.0, 9.0, 4.0], 'out': [380], 'sorted_id': 376}
        {
            Tensor::shape_type shape = {4};
            Constant4.reshape( shape );
            forward_result[376] = new VariableTensor( Constant4, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/1319', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [380], 'sorted_id': 377}
        {
            forward_result[377] = NULL;
        }
        
        // {'name': 'Model/Loss[loss]/1318', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [380], 'sorted_id': 378}
        {
            Tensor c = (fprec)2.0;
            forward_result[378] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/1317', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [380], 'sorted_id': 379}
        {
            Tensor c = (fprec)-100.0;
            forward_result[379] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/nll', 'op': 'aten::nll_loss_nd', 'in': [375, 376, 377, 378, 379], 'output_id': 0, 'shape': [], 'out': [381], 'sorted_id': 380}
        {
            NLLLossOp* op = new NLLLossOp();
            forward_result[380] = op;
            
            op->set_inputs( forward_result[375] );
            op->set_inputs( forward_result[376] );
            op->set_inputs( forward_result[377] );
            op->set_inputs( forward_result[378] );
            op->set_inputs( forward_result[379] );
        }
        
        // {'name': 'Model/Loss[loss]/1324', 'op': 'aten::add', 'in': [374, 380, 371], 'output_id': 0, 'shape': [], 'out': [382], 'sorted_id': 381}
        {
            AddOp* op = new AddOp();
            forward_result[381] = op;
            
            op->set_inputs( forward_result[374] );
            op->set_inputs( forward_result[380] );
            op->set_inputs( forward_result[371] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [381], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 382}
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
        vector<MCTNode*> forward_result(383);
    
        // input data
        Tensor::shape_type shape = {4,1,28,28};
        xin.reshape( shape );
        VariableTensor input_var( xin, 3 );
    
        xt::random::seed( 1 );
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 381 );
    #else
        do_train1( forward_result, input_var, 381 );
    #endif
        
        return 0;
    }
    