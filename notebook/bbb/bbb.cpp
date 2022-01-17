
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
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/955', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -1.0, 'out': [3], 'sorted_id': 1}
        {
            Tensor c = (fprec)-1.0;
            forward_result[1] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/956', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 784.0, 'out': [3], 'sorted_id': 2}
        {
            Tensor c = (fprec)784.0;
            forward_result[2] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/976', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [4], 'sorted_id': 3}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.1', 'op': 'aten::view', 'in': [0, 3], 'output_id': 0, 'shape': [4, 784], 'out': [40], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/weight_mu/weight_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [104, 26], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[5] = new VariableTensor( l1_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/weight_rho/weight_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7, 13, 11, 106, 99], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[6] = new VariableTensor( l1_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/990', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/991', 'op': 'aten::log1p', 'in': [7], 'output_id': 0, 'shape': [400, 784], 'out': [25], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/970', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [261, 16, 138, 151, 33, 248], 'sorted_id': 9}
        {
            Tensor c = (fprec)0.0;
            forward_result[9] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/972', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [155, 369, 135, 245, 149, 252, 24, 142, 259, 376, 11, 377, 31, 37, 361, 265], 'sorted_id': 10}
        {
            Tensor c = (fprec)0.0;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/982', 'op': 'aten::size', 'in': [6, 10], 'output_id': 0, 'shape': [], 'out': [14, 18], 'sorted_id': 11}
        {
            SizeOp* op = new SizeOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/971', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [301, 214, 227, 104, 13, 254, 117, 170, 192, 353, 116, 91, 365, 294, 26, 246, 206, 357, 371, 302, 157, 58, 324, 213, 202, 346, 287, 344, 84, 288, 323, 184, 52, 124, 98, 269, 319, 54, 291, 70, 203, 305, 267, 81, 111, 73, 367, 331, 167, 126, 359, 144, 276, 280, 92, 226, 95, 66, 337, 316, 42, 80, 195, 67, 181, 221, 136, 166, 313, 382, 209, 277, 159, 236, 191, 178, 103, 39, 336, 177, 234, 312], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/983', 'op': 'aten::size', 'in': [6, 12], 'output_id': 0, 'shape': [], 'out': [14, 18], 'sorted_id': 13}
        {
            SizeOp* op = new SizeOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/984', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/969', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [261, 263, 151, 265, 33, 250, 142, 140, 138, 153, 37, 24, 16, 35, 19, 155, 378, 252, 248], 'sorted_id': 15}
        {
            Tensor c = (fprec)0.0;
            forward_result[15] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/985', 'op': 'aten::expand', 'in': [9, 14, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/968', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [140, 263, 153, 35, 250, 19], 'sorted_id': 17}
        {
            Tensor c = (fprec)1.0;
            forward_result[17] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/986', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/987', 'op': 'aten::expand', 'in': [17, 18, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/967', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [141, 318, 183, 362, 235, 265, 21, 332, 142, 264, 370, 208, 37, 345, 112, 353, 24, 222, 97, 155, 125, 378, 252, 154, 36, 72, 381, 293, 251], 'sorted_id': 20}
        {
            forward_result[20] = NULL;
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/988', 'op': 'aten::normal', 'in': [16, 19, 20], 'output_id': 0, 'shape': [400, 784], 'out': [24], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/966', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [155, 252, 24, 142, 37, 265], 'sorted_id': 22}
        {
            Tensor c = (fprec)6.0;
            forward_result[22] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/965', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [155, 252, 24, 142, 37, 265], 'sorted_id': 23}
        {
            forward_result[23] = NULL;
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/epsilon.1', 'op': 'aten::to', 'in': [21, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 784], 'out': [25], 'sorted_id': 24}
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
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/992', 'op': 'aten::mul', 'in': [8, 24], 'output_id': 0, 'shape': [400, 784], 'out': [26], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/value.1', 'op': 'aten::add', 'in': [5, 25, 12], 'output_id': 0, 'shape': [400, 784], 'out': [104, 40, 42, 58], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/bias_mu/bias_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [117, 39], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[27] = new VariableTensor( l1_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/bias_rho/bias_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [29, 119, 31, 113], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[28] = new VariableTensor( l1_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1001', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [30], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1002', 'op': 'aten::log1p', 'in': [29], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/994', 'op': 'aten::size', 'in': [28, 10], 'output_id': 0, 'shape': [], 'out': [34, 32], 'sorted_id': 31}
        {
            SizeOp* op = new SizeOp();
            forward_result[31] = op;
            
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/995', 'op': 'prim::ListConstruct', 'in': [31], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/996', 'op': 'aten::expand', 'in': [9, 32, 15], 'output_id': 0, 'shape': [400], 'out': [36], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/997', 'op': 'prim::ListConstruct', 'in': [31], 'output_id': 0, 'shape': [], 'out': [35], 'sorted_id': 34}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/998', 'op': 'aten::expand', 'in': [17, 34, 15], 'output_id': 0, 'shape': [400], 'out': [36], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/999', 'op': 'aten::normal', 'in': [33, 35, 20], 'output_id': 0, 'shape': [400], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/epsilon.3', 'op': 'aten::to', 'in': [36, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 37}
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
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1003', 'op': 'aten::mul', 'in': [30, 37], 'output_id': 0, 'shape': [400], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/value.3', 'op': 'aten::add', 'in': [27, 38, 12], 'output_id': 0, 'shape': [400], 'out': [73, 117, 40, 84], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/input.3', 'op': 'aten::linear', 'in': [4, 26, 39], 'output_id': 0, 'shape': [4, 400], 'out': [127], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/962', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [184, 305, 84, 280, 269, 195, 73, 159, 294, 170, 42, 58], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {1};
            Constant1.reshape( shape );
            forward_result[41] = new VariableTensor( Constant1, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1007', 'op': 'aten::sub', 'in': [26, 41, 12], 'output_id': 0, 'shape': [400, 784], 'out': [44], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/963', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [118, 198, 231, 44, 47, 59, 325, 196, 87, 108, 215, 187, 281, 328, 270, 76, 105, 308, 160, 218, 85, 185, 283, 74, 297, 121, 341, 62, 228, 162, 295, 173, 272, 171, 338, 306, 381], 'sorted_id': 43}
        {
            Tensor c = (fprec)2.0;
            forward_result[43] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1008', 'op': 'aten::pow', 'in': [42, 43], 'output_id': 0, 'shape': [400, 784], 'out': [45], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[42] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1009', 'op': 'aten::neg', 'in': [44], 'output_id': 0, 'shape': [400, 784], 'out': [50], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[45] = op;
            
            op->set_inputs( forward_result[44] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/964', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [79, 162, 275, 187, 300, 190, 76, 51, 165, 272, 297, 47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {1};
            Constant2.reshape( shape );
            forward_result[46] = new VariableTensor( Constant2, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.1', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [49], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/961', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [77, 88, 174, 273, 309, 49, 63, 109, 219, 329, 188, 284, 232, 122, 342, 163, 199, 298], 'sorted_id': 48}
        {
            Tensor c = (fprec)2.0;
            forward_result[48] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1010', 'op': 'aten::mul', 'in': [47, 48], 'output_id': 0, 'shape': [1], 'out': [50], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1011', 'op': 'aten::div', 'in': [45, 49], 'output_id': 0, 'shape': [400, 784], 'out': [52], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.1', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1012', 'op': 'aten::sub', 'in': [50, 51, 12], 'output_id': 0, 'shape': [400, 784], 'out': [54], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/960', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.9189, 'out': [92, 178, 81, 67, 288, 277, 302, 54, 313, 167, 192, 203], 'sorted_id': 53}
        {
            Tensor c = (fprec)0.9189;
            forward_result[53] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1013', 'op': 'aten::sub', 'in': [52, 53, 12], 'output_id': 0, 'shape': [400, 784], 'out': [55], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob1.1', 'op': 'aten::exp', 'in': [54], 'output_id': 0, 'shape': [400, 784], 'out': [57], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/958', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [94, 83, 69, 290, 169, 205, 304, 315, 194, 57, 180, 279], 'sorted_id': 56}
        {
            Tensor c = (fprec)0.5;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1025', 'op': 'aten::mul', 'in': [55, 56], 'output_id': 0, 'shape': [400, 784], 'out': [70], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1017', 'op': 'aten::sub', 'in': [26, 41, 12], 'output_id': 0, 'shape': [400, 784], 'out': [59], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[58] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1018', 'op': 'aten::pow', 'in': [58, 43], 'output_id': 0, 'shape': [400, 784], 'out': [60], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1019', 'op': 'aten::neg', 'in': [59], 'output_id': 0, 'shape': [400, 784], 'out': [64], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[60] = op;
            
            op->set_inputs( forward_result[59] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/959', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [176, 198, 286, 283, 173, 90, 65, 308, 62, 201, 87, 311], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {1};
            Constant3.reshape( shape );
            forward_result[61] = new VariableTensor( Constant3, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.3', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1020', 'op': 'aten::mul', 'in': [62, 48], 'output_id': 0, 'shape': [1], 'out': [64], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[62] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1021', 'op': 'aten::div', 'in': [60, 63], 'output_id': 0, 'shape': [400, 784], 'out': [66], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.3', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1022', 'op': 'aten::sub', 'in': [64, 65, 12], 'output_id': 0, 'shape': [400, 784], 'out': [67], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1023', 'op': 'aten::sub', 'in': [66, 53, 12], 'output_id': 0, 'shape': [400, 784], 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob2.1', 'op': 'aten::exp', 'in': [67], 'output_id': 0, 'shape': [400, 784], 'out': [69], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[67] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1026', 'op': 'aten::mul', 'in': [68, 56], 'output_id': 0, 'shape': [400, 784], 'out': [70], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[68] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1027', 'op': 'aten::add', 'in': [57, 69, 12], 'output_id': 0, 'shape': [400, 784], 'out': [71], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[70] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1028', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [400, 784], 'out': [72], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1029', 'op': 'aten::sum', 'in': [71, 20], 'output_id': 0, 'shape': [], 'out': [98], 'sorted_id': 72}
        {
            SumOp* op = new SumOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1032', 'op': 'aten::sub', 'in': [39, 41, 12], 'output_id': 0, 'shape': [400], 'out': [74], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1033', 'op': 'aten::pow', 'in': [73, 43], 'output_id': 0, 'shape': [400], 'out': [75], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1034', 'op': 'aten::neg', 'in': [74], 'output_id': 0, 'shape': [400], 'out': [78], 'sorted_id': 75}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[75] = op;
            
            op->set_inputs( forward_result[74] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.5', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [77], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1035', 'op': 'aten::mul', 'in': [76, 48], 'output_id': 0, 'shape': [1], 'out': [78], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1036', 'op': 'aten::div', 'in': [75, 77], 'output_id': 0, 'shape': [400], 'out': [80], 'sorted_id': 78}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[78] = op;
            
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[77] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.5', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [80], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1037', 'op': 'aten::sub', 'in': [78, 79, 12], 'output_id': 0, 'shape': [400], 'out': [81], 'sorted_id': 80}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[80] = op;
            
            op->set_inputs( forward_result[78] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1038', 'op': 'aten::sub', 'in': [80, 53, 12], 'output_id': 0, 'shape': [400], 'out': [82], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[80] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob1.3', 'op': 'aten::exp', 'in': [81], 'output_id': 0, 'shape': [400], 'out': [83], 'sorted_id': 82}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[82] = op;
            
            op->set_inputs( forward_result[81] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1050', 'op': 'aten::mul', 'in': [82, 56], 'output_id': 0, 'shape': [400], 'out': [95], 'sorted_id': 83}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[83] = op;
            
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1042', 'op': 'aten::sub', 'in': [39, 41, 12], 'output_id': 0, 'shape': [400], 'out': [85], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1043', 'op': 'aten::pow', 'in': [84, 43], 'output_id': 0, 'shape': [400], 'out': [86], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1044', 'op': 'aten::neg', 'in': [85], 'output_id': 0, 'shape': [400], 'out': [89], 'sorted_id': 86}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[86] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.7', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [88], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1045', 'op': 'aten::mul', 'in': [87, 48], 'output_id': 0, 'shape': [1], 'out': [89], 'sorted_id': 88}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[88] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1046', 'op': 'aten::div', 'in': [86, 88], 'output_id': 0, 'shape': [400], 'out': [91], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[86] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.7', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [91], 'sorted_id': 90}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[90] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1047', 'op': 'aten::sub', 'in': [89, 90, 12], 'output_id': 0, 'shape': [400], 'out': [92], 'sorted_id': 91}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[91] = op;
            
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[90] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1048', 'op': 'aten::sub', 'in': [91, 53, 12], 'output_id': 0, 'shape': [400], 'out': [93], 'sorted_id': 92}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[92] = op;
            
            op->set_inputs( forward_result[91] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob2.3', 'op': 'aten::exp', 'in': [92], 'output_id': 0, 'shape': [400], 'out': [94], 'sorted_id': 93}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[92] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1051', 'op': 'aten::mul', 'in': [93, 56], 'output_id': 0, 'shape': [400], 'out': [95], 'sorted_id': 94}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[94] = op;
            
            op->set_inputs( forward_result[93] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1052', 'op': 'aten::add', 'in': [83, 94, 12], 'output_id': 0, 'shape': [400], 'out': [96], 'sorted_id': 95}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[95] = op;
            
            op->set_inputs( forward_result[83] );
            op->set_inputs( forward_result[94] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1053', 'op': 'aten::log', 'in': [95], 'output_id': 0, 'shape': [400], 'out': [97], 'sorted_id': 96}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[96] = op;
            
            op->set_inputs( forward_result[95] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1054', 'op': 'aten::sum', 'in': [96, 20], 'output_id': 0, 'shape': [], 'out': [98], 'sorted_id': 97}
        {
            SumOp* op = new SumOp();
            forward_result[97] = op;
            
            op->set_inputs( forward_result[96] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1055', 'op': 'aten::add', 'in': [72, 97, 12], 'output_id': 0, 'shape': [], 'out': [127], 'sorted_id': 98}
        {
            AddOp* op = new AddOp();
            forward_result[98] = op;
            
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[97] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1056', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [100], 'sorted_id': 99}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[99] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1057', 'op': 'aten::log1p', 'in': [99], 'output_id': 0, 'shape': [400, 784], 'out': [101], 'sorted_id': 100}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[100] = op;
            
            op->set_inputs( forward_result[99] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1058', 'op': 'aten::log', 'in': [100], 'output_id': 0, 'shape': [400, 784], 'out': [103], 'sorted_id': 101}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[101] = op;
            
            op->set_inputs( forward_result[100] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/957', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -0.9189385332046727, 'out': [226, 116, 103, 336, 213, 323], 'sorted_id': 102}
        {
            Tensor c = (fprec)-0.9189385332046727;
            forward_result[102] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1059', 'op': 'aten::rsub', 'in': [101, 102, 12], 'output_id': 0, 'shape': [400, 784], 'out': [111], 'sorted_id': 103}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[103] = op;
            
            op->set_inputs( forward_result[101] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1060', 'op': 'aten::sub', 'in': [26, 5, 12], 'output_id': 0, 'shape': [400, 784], 'out': [105], 'sorted_id': 104}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[104] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1061', 'op': 'aten::pow', 'in': [104, 43], 'output_id': 0, 'shape': [400, 784], 'out': [110], 'sorted_id': 105}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[105] = op;
            
            op->set_inputs( forward_result[104] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1062', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [107], 'sorted_id': 106}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[106] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1063', 'op': 'aten::log1p', 'in': [106], 'output_id': 0, 'shape': [400, 784], 'out': [108], 'sorted_id': 107}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[107] = op;
            
            op->set_inputs( forward_result[106] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1064', 'op': 'aten::pow', 'in': [107, 43], 'output_id': 0, 'shape': [400, 784], 'out': [109], 'sorted_id': 108}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[108] = op;
            
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1065', 'op': 'aten::mul', 'in': [108, 48], 'output_id': 0, 'shape': [400, 784], 'out': [110], 'sorted_id': 109}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[109] = op;
            
            op->set_inputs( forward_result[108] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1066', 'op': 'aten::div', 'in': [105, 109], 'output_id': 0, 'shape': [400, 784], 'out': [111], 'sorted_id': 110}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[110] = op;
            
            op->set_inputs( forward_result[105] );
            op->set_inputs( forward_result[109] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1067', 'op': 'aten::sub', 'in': [103, 110, 12], 'output_id': 0, 'shape': [400, 784], 'out': [112], 'sorted_id': 111}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[111] = op;
            
            op->set_inputs( forward_result[103] );
            op->set_inputs( forward_result[110] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1068', 'op': 'aten::sum', 'in': [111, 20], 'output_id': 0, 'shape': [], 'out': [126], 'sorted_id': 112}
        {
            SumOp* op = new SumOp();
            forward_result[112] = op;
            
            op->set_inputs( forward_result[111] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1069', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [114], 'sorted_id': 113}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[113] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1070', 'op': 'aten::log1p', 'in': [113], 'output_id': 0, 'shape': [400], 'out': [115], 'sorted_id': 114}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[114] = op;
            
            op->set_inputs( forward_result[113] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1071', 'op': 'aten::log', 'in': [114], 'output_id': 0, 'shape': [400], 'out': [116], 'sorted_id': 115}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[115] = op;
            
            op->set_inputs( forward_result[114] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1072', 'op': 'aten::rsub', 'in': [115, 102, 12], 'output_id': 0, 'shape': [400], 'out': [124], 'sorted_id': 116}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[116] = op;
            
            op->set_inputs( forward_result[115] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1073', 'op': 'aten::sub', 'in': [39, 27, 12], 'output_id': 0, 'shape': [400], 'out': [118], 'sorted_id': 117}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[117] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1074', 'op': 'aten::pow', 'in': [117, 43], 'output_id': 0, 'shape': [400], 'out': [123], 'sorted_id': 118}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[118] = op;
            
            op->set_inputs( forward_result[117] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1075', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [120], 'sorted_id': 119}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[119] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1076', 'op': 'aten::log1p', 'in': [119], 'output_id': 0, 'shape': [400], 'out': [121], 'sorted_id': 120}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[120] = op;
            
            op->set_inputs( forward_result[119] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1077', 'op': 'aten::pow', 'in': [120, 43], 'output_id': 0, 'shape': [400], 'out': [122], 'sorted_id': 121}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[121] = op;
            
            op->set_inputs( forward_result[120] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1078', 'op': 'aten::mul', 'in': [121, 48], 'output_id': 0, 'shape': [400], 'out': [123], 'sorted_id': 122}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[122] = op;
            
            op->set_inputs( forward_result[121] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1079', 'op': 'aten::div', 'in': [118, 122], 'output_id': 0, 'shape': [400], 'out': [124], 'sorted_id': 123}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[123] = op;
            
            op->set_inputs( forward_result[118] );
            op->set_inputs( forward_result[122] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1080', 'op': 'aten::sub', 'in': [116, 123, 12], 'output_id': 0, 'shape': [400], 'out': [125], 'sorted_id': 124}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[124] = op;
            
            op->set_inputs( forward_result[116] );
            op->set_inputs( forward_result[123] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1081', 'op': 'aten::sum', 'in': [124, 20], 'output_id': 0, 'shape': [], 'out': [126], 'sorted_id': 125}
        {
            SumOp* op = new SumOp();
            forward_result[125] = op;
            
            op->set_inputs( forward_result[124] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1082', 'op': 'aten::add', 'in': [112, 125, 12], 'output_id': 0, 'shape': [], 'out': [127], 'sorted_id': 126}
        {
            AddOp* op = new AddOp();
            forward_result[126] = op;
            
            op->set_inputs( forward_result[112] );
            op->set_inputs( forward_result[125] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/1084', 'op': 'prim::TupleConstruct', 'in': [40, 98, 126], 'output_id': 0, 'shape': [], 'out': [349, 128, 129], 'sorted_id': 127}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[127] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[98] );
            op->set_inputs( forward_result[126] );
        }
        
        // {'name': 'Net/1086', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 1, 'shape': [], 'out': [354], 'sorted_id': 128}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[128] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Net/1085', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 0, 'shape': [4, 400], 'out': [130], 'sorted_id': 129}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[129] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.5', 'op': 'aten::relu', 'in': [129], 'output_id': 0, 'shape': [4, 400], 'out': [158], 'sorted_id': 130}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[130] = op;
            
            op->set_inputs( forward_result[129] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/weight_mu/weight_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [144, 214], 'sorted_id': 131}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[131] = new VariableTensor( l2_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/weight_rho/weight_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [216, 135, 210, 136, 133], 'sorted_id': 132}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[132] = new VariableTensor( l2_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1101', 'op': 'aten::exp', 'in': [132], 'output_id': 0, 'shape': [400, 400], 'out': [134], 'sorted_id': 133}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[133] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1102', 'op': 'aten::log1p', 'in': [133], 'output_id': 0, 'shape': [400, 400], 'out': [143], 'sorted_id': 134}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[134] = op;
            
            op->set_inputs( forward_result[133] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1093', 'op': 'aten::size', 'in': [132, 10], 'output_id': 0, 'shape': [], 'out': [139, 137], 'sorted_id': 135}
        {
            SizeOp* op = new SizeOp();
            forward_result[135] = op;
            
            op->set_inputs( forward_result[132] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1094', 'op': 'aten::size', 'in': [132, 12], 'output_id': 0, 'shape': [], 'out': [139, 137], 'sorted_id': 136}
        {
            SizeOp* op = new SizeOp();
            forward_result[136] = op;
            
            op->set_inputs( forward_result[132] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1095', 'op': 'prim::ListConstruct', 'in': [135, 136], 'output_id': 0, 'shape': [], 'out': [138], 'sorted_id': 137}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[137] = op;
            
            op->set_inputs( forward_result[135] );
            op->set_inputs( forward_result[136] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1096', 'op': 'aten::expand', 'in': [9, 137, 15], 'output_id': 0, 'shape': [400, 400], 'out': [141], 'sorted_id': 138}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[138] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[137] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1097', 'op': 'prim::ListConstruct', 'in': [135, 136], 'output_id': 0, 'shape': [], 'out': [140], 'sorted_id': 139}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[139] = op;
            
            op->set_inputs( forward_result[135] );
            op->set_inputs( forward_result[136] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1098', 'op': 'aten::expand', 'in': [17, 139, 15], 'output_id': 0, 'shape': [400, 400], 'out': [141], 'sorted_id': 140}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[140] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[139] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1099', 'op': 'aten::normal', 'in': [138, 140, 20], 'output_id': 0, 'shape': [400, 400], 'out': [142], 'sorted_id': 141}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[141] = op;
            
            op->set_inputs( forward_result[138] );
            op->set_inputs( forward_result[140] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/epsilon.5', 'op': 'aten::to', 'in': [141, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 400], 'out': [143], 'sorted_id': 142}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[142] = op;
            
            op->set_inputs( forward_result[141] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1103', 'op': 'aten::mul', 'in': [134, 142], 'output_id': 0, 'shape': [400, 400], 'out': [144], 'sorted_id': 143}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[143] = op;
            
            op->set_inputs( forward_result[134] );
            op->set_inputs( forward_result[142] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/value.5', 'op': 'aten::add', 'in': [131, 143, 12], 'output_id': 0, 'shape': [400, 400], 'out': [159, 170, 158, 214], 'sorted_id': 144}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[144] = op;
            
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/bias_mu/bias_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [157, 227], 'sorted_id': 145}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[145] = new VariableTensor( l2_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/bias_rho/bias_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [149, 223, 147, 229], 'sorted_id': 146}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[146] = new VariableTensor( l2_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1112', 'op': 'aten::exp', 'in': [146], 'output_id': 0, 'shape': [400], 'out': [148], 'sorted_id': 147}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[147] = op;
            
            op->set_inputs( forward_result[146] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1113', 'op': 'aten::log1p', 'in': [147], 'output_id': 0, 'shape': [400], 'out': [156], 'sorted_id': 148}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[148] = op;
            
            op->set_inputs( forward_result[147] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1105', 'op': 'aten::size', 'in': [146, 10], 'output_id': 0, 'shape': [], 'out': [150, 152], 'sorted_id': 149}
        {
            SizeOp* op = new SizeOp();
            forward_result[149] = op;
            
            op->set_inputs( forward_result[146] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1106', 'op': 'prim::ListConstruct', 'in': [149], 'output_id': 0, 'shape': [], 'out': [151], 'sorted_id': 150}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[150] = op;
            
            op->set_inputs( forward_result[149] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1107', 'op': 'aten::expand', 'in': [9, 150, 15], 'output_id': 0, 'shape': [400], 'out': [154], 'sorted_id': 151}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[151] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1108', 'op': 'prim::ListConstruct', 'in': [149], 'output_id': 0, 'shape': [], 'out': [153], 'sorted_id': 152}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[152] = op;
            
            op->set_inputs( forward_result[149] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1109', 'op': 'aten::expand', 'in': [17, 152, 15], 'output_id': 0, 'shape': [400], 'out': [154], 'sorted_id': 153}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[153] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[152] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1110', 'op': 'aten::normal', 'in': [151, 153, 20], 'output_id': 0, 'shape': [400], 'out': [155], 'sorted_id': 154}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[154] = op;
            
            op->set_inputs( forward_result[151] );
            op->set_inputs( forward_result[153] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/epsilon.7', 'op': 'aten::to', 'in': [154, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [156], 'sorted_id': 155}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[155] = op;
            
            op->set_inputs( forward_result[154] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1114', 'op': 'aten::mul', 'in': [148, 155], 'output_id': 0, 'shape': [400], 'out': [157], 'sorted_id': 156}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[156] = op;
            
            op->set_inputs( forward_result[148] );
            op->set_inputs( forward_result[155] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/value.7', 'op': 'aten::add', 'in': [145, 156, 12], 'output_id': 0, 'shape': [400], 'out': [184, 195, 158, 227], 'sorted_id': 157}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[157] = op;
            
            op->set_inputs( forward_result[145] );
            op->set_inputs( forward_result[156] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/input.7', 'op': 'aten::linear', 'in': [130, 144, 157], 'output_id': 0, 'shape': [4, 400], 'out': [237], 'sorted_id': 158}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[158] = op;
            
            op->set_inputs( forward_result[130] );
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[157] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1118', 'op': 'aten::sub', 'in': [144, 41, 12], 'output_id': 0, 'shape': [400, 400], 'out': [160], 'sorted_id': 159}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[159] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1119', 'op': 'aten::pow', 'in': [159, 43], 'output_id': 0, 'shape': [400, 400], 'out': [161], 'sorted_id': 160}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[160] = op;
            
            op->set_inputs( forward_result[159] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1120', 'op': 'aten::neg', 'in': [160], 'output_id': 0, 'shape': [400, 400], 'out': [164], 'sorted_id': 161}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[161] = op;
            
            op->set_inputs( forward_result[160] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.9', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [163], 'sorted_id': 162}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[162] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1121', 'op': 'aten::mul', 'in': [162, 48], 'output_id': 0, 'shape': [1], 'out': [164], 'sorted_id': 163}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[163] = op;
            
            op->set_inputs( forward_result[162] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1122', 'op': 'aten::div', 'in': [161, 163], 'output_id': 0, 'shape': [400, 400], 'out': [166], 'sorted_id': 164}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[164] = op;
            
            op->set_inputs( forward_result[161] );
            op->set_inputs( forward_result[163] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.9', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [166], 'sorted_id': 165}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[165] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1123', 'op': 'aten::sub', 'in': [164, 165, 12], 'output_id': 0, 'shape': [400, 400], 'out': [167], 'sorted_id': 166}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[166] = op;
            
            op->set_inputs( forward_result[164] );
            op->set_inputs( forward_result[165] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1124', 'op': 'aten::sub', 'in': [166, 53, 12], 'output_id': 0, 'shape': [400, 400], 'out': [168], 'sorted_id': 167}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[167] = op;
            
            op->set_inputs( forward_result[166] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob1.5', 'op': 'aten::exp', 'in': [167], 'output_id': 0, 'shape': [400, 400], 'out': [169], 'sorted_id': 168}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[168] = op;
            
            op->set_inputs( forward_result[167] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1136', 'op': 'aten::mul', 'in': [168, 56], 'output_id': 0, 'shape': [400, 400], 'out': [181], 'sorted_id': 169}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[169] = op;
            
            op->set_inputs( forward_result[168] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1128', 'op': 'aten::sub', 'in': [144, 41, 12], 'output_id': 0, 'shape': [400, 400], 'out': [171], 'sorted_id': 170}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[170] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1129', 'op': 'aten::pow', 'in': [170, 43], 'output_id': 0, 'shape': [400, 400], 'out': [172], 'sorted_id': 171}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[171] = op;
            
            op->set_inputs( forward_result[170] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1130', 'op': 'aten::neg', 'in': [171], 'output_id': 0, 'shape': [400, 400], 'out': [175], 'sorted_id': 172}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[172] = op;
            
            op->set_inputs( forward_result[171] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.11', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [174], 'sorted_id': 173}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[173] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1131', 'op': 'aten::mul', 'in': [173, 48], 'output_id': 0, 'shape': [1], 'out': [175], 'sorted_id': 174}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[174] = op;
            
            op->set_inputs( forward_result[173] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1132', 'op': 'aten::div', 'in': [172, 174], 'output_id': 0, 'shape': [400, 400], 'out': [177], 'sorted_id': 175}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[175] = op;
            
            op->set_inputs( forward_result[172] );
            op->set_inputs( forward_result[174] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.11', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [177], 'sorted_id': 176}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[176] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1133', 'op': 'aten::sub', 'in': [175, 176, 12], 'output_id': 0, 'shape': [400, 400], 'out': [178], 'sorted_id': 177}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[177] = op;
            
            op->set_inputs( forward_result[175] );
            op->set_inputs( forward_result[176] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1134', 'op': 'aten::sub', 'in': [177, 53, 12], 'output_id': 0, 'shape': [400, 400], 'out': [179], 'sorted_id': 178}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[178] = op;
            
            op->set_inputs( forward_result[177] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob2.5', 'op': 'aten::exp', 'in': [178], 'output_id': 0, 'shape': [400, 400], 'out': [180], 'sorted_id': 179}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[179] = op;
            
            op->set_inputs( forward_result[178] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1137', 'op': 'aten::mul', 'in': [179, 56], 'output_id': 0, 'shape': [400, 400], 'out': [181], 'sorted_id': 180}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[180] = op;
            
            op->set_inputs( forward_result[179] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1138', 'op': 'aten::add', 'in': [169, 180, 12], 'output_id': 0, 'shape': [400, 400], 'out': [182], 'sorted_id': 181}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[181] = op;
            
            op->set_inputs( forward_result[169] );
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1139', 'op': 'aten::log', 'in': [181], 'output_id': 0, 'shape': [400, 400], 'out': [183], 'sorted_id': 182}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[182] = op;
            
            op->set_inputs( forward_result[181] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1140', 'op': 'aten::sum', 'in': [182, 20], 'output_id': 0, 'shape': [], 'out': [209], 'sorted_id': 183}
        {
            SumOp* op = new SumOp();
            forward_result[183] = op;
            
            op->set_inputs( forward_result[182] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1143', 'op': 'aten::sub', 'in': [157, 41, 12], 'output_id': 0, 'shape': [400], 'out': [185], 'sorted_id': 184}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[184] = op;
            
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1144', 'op': 'aten::pow', 'in': [184, 43], 'output_id': 0, 'shape': [400], 'out': [186], 'sorted_id': 185}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[185] = op;
            
            op->set_inputs( forward_result[184] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1145', 'op': 'aten::neg', 'in': [185], 'output_id': 0, 'shape': [400], 'out': [189], 'sorted_id': 186}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[186] = op;
            
            op->set_inputs( forward_result[185] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.13', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [188], 'sorted_id': 187}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[187] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1146', 'op': 'aten::mul', 'in': [187, 48], 'output_id': 0, 'shape': [1], 'out': [189], 'sorted_id': 188}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[188] = op;
            
            op->set_inputs( forward_result[187] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1147', 'op': 'aten::div', 'in': [186, 188], 'output_id': 0, 'shape': [400], 'out': [191], 'sorted_id': 189}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[189] = op;
            
            op->set_inputs( forward_result[186] );
            op->set_inputs( forward_result[188] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.13', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [191], 'sorted_id': 190}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[190] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1148', 'op': 'aten::sub', 'in': [189, 190, 12], 'output_id': 0, 'shape': [400], 'out': [192], 'sorted_id': 191}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[191] = op;
            
            op->set_inputs( forward_result[189] );
            op->set_inputs( forward_result[190] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1149', 'op': 'aten::sub', 'in': [191, 53, 12], 'output_id': 0, 'shape': [400], 'out': [193], 'sorted_id': 192}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[192] = op;
            
            op->set_inputs( forward_result[191] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob1.7', 'op': 'aten::exp', 'in': [192], 'output_id': 0, 'shape': [400], 'out': [194], 'sorted_id': 193}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[193] = op;
            
            op->set_inputs( forward_result[192] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1161', 'op': 'aten::mul', 'in': [193, 56], 'output_id': 0, 'shape': [400], 'out': [206], 'sorted_id': 194}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[194] = op;
            
            op->set_inputs( forward_result[193] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1153', 'op': 'aten::sub', 'in': [157, 41, 12], 'output_id': 0, 'shape': [400], 'out': [196], 'sorted_id': 195}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[195] = op;
            
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1154', 'op': 'aten::pow', 'in': [195, 43], 'output_id': 0, 'shape': [400], 'out': [197], 'sorted_id': 196}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[196] = op;
            
            op->set_inputs( forward_result[195] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1155', 'op': 'aten::neg', 'in': [196], 'output_id': 0, 'shape': [400], 'out': [200], 'sorted_id': 197}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[197] = op;
            
            op->set_inputs( forward_result[196] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.15', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [199], 'sorted_id': 198}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[198] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1156', 'op': 'aten::mul', 'in': [198, 48], 'output_id': 0, 'shape': [1], 'out': [200], 'sorted_id': 199}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[199] = op;
            
            op->set_inputs( forward_result[198] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1157', 'op': 'aten::div', 'in': [197, 199], 'output_id': 0, 'shape': [400], 'out': [202], 'sorted_id': 200}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[200] = op;
            
            op->set_inputs( forward_result[197] );
            op->set_inputs( forward_result[199] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.15', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [202], 'sorted_id': 201}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[201] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1158', 'op': 'aten::sub', 'in': [200, 201, 12], 'output_id': 0, 'shape': [400], 'out': [203], 'sorted_id': 202}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[202] = op;
            
            op->set_inputs( forward_result[200] );
            op->set_inputs( forward_result[201] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1159', 'op': 'aten::sub', 'in': [202, 53, 12], 'output_id': 0, 'shape': [400], 'out': [204], 'sorted_id': 203}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[203] = op;
            
            op->set_inputs( forward_result[202] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob2.7', 'op': 'aten::exp', 'in': [203], 'output_id': 0, 'shape': [400], 'out': [205], 'sorted_id': 204}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[204] = op;
            
            op->set_inputs( forward_result[203] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1162', 'op': 'aten::mul', 'in': [204, 56], 'output_id': 0, 'shape': [400], 'out': [206], 'sorted_id': 205}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[205] = op;
            
            op->set_inputs( forward_result[204] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1163', 'op': 'aten::add', 'in': [194, 205, 12], 'output_id': 0, 'shape': [400], 'out': [207], 'sorted_id': 206}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[206] = op;
            
            op->set_inputs( forward_result[194] );
            op->set_inputs( forward_result[205] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1164', 'op': 'aten::log', 'in': [206], 'output_id': 0, 'shape': [400], 'out': [208], 'sorted_id': 207}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[207] = op;
            
            op->set_inputs( forward_result[206] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1165', 'op': 'aten::sum', 'in': [207, 20], 'output_id': 0, 'shape': [], 'out': [209], 'sorted_id': 208}
        {
            SumOp* op = new SumOp();
            forward_result[208] = op;
            
            op->set_inputs( forward_result[207] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1166', 'op': 'aten::add', 'in': [183, 208, 12], 'output_id': 0, 'shape': [], 'out': [237], 'sorted_id': 209}
        {
            AddOp* op = new AddOp();
            forward_result[209] = op;
            
            op->set_inputs( forward_result[183] );
            op->set_inputs( forward_result[208] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1167', 'op': 'aten::exp', 'in': [132], 'output_id': 0, 'shape': [400, 400], 'out': [211], 'sorted_id': 210}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[210] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1168', 'op': 'aten::log1p', 'in': [210], 'output_id': 0, 'shape': [400, 400], 'out': [212], 'sorted_id': 211}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[211] = op;
            
            op->set_inputs( forward_result[210] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1169', 'op': 'aten::log', 'in': [211], 'output_id': 0, 'shape': [400, 400], 'out': [213], 'sorted_id': 212}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[212] = op;
            
            op->set_inputs( forward_result[211] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1170', 'op': 'aten::rsub', 'in': [212, 102, 12], 'output_id': 0, 'shape': [400, 400], 'out': [221], 'sorted_id': 213}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[213] = op;
            
            op->set_inputs( forward_result[212] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1171', 'op': 'aten::sub', 'in': [144, 131, 12], 'output_id': 0, 'shape': [400, 400], 'out': [215], 'sorted_id': 214}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[214] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1172', 'op': 'aten::pow', 'in': [214, 43], 'output_id': 0, 'shape': [400, 400], 'out': [220], 'sorted_id': 215}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[215] = op;
            
            op->set_inputs( forward_result[214] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1173', 'op': 'aten::exp', 'in': [132], 'output_id': 0, 'shape': [400, 400], 'out': [217], 'sorted_id': 216}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[216] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1174', 'op': 'aten::log1p', 'in': [216], 'output_id': 0, 'shape': [400, 400], 'out': [218], 'sorted_id': 217}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[217] = op;
            
            op->set_inputs( forward_result[216] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1175', 'op': 'aten::pow', 'in': [217, 43], 'output_id': 0, 'shape': [400, 400], 'out': [219], 'sorted_id': 218}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[218] = op;
            
            op->set_inputs( forward_result[217] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1176', 'op': 'aten::mul', 'in': [218, 48], 'output_id': 0, 'shape': [400, 400], 'out': [220], 'sorted_id': 219}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[219] = op;
            
            op->set_inputs( forward_result[218] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1177', 'op': 'aten::div', 'in': [215, 219], 'output_id': 0, 'shape': [400, 400], 'out': [221], 'sorted_id': 220}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[220] = op;
            
            op->set_inputs( forward_result[215] );
            op->set_inputs( forward_result[219] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1178', 'op': 'aten::sub', 'in': [213, 220, 12], 'output_id': 0, 'shape': [400, 400], 'out': [222], 'sorted_id': 221}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[221] = op;
            
            op->set_inputs( forward_result[213] );
            op->set_inputs( forward_result[220] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1179', 'op': 'aten::sum', 'in': [221, 20], 'output_id': 0, 'shape': [], 'out': [236], 'sorted_id': 222}
        {
            SumOp* op = new SumOp();
            forward_result[222] = op;
            
            op->set_inputs( forward_result[221] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1180', 'op': 'aten::exp', 'in': [146], 'output_id': 0, 'shape': [400], 'out': [224], 'sorted_id': 223}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[223] = op;
            
            op->set_inputs( forward_result[146] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1181', 'op': 'aten::log1p', 'in': [223], 'output_id': 0, 'shape': [400], 'out': [225], 'sorted_id': 224}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[224] = op;
            
            op->set_inputs( forward_result[223] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1182', 'op': 'aten::log', 'in': [224], 'output_id': 0, 'shape': [400], 'out': [226], 'sorted_id': 225}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[225] = op;
            
            op->set_inputs( forward_result[224] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1183', 'op': 'aten::rsub', 'in': [225, 102, 12], 'output_id': 0, 'shape': [400], 'out': [234], 'sorted_id': 226}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[226] = op;
            
            op->set_inputs( forward_result[225] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1184', 'op': 'aten::sub', 'in': [157, 145, 12], 'output_id': 0, 'shape': [400], 'out': [228], 'sorted_id': 227}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[227] = op;
            
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[145] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1185', 'op': 'aten::pow', 'in': [227, 43], 'output_id': 0, 'shape': [400], 'out': [233], 'sorted_id': 228}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[228] = op;
            
            op->set_inputs( forward_result[227] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1186', 'op': 'aten::exp', 'in': [146], 'output_id': 0, 'shape': [400], 'out': [230], 'sorted_id': 229}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[229] = op;
            
            op->set_inputs( forward_result[146] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1187', 'op': 'aten::log1p', 'in': [229], 'output_id': 0, 'shape': [400], 'out': [231], 'sorted_id': 230}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[230] = op;
            
            op->set_inputs( forward_result[229] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1188', 'op': 'aten::pow', 'in': [230, 43], 'output_id': 0, 'shape': [400], 'out': [232], 'sorted_id': 231}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[231] = op;
            
            op->set_inputs( forward_result[230] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1189', 'op': 'aten::mul', 'in': [231, 48], 'output_id': 0, 'shape': [400], 'out': [233], 'sorted_id': 232}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[232] = op;
            
            op->set_inputs( forward_result[231] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1190', 'op': 'aten::div', 'in': [228, 232], 'output_id': 0, 'shape': [400], 'out': [234], 'sorted_id': 233}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[233] = op;
            
            op->set_inputs( forward_result[228] );
            op->set_inputs( forward_result[232] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1191', 'op': 'aten::sub', 'in': [226, 233, 12], 'output_id': 0, 'shape': [400], 'out': [235], 'sorted_id': 234}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[234] = op;
            
            op->set_inputs( forward_result[226] );
            op->set_inputs( forward_result[233] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1192', 'op': 'aten::sum', 'in': [234, 20], 'output_id': 0, 'shape': [], 'out': [236], 'sorted_id': 235}
        {
            SumOp* op = new SumOp();
            forward_result[235] = op;
            
            op->set_inputs( forward_result[234] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1193', 'op': 'aten::add', 'in': [222, 235, 12], 'output_id': 0, 'shape': [], 'out': [237], 'sorted_id': 236}
        {
            AddOp* op = new AddOp();
            forward_result[236] = op;
            
            op->set_inputs( forward_result[222] );
            op->set_inputs( forward_result[235] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/1195', 'op': 'prim::TupleConstruct', 'in': [158, 209, 236], 'output_id': 0, 'shape': [], 'out': [239, 238, 350], 'sorted_id': 237}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[237] = op;
            
            op->set_inputs( forward_result[158] );
            op->set_inputs( forward_result[209] );
            op->set_inputs( forward_result[236] );
        }
        
        // {'name': 'Net/1197', 'op': 'prim::TupleUnpack', 'in': [237], 'output_id': 1, 'shape': [], 'out': [354], 'sorted_id': 238}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[238] = op;
            
            op->set_inputs( forward_result[237] );
        }
        
        // {'name': 'Net/1196', 'op': 'prim::TupleUnpack', 'in': [237], 'output_id': 0, 'shape': [4, 400], 'out': [240], 'sorted_id': 239}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[239] = op;
            
            op->set_inputs( forward_result[237] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.9', 'op': 'aten::relu', 'in': [239], 'output_id': 0, 'shape': [4, 400], 'out': [268], 'sorted_id': 240}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[240] = op;
            
            op->set_inputs( forward_result[239] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/weight_mu/weight_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [324, 254], 'sorted_id': 241}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[241] = new VariableTensor( l3_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/weight_rho/weight_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [245, 243, 326, 320, 246], 'sorted_id': 242}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[242] = new VariableTensor( l3_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1212', 'op': 'aten::exp', 'in': [242], 'output_id': 0, 'shape': [10, 400], 'out': [244], 'sorted_id': 243}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[243] = op;
            
            op->set_inputs( forward_result[242] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1213', 'op': 'aten::log1p', 'in': [243], 'output_id': 0, 'shape': [10, 400], 'out': [253], 'sorted_id': 244}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[244] = op;
            
            op->set_inputs( forward_result[243] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1204', 'op': 'aten::size', 'in': [242, 10], 'output_id': 0, 'shape': [], 'out': [247, 249], 'sorted_id': 245}
        {
            SizeOp* op = new SizeOp();
            forward_result[245] = op;
            
            op->set_inputs( forward_result[242] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1205', 'op': 'aten::size', 'in': [242, 12], 'output_id': 0, 'shape': [], 'out': [247, 249], 'sorted_id': 246}
        {
            SizeOp* op = new SizeOp();
            forward_result[246] = op;
            
            op->set_inputs( forward_result[242] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1206', 'op': 'prim::ListConstruct', 'in': [245, 246], 'output_id': 0, 'shape': [], 'out': [248], 'sorted_id': 247}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[247] = op;
            
            op->set_inputs( forward_result[245] );
            op->set_inputs( forward_result[246] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1207', 'op': 'aten::expand', 'in': [9, 247, 15], 'output_id': 0, 'shape': [10, 400], 'out': [251], 'sorted_id': 248}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[248] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[247] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1208', 'op': 'prim::ListConstruct', 'in': [245, 246], 'output_id': 0, 'shape': [], 'out': [250], 'sorted_id': 249}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[249] = op;
            
            op->set_inputs( forward_result[245] );
            op->set_inputs( forward_result[246] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1209', 'op': 'aten::expand', 'in': [17, 249, 15], 'output_id': 0, 'shape': [10, 400], 'out': [251], 'sorted_id': 250}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[250] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[249] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1210', 'op': 'aten::normal', 'in': [248, 250, 20], 'output_id': 0, 'shape': [10, 400], 'out': [252], 'sorted_id': 251}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[251] = op;
            
            op->set_inputs( forward_result[248] );
            op->set_inputs( forward_result[250] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/epsilon.9', 'op': 'aten::to', 'in': [251, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10, 400], 'out': [253], 'sorted_id': 252}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[252] = op;
            
            op->set_inputs( forward_result[251] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1214', 'op': 'aten::mul', 'in': [244, 252], 'output_id': 0, 'shape': [10, 400], 'out': [254], 'sorted_id': 253}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[253] = op;
            
            op->set_inputs( forward_result[244] );
            op->set_inputs( forward_result[252] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/value.9', 'op': 'aten::add', 'in': [241, 253, 12], 'output_id': 0, 'shape': [10, 400], 'out': [268, 269, 324, 280], 'sorted_id': 254}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[254] = op;
            
            op->set_inputs( forward_result[241] );
            op->set_inputs( forward_result[253] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/bias_mu/bias_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [267, 337], 'sorted_id': 255}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[255] = new VariableTensor( l3_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/bias_rho/bias_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [257, 339, 259, 333], 'sorted_id': 256}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[256] = new VariableTensor( l3_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1223', 'op': 'aten::exp', 'in': [256], 'output_id': 0, 'shape': [10], 'out': [258], 'sorted_id': 257}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[257] = op;
            
            op->set_inputs( forward_result[256] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1224', 'op': 'aten::log1p', 'in': [257], 'output_id': 0, 'shape': [10], 'out': [266], 'sorted_id': 258}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[258] = op;
            
            op->set_inputs( forward_result[257] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1216', 'op': 'aten::size', 'in': [256, 10], 'output_id': 0, 'shape': [], 'out': [260, 262], 'sorted_id': 259}
        {
            SizeOp* op = new SizeOp();
            forward_result[259] = op;
            
            op->set_inputs( forward_result[256] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1217', 'op': 'prim::ListConstruct', 'in': [259], 'output_id': 0, 'shape': [], 'out': [261], 'sorted_id': 260}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[260] = op;
            
            op->set_inputs( forward_result[259] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1218', 'op': 'aten::expand', 'in': [9, 260, 15], 'output_id': 0, 'shape': [10], 'out': [264], 'sorted_id': 261}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[261] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[260] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1219', 'op': 'prim::ListConstruct', 'in': [259], 'output_id': 0, 'shape': [], 'out': [263], 'sorted_id': 262}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[262] = op;
            
            op->set_inputs( forward_result[259] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1220', 'op': 'aten::expand', 'in': [17, 262, 15], 'output_id': 0, 'shape': [10], 'out': [264], 'sorted_id': 263}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[263] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[262] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1221', 'op': 'aten::normal', 'in': [261, 263, 20], 'output_id': 0, 'shape': [10], 'out': [265], 'sorted_id': 264}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[264] = op;
            
            op->set_inputs( forward_result[261] );
            op->set_inputs( forward_result[263] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/epsilon', 'op': 'aten::to', 'in': [264, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10], 'out': [266], 'sorted_id': 265}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[265] = op;
            
            op->set_inputs( forward_result[264] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1225', 'op': 'aten::mul', 'in': [258, 265], 'output_id': 0, 'shape': [10], 'out': [267], 'sorted_id': 266}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[266] = op;
            
            op->set_inputs( forward_result[258] );
            op->set_inputs( forward_result[265] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/value', 'op': 'aten::add', 'in': [255, 266, 12], 'output_id': 0, 'shape': [10], 'out': [337, 294, 305, 268], 'sorted_id': 267}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[267] = op;
            
            op->set_inputs( forward_result[255] );
            op->set_inputs( forward_result[266] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/input.11', 'op': 'aten::linear', 'in': [240, 254, 267], 'output_id': 0, 'shape': [4, 10], 'out': [347], 'sorted_id': 268}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[268] = op;
            
            op->set_inputs( forward_result[240] );
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[267] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1229', 'op': 'aten::sub', 'in': [254, 41, 12], 'output_id': 0, 'shape': [10, 400], 'out': [270], 'sorted_id': 269}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[269] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1230', 'op': 'aten::pow', 'in': [269, 43], 'output_id': 0, 'shape': [10, 400], 'out': [271], 'sorted_id': 270}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[270] = op;
            
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1231', 'op': 'aten::neg', 'in': [270], 'output_id': 0, 'shape': [10, 400], 'out': [274], 'sorted_id': 271}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[271] = op;
            
            op->set_inputs( forward_result[270] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.17', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [273], 'sorted_id': 272}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[272] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1232', 'op': 'aten::mul', 'in': [272, 48], 'output_id': 0, 'shape': [1], 'out': [274], 'sorted_id': 273}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[273] = op;
            
            op->set_inputs( forward_result[272] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1233', 'op': 'aten::div', 'in': [271, 273], 'output_id': 0, 'shape': [10, 400], 'out': [276], 'sorted_id': 274}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[274] = op;
            
            op->set_inputs( forward_result[271] );
            op->set_inputs( forward_result[273] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.17', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [276], 'sorted_id': 275}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[275] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1234', 'op': 'aten::sub', 'in': [274, 275, 12], 'output_id': 0, 'shape': [10, 400], 'out': [277], 'sorted_id': 276}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[276] = op;
            
            op->set_inputs( forward_result[274] );
            op->set_inputs( forward_result[275] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1235', 'op': 'aten::sub', 'in': [276, 53, 12], 'output_id': 0, 'shape': [10, 400], 'out': [278], 'sorted_id': 277}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[277] = op;
            
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob1.9', 'op': 'aten::exp', 'in': [277], 'output_id': 0, 'shape': [10, 400], 'out': [279], 'sorted_id': 278}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[278] = op;
            
            op->set_inputs( forward_result[277] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1247', 'op': 'aten::mul', 'in': [278, 56], 'output_id': 0, 'shape': [10, 400], 'out': [291], 'sorted_id': 279}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[279] = op;
            
            op->set_inputs( forward_result[278] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1239', 'op': 'aten::sub', 'in': [254, 41, 12], 'output_id': 0, 'shape': [10, 400], 'out': [281], 'sorted_id': 280}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[280] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1240', 'op': 'aten::pow', 'in': [280, 43], 'output_id': 0, 'shape': [10, 400], 'out': [282], 'sorted_id': 281}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[281] = op;
            
            op->set_inputs( forward_result[280] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1241', 'op': 'aten::neg', 'in': [281], 'output_id': 0, 'shape': [10, 400], 'out': [285], 'sorted_id': 282}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[282] = op;
            
            op->set_inputs( forward_result[281] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.19', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [284], 'sorted_id': 283}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[283] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1242', 'op': 'aten::mul', 'in': [283, 48], 'output_id': 0, 'shape': [1], 'out': [285], 'sorted_id': 284}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[284] = op;
            
            op->set_inputs( forward_result[283] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1243', 'op': 'aten::div', 'in': [282, 284], 'output_id': 0, 'shape': [10, 400], 'out': [287], 'sorted_id': 285}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[285] = op;
            
            op->set_inputs( forward_result[282] );
            op->set_inputs( forward_result[284] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.19', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [287], 'sorted_id': 286}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[286] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1244', 'op': 'aten::sub', 'in': [285, 286, 12], 'output_id': 0, 'shape': [10, 400], 'out': [288], 'sorted_id': 287}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[287] = op;
            
            op->set_inputs( forward_result[285] );
            op->set_inputs( forward_result[286] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1245', 'op': 'aten::sub', 'in': [287, 53, 12], 'output_id': 0, 'shape': [10, 400], 'out': [289], 'sorted_id': 288}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[288] = op;
            
            op->set_inputs( forward_result[287] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob2.9', 'op': 'aten::exp', 'in': [288], 'output_id': 0, 'shape': [10, 400], 'out': [290], 'sorted_id': 289}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[289] = op;
            
            op->set_inputs( forward_result[288] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1248', 'op': 'aten::mul', 'in': [289, 56], 'output_id': 0, 'shape': [10, 400], 'out': [291], 'sorted_id': 290}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[290] = op;
            
            op->set_inputs( forward_result[289] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1249', 'op': 'aten::add', 'in': [279, 290, 12], 'output_id': 0, 'shape': [10, 400], 'out': [292], 'sorted_id': 291}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[291] = op;
            
            op->set_inputs( forward_result[279] );
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1250', 'op': 'aten::log', 'in': [291], 'output_id': 0, 'shape': [10, 400], 'out': [293], 'sorted_id': 292}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[292] = op;
            
            op->set_inputs( forward_result[291] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1251', 'op': 'aten::sum', 'in': [292, 20], 'output_id': 0, 'shape': [], 'out': [319], 'sorted_id': 293}
        {
            SumOp* op = new SumOp();
            forward_result[293] = op;
            
            op->set_inputs( forward_result[292] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1254', 'op': 'aten::sub', 'in': [267, 41, 12], 'output_id': 0, 'shape': [10], 'out': [295], 'sorted_id': 294}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[294] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1255', 'op': 'aten::pow', 'in': [294, 43], 'output_id': 0, 'shape': [10], 'out': [296], 'sorted_id': 295}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[295] = op;
            
            op->set_inputs( forward_result[294] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1256', 'op': 'aten::neg', 'in': [295], 'output_id': 0, 'shape': [10], 'out': [299], 'sorted_id': 296}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[296] = op;
            
            op->set_inputs( forward_result[295] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.21', 'op': 'aten::pow', 'in': [46, 43], 'output_id': 0, 'shape': [1], 'out': [298], 'sorted_id': 297}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[297] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1257', 'op': 'aten::mul', 'in': [297, 48], 'output_id': 0, 'shape': [1], 'out': [299], 'sorted_id': 298}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[298] = op;
            
            op->set_inputs( forward_result[297] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1258', 'op': 'aten::div', 'in': [296, 298], 'output_id': 0, 'shape': [10], 'out': [301], 'sorted_id': 299}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[299] = op;
            
            op->set_inputs( forward_result[296] );
            op->set_inputs( forward_result[298] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.21', 'op': 'aten::log', 'in': [46], 'output_id': 0, 'shape': [1], 'out': [301], 'sorted_id': 300}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[300] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1259', 'op': 'aten::sub', 'in': [299, 300, 12], 'output_id': 0, 'shape': [10], 'out': [302], 'sorted_id': 301}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[301] = op;
            
            op->set_inputs( forward_result[299] );
            op->set_inputs( forward_result[300] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1260', 'op': 'aten::sub', 'in': [301, 53, 12], 'output_id': 0, 'shape': [10], 'out': [303], 'sorted_id': 302}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[302] = op;
            
            op->set_inputs( forward_result[301] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob1', 'op': 'aten::exp', 'in': [302], 'output_id': 0, 'shape': [10], 'out': [304], 'sorted_id': 303}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[303] = op;
            
            op->set_inputs( forward_result[302] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1272', 'op': 'aten::mul', 'in': [303, 56], 'output_id': 0, 'shape': [10], 'out': [316], 'sorted_id': 304}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[304] = op;
            
            op->set_inputs( forward_result[303] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1264', 'op': 'aten::sub', 'in': [267, 41, 12], 'output_id': 0, 'shape': [10], 'out': [306], 'sorted_id': 305}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[305] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1265', 'op': 'aten::pow', 'in': [305, 43], 'output_id': 0, 'shape': [10], 'out': [307], 'sorted_id': 306}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[306] = op;
            
            op->set_inputs( forward_result[305] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1266', 'op': 'aten::neg', 'in': [306], 'output_id': 0, 'shape': [10], 'out': [310], 'sorted_id': 307}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[307] = op;
            
            op->set_inputs( forward_result[306] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var', 'op': 'aten::pow', 'in': [61, 43], 'output_id': 0, 'shape': [1], 'out': [309], 'sorted_id': 308}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[308] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1267', 'op': 'aten::mul', 'in': [308, 48], 'output_id': 0, 'shape': [1], 'out': [310], 'sorted_id': 309}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[309] = op;
            
            op->set_inputs( forward_result[308] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1268', 'op': 'aten::div', 'in': [307, 309], 'output_id': 0, 'shape': [10], 'out': [312], 'sorted_id': 310}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[310] = op;
            
            op->set_inputs( forward_result[307] );
            op->set_inputs( forward_result[309] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [1], 'out': [312], 'sorted_id': 311}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[311] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1269', 'op': 'aten::sub', 'in': [310, 311, 12], 'output_id': 0, 'shape': [10], 'out': [313], 'sorted_id': 312}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[312] = op;
            
            op->set_inputs( forward_result[310] );
            op->set_inputs( forward_result[311] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1270', 'op': 'aten::sub', 'in': [312, 53, 12], 'output_id': 0, 'shape': [10], 'out': [314], 'sorted_id': 313}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[313] = op;
            
            op->set_inputs( forward_result[312] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob2', 'op': 'aten::exp', 'in': [313], 'output_id': 0, 'shape': [10], 'out': [315], 'sorted_id': 314}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[314] = op;
            
            op->set_inputs( forward_result[313] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1273', 'op': 'aten::mul', 'in': [314, 56], 'output_id': 0, 'shape': [10], 'out': [316], 'sorted_id': 315}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[315] = op;
            
            op->set_inputs( forward_result[314] );
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1274', 'op': 'aten::add', 'in': [304, 315, 12], 'output_id': 0, 'shape': [10], 'out': [317], 'sorted_id': 316}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[316] = op;
            
            op->set_inputs( forward_result[304] );
            op->set_inputs( forward_result[315] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1275', 'op': 'aten::log', 'in': [316], 'output_id': 0, 'shape': [10], 'out': [318], 'sorted_id': 317}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[317] = op;
            
            op->set_inputs( forward_result[316] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1276', 'op': 'aten::sum', 'in': [317, 20], 'output_id': 0, 'shape': [], 'out': [319], 'sorted_id': 318}
        {
            SumOp* op = new SumOp();
            forward_result[318] = op;
            
            op->set_inputs( forward_result[317] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1277', 'op': 'aten::add', 'in': [293, 318, 12], 'output_id': 0, 'shape': [], 'out': [347], 'sorted_id': 319}
        {
            AddOp* op = new AddOp();
            forward_result[319] = op;
            
            op->set_inputs( forward_result[293] );
            op->set_inputs( forward_result[318] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1278', 'op': 'aten::exp', 'in': [242], 'output_id': 0, 'shape': [10, 400], 'out': [321], 'sorted_id': 320}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[320] = op;
            
            op->set_inputs( forward_result[242] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1279', 'op': 'aten::log1p', 'in': [320], 'output_id': 0, 'shape': [10, 400], 'out': [322], 'sorted_id': 321}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[321] = op;
            
            op->set_inputs( forward_result[320] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1280', 'op': 'aten::log', 'in': [321], 'output_id': 0, 'shape': [10, 400], 'out': [323], 'sorted_id': 322}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[322] = op;
            
            op->set_inputs( forward_result[321] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1281', 'op': 'aten::rsub', 'in': [322, 102, 12], 'output_id': 0, 'shape': [10, 400], 'out': [331], 'sorted_id': 323}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[323] = op;
            
            op->set_inputs( forward_result[322] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1282', 'op': 'aten::sub', 'in': [254, 241, 12], 'output_id': 0, 'shape': [10, 400], 'out': [325], 'sorted_id': 324}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[324] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[241] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1283', 'op': 'aten::pow', 'in': [324, 43], 'output_id': 0, 'shape': [10, 400], 'out': [330], 'sorted_id': 325}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[325] = op;
            
            op->set_inputs( forward_result[324] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1284', 'op': 'aten::exp', 'in': [242], 'output_id': 0, 'shape': [10, 400], 'out': [327], 'sorted_id': 326}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[326] = op;
            
            op->set_inputs( forward_result[242] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1285', 'op': 'aten::log1p', 'in': [326], 'output_id': 0, 'shape': [10, 400], 'out': [328], 'sorted_id': 327}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[327] = op;
            
            op->set_inputs( forward_result[326] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1286', 'op': 'aten::pow', 'in': [327, 43], 'output_id': 0, 'shape': [10, 400], 'out': [329], 'sorted_id': 328}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[328] = op;
            
            op->set_inputs( forward_result[327] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1287', 'op': 'aten::mul', 'in': [328, 48], 'output_id': 0, 'shape': [10, 400], 'out': [330], 'sorted_id': 329}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[329] = op;
            
            op->set_inputs( forward_result[328] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1288', 'op': 'aten::div', 'in': [325, 329], 'output_id': 0, 'shape': [10, 400], 'out': [331], 'sorted_id': 330}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[330] = op;
            
            op->set_inputs( forward_result[325] );
            op->set_inputs( forward_result[329] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1289', 'op': 'aten::sub', 'in': [323, 330, 12], 'output_id': 0, 'shape': [10, 400], 'out': [332], 'sorted_id': 331}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[331] = op;
            
            op->set_inputs( forward_result[323] );
            op->set_inputs( forward_result[330] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1290', 'op': 'aten::sum', 'in': [331, 20], 'output_id': 0, 'shape': [], 'out': [346], 'sorted_id': 332}
        {
            SumOp* op = new SumOp();
            forward_result[332] = op;
            
            op->set_inputs( forward_result[331] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1291', 'op': 'aten::exp', 'in': [256], 'output_id': 0, 'shape': [10], 'out': [334], 'sorted_id': 333}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[333] = op;
            
            op->set_inputs( forward_result[256] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1292', 'op': 'aten::log1p', 'in': [333], 'output_id': 0, 'shape': [10], 'out': [335], 'sorted_id': 334}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[334] = op;
            
            op->set_inputs( forward_result[333] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1293', 'op': 'aten::log', 'in': [334], 'output_id': 0, 'shape': [10], 'out': [336], 'sorted_id': 335}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[335] = op;
            
            op->set_inputs( forward_result[334] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1294', 'op': 'aten::rsub', 'in': [335, 102, 12], 'output_id': 0, 'shape': [10], 'out': [344], 'sorted_id': 336}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[336] = op;
            
            op->set_inputs( forward_result[335] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1295', 'op': 'aten::sub', 'in': [267, 255, 12], 'output_id': 0, 'shape': [10], 'out': [338], 'sorted_id': 337}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[337] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[255] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1296', 'op': 'aten::pow', 'in': [337, 43], 'output_id': 0, 'shape': [10], 'out': [343], 'sorted_id': 338}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[338] = op;
            
            op->set_inputs( forward_result[337] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1297', 'op': 'aten::exp', 'in': [256], 'output_id': 0, 'shape': [10], 'out': [340], 'sorted_id': 339}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[339] = op;
            
            op->set_inputs( forward_result[256] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1298', 'op': 'aten::log1p', 'in': [339], 'output_id': 0, 'shape': [10], 'out': [341], 'sorted_id': 340}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[340] = op;
            
            op->set_inputs( forward_result[339] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1299', 'op': 'aten::pow', 'in': [340, 43], 'output_id': 0, 'shape': [10], 'out': [342], 'sorted_id': 341}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[341] = op;
            
            op->set_inputs( forward_result[340] );
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1300', 'op': 'aten::mul', 'in': [341, 48], 'output_id': 0, 'shape': [10], 'out': [343], 'sorted_id': 342}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[342] = op;
            
            op->set_inputs( forward_result[341] );
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1301', 'op': 'aten::div', 'in': [338, 342], 'output_id': 0, 'shape': [10], 'out': [344], 'sorted_id': 343}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[343] = op;
            
            op->set_inputs( forward_result[338] );
            op->set_inputs( forward_result[342] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1302', 'op': 'aten::sub', 'in': [336, 343, 12], 'output_id': 0, 'shape': [10], 'out': [345], 'sorted_id': 344}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[344] = op;
            
            op->set_inputs( forward_result[336] );
            op->set_inputs( forward_result[343] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1303', 'op': 'aten::sum', 'in': [344, 20], 'output_id': 0, 'shape': [], 'out': [346], 'sorted_id': 345}
        {
            SumOp* op = new SumOp();
            forward_result[345] = op;
            
            op->set_inputs( forward_result[344] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1304', 'op': 'aten::add', 'in': [332, 345, 12], 'output_id': 0, 'shape': [], 'out': [347], 'sorted_id': 346}
        {
            AddOp* op = new AddOp();
            forward_result[346] = op;
            
            op->set_inputs( forward_result[332] );
            op->set_inputs( forward_result[345] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/1306', 'op': 'prim::TupleConstruct', 'in': [268, 319, 346], 'output_id': 0, 'shape': [], 'out': [348, 351, 352], 'sorted_id': 347}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[347] = op;
            
            op->set_inputs( forward_result[268] );
            op->set_inputs( forward_result[319] );
            op->set_inputs( forward_result[346] );
        }
        
        // {'name': 'Net/1308', 'op': 'prim::TupleUnpack', 'in': [347], 'output_id': 1, 'shape': [], 'out': [354], 'sorted_id': 348}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[348] = op;
            
            op->set_inputs( forward_result[347] );
        }
        
        // {'name': 'Net/1087', 'op': 'prim::TupleUnpack', 'in': [127], 'output_id': 2, 'shape': [], 'out': [354], 'sorted_id': 349}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[349] = op;
            
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Net/1198', 'op': 'prim::TupleUnpack', 'in': [237], 'output_id': 2, 'shape': [], 'out': [354], 'sorted_id': 350}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[350] = op;
            
            op->set_inputs( forward_result[237] );
        }
        
        // {'name': 'Net/1309', 'op': 'prim::TupleUnpack', 'in': [347], 'output_id': 2, 'shape': [], 'out': [354], 'sorted_id': 351}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[351] = op;
            
            op->set_inputs( forward_result[347] );
        }
        
        // {'name': 'Net/1307', 'op': 'prim::TupleUnpack', 'in': [347], 'output_id': 0, 'shape': [4, 10], 'out': [353], 'sorted_id': 352}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[352] = op;
            
            op->set_inputs( forward_result[347] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/1310', 'op': 'aten::log_softmax', 'in': [352, 12, 20], 'output_id': 0, 'shape': [4, 10], 'out': [354], 'sorted_id': 353}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[353] = op;
            
            op->set_inputs( forward_result[352] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/1311', 'op': 'prim::TupleConstruct', 'in': [128, 238, 348, 349, 350, 351, 353], 'output_id': 0, 'shape': [], 'out': [355, 363, 374, 356, 366, 364, 358], 'sorted_id': 354}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[354] = op;
            
            op->set_inputs( forward_result[128] );
            op->set_inputs( forward_result[238] );
            op->set_inputs( forward_result[348] );
            op->set_inputs( forward_result[349] );
            op->set_inputs( forward_result[350] );
            op->set_inputs( forward_result[351] );
            op->set_inputs( forward_result[353] );
        }
        
        // {'name': 'Net/1315', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 3, 'shape': [], 'out': [357], 'sorted_id': 355}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[355] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/1316', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 4, 'shape': [], 'out': [357], 'sorted_id': 356}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[356] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1321', 'op': 'aten::add', 'in': [355, 356, 12], 'output_id': 0, 'shape': [], 'out': [359], 'sorted_id': 357}
        {
            AddOp* op = new AddOp();
            forward_result[357] = op;
            
            op->set_inputs( forward_result[355] );
            op->set_inputs( forward_result[356] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/1317', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 5, 'shape': [], 'out': [359], 'sorted_id': 358}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[358] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1322', 'op': 'aten::add', 'in': [357, 358, 12], 'output_id': 0, 'shape': [], 'out': [360], 'sorted_id': 359}
        {
            AddOp* op = new AddOp();
            forward_result[359] = op;
            
            op->set_inputs( forward_result[357] );
            op->set_inputs( forward_result[358] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1327', 'op': 'prim::ListConstruct', 'in': [359], 'output_id': 0, 'shape': [], 'out': [361], 'sorted_id': 360}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[360] = op;
            
            op->set_inputs( forward_result[359] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_qs', 'op': 'aten::stack', 'in': [360, 10], 'output_id': 0, 'shape': [1], 'out': [362], 'sorted_id': 361}
        {
            Tensor::shape_type shape = {1};
            StackOp* op = new StackOp();
            forward_result[361] = op;
            
            op->set_inputs( forward_result[360] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_q', 'op': 'aten::mean', 'in': [361, 20], 'output_id': 0, 'shape': [], 'out': [371], 'sorted_id': 362}
        {
            MeanOp* op = new MeanOp();
            forward_result[362] = op;
            
            op->set_inputs( forward_result[361] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/1312', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 0, 'shape': [], 'out': [365], 'sorted_id': 363}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[363] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/1313', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 1, 'shape': [], 'out': [365], 'sorted_id': 364}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[364] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1319', 'op': 'aten::add', 'in': [363, 364, 12], 'output_id': 0, 'shape': [], 'out': [367], 'sorted_id': 365}
        {
            AddOp* op = new AddOp();
            forward_result[365] = op;
            
            op->set_inputs( forward_result[363] );
            op->set_inputs( forward_result[364] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/1314', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 2, 'shape': [], 'out': [367], 'sorted_id': 366}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[366] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1320', 'op': 'aten::add', 'in': [365, 366, 12], 'output_id': 0, 'shape': [], 'out': [368], 'sorted_id': 367}
        {
            AddOp* op = new AddOp();
            forward_result[367] = op;
            
            op->set_inputs( forward_result[365] );
            op->set_inputs( forward_result[366] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1325', 'op': 'prim::ListConstruct', 'in': [367], 'output_id': 0, 'shape': [], 'out': [369], 'sorted_id': 368}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[368] = op;
            
            op->set_inputs( forward_result[367] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_ps', 'op': 'aten::stack', 'in': [368, 10], 'output_id': 0, 'shape': [1], 'out': [370], 'sorted_id': 369}
        {
            Tensor::shape_type shape = {1};
            StackOp* op = new StackOp();
            forward_result[369] = op;
            
            op->set_inputs( forward_result[368] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_p', 'op': 'aten::mean', 'in': [369, 20], 'output_id': 0, 'shape': [], 'out': [371], 'sorted_id': 370}
        {
            MeanOp* op = new MeanOp();
            forward_result[370] = op;
            
            op->set_inputs( forward_result[369] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1334', 'op': 'aten::sub', 'in': [362, 370, 12], 'output_id': 0, 'shape': [], 'out': [373], 'sorted_id': 371}
        {
            SubOp* op = new SubOp();
            forward_result[371] = op;
            
            op->set_inputs( forward_result[362] );
            op->set_inputs( forward_result[370] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/952', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 15000.0, 'out': [373], 'sorted_id': 372}
        {
            Tensor c = (fprec)15000.0;
            forward_result[372] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1335', 'op': 'aten::div', 'in': [371, 372], 'output_id': 0, 'shape': [], 'out': [382], 'sorted_id': 373}
        {
            DivOp* op = new DivOp();
            forward_result[373] = op;
            
            op->set_inputs( forward_result[371] );
            op->set_inputs( forward_result[372] );
        }
        
        // {'name': 'Net/1318', 'op': 'prim::TupleUnpack', 'in': [354], 'output_id': 6, 'shape': [4, 10], 'out': [375], 'sorted_id': 374}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 6 );
            forward_result[374] = op;
            
            op->set_inputs( forward_result[354] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1323', 'op': 'prim::ListConstruct', 'in': [374], 'output_id': 0, 'shape': [], 'out': [376], 'sorted_id': 375}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[375] = op;
            
            op->set_inputs( forward_result[374] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/outputs', 'op': 'aten::stack', 'in': [375, 10], 'output_id': 0, 'shape': [1, 4, 10], 'out': [378], 'sorted_id': 376}
        {
            Tensor::shape_type shape = {1,4,10};
            StackOp* op = new StackOp();
            forward_result[376] = op;
            
            op->set_inputs( forward_result[375] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1331', 'op': 'prim::ListConstruct', 'in': [10], 'output_id': 0, 'shape': [], 'out': [378], 'sorted_id': 377}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[377] = op;
            
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/input', 'op': 'aten::mean', 'in': [376, 377, 15, 20], 'output_id': 0, 'shape': [4, 10], 'out': [381], 'sorted_id': 378}
        {
            Tensor::shape_type shape = {4,10};
            MeanOp* op = new MeanOp();
            forward_result[378] = op;
            
            op->set_inputs( forward_result[376] );
            op->set_inputs( forward_result[377] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/954', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [4], 'constant_value': [5.0, 4.0, 9.0, 4.0], 'out': [381], 'sorted_id': 379}
        {
            Tensor::shape_type shape = {4};
            Constant4.reshape( shape );
            forward_result[379] = new VariableTensor( Constant4, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/953', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [381], 'sorted_id': 380}
        {
            Tensor c = (fprec)-100.0;
            forward_result[380] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/nll', 'op': 'aten::nll_loss_nd', 'in': [378, 379, 20, 43, 380], 'output_id': 0, 'shape': [], 'out': [382], 'sorted_id': 381}
        {
            NLLLossOp* op = new NLLLossOp();
            forward_result[381] = op;
            
            op->set_inputs( forward_result[378] );
            op->set_inputs( forward_result[379] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[380] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1336', 'op': 'aten::add', 'in': [373, 381, 12], 'output_id': 0, 'shape': [], 'out': [383], 'sorted_id': 382}
        {
            AddOp* op = new AddOp();
            forward_result[382] = op;
            
            op->set_inputs( forward_result[373] );
            op->set_inputs( forward_result[381] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [382], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 383}
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
        vector<MCTNode*> forward_result(384);
    
        // input data
        Tensor::shape_type shape = {4,1,28,28};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 382 );
    #else
        do_train1( forward_result, input_var, 382 );
    #endif
        
        return 0;
    }
    