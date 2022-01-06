
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
    extern Tensor  Constant1;
    extern Tensor  l1_weight_mu;
    extern Tensor  l1_weight_rho;
    extern Tensor  l1_bias_mu;
    extern Tensor  l1_bias_rho;
    extern Tensor  Constant2;
    extern Tensor  Constant3;
    extern Tensor  Constant4;
    extern Tensor  l2_weight_mu;
    extern Tensor  l2_weight_rho;
    extern Tensor  l2_bias_mu;
    extern Tensor  l2_bias_rho;
    extern Tensor  l3_weight_mu;
    extern Tensor  l3_weight_rho;
    extern Tensor  l3_bias_mu;
    extern Tensor  l3_bias_rho;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'Net/BBBLoss[loss_func]/1027', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [90, 299, 19, 200, 119, 91, 313, 215, 108, 386, 338, 244, 67, 371, 104, 128, 334, 276, 76, 226, 393, 1, 180, 324, 97, 327, 345, 214, 341, 384, 64, 105, 51, 182, 335, 310, 201, 159, 249, 391, 289, 268, 368, 232, 10, 229, 302, 193, 127, 28, 42, 204, 225, 167, 237, 298, 218, 353, 207, 366, 309, 346, 14, 190, 135, 141, 358, 189, 250, 148, 323, 94, 116, 259, 236, 82, 78, 316, 140, 257, 291, 122, 115, 150, 359], 'sorted_id': 0}
        {
            Tensor c = (fprec)1.0;
            forward_result[0] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1034', 'op': 'prim::ListConstruct', 'in': [0], 'output_id': 0, 'shape': [], 'out': [6], 'sorted_id': 1}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[1] = op;
            
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1024', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [49, 165, 178, 21, 12, 274, 62, 11, 6, 8, 287, 20], 'sorted_id': 2}
        {
            Tensor c = (fprec)6.0;
            forward_result[2] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1023', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [9, 12, 177, 8, 286, 273, 340, 20, 49, 11, 136, 164, 48, 315, 367, 178, 27, 21, 206, 6, 258, 287, 354, 121, 149, 165, 23, 274, 62, 96, 231, 13, 245, 61, 371], 'sorted_id': 3}
        {
            forward_result[3] = NULL;
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1022', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [49, 165, 178, 21, 12, 274, 62, 11, 6, 8, 287, 20], 'sorted_id': 4}
        {
            forward_result[4] = NULL;
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1021', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [12, 285, 387, 8, 163, 20, 49, 11, 283, 176, 394, 174, 161, 272, 47, 178, 21, 6, 58, 287, 44, 60, 165, 23, 274, 62, 270, 380], 'sorted_id': 5}
        {
            Tensor c = (fprec)0.0;
            forward_result[5] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1035', 'op': 'aten::zeros', 'in': [1, 2, 3, 4, 5], 'output_id': 0, 'shape': [1], 'out': [8], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {1};
            ZerosOp* op = new ZerosOp();
            forward_result[6] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1020', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [12, 8, 172, 56, 49, 41, 22, 178, 21, 158, 388, 281, 287, 381, 165, 30, 274, 62, 267], 'sorted_id': 7}
        {
            Tensor c = (fprec)0.0;
            forward_result[7] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_qs', 'op': 'aten::to', 'in': [6, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [1], 'out': [388, 9], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {1};
            ToOp* op = new ToOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_q', 'op': 'aten::mean', 'in': [8, 3], 'output_id': 0, 'shape': [], 'out': [14], 'sorted_id': 9}
        {
            MeanOp* op = new MeanOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1031', 'op': 'prim::ListConstruct', 'in': [0], 'output_id': 0, 'shape': [], 'out': [11], 'sorted_id': 10}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[10] = op;
            
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1032', 'op': 'aten::zeros', 'in': [10, 2, 3, 4, 5], 'output_id': 0, 'shape': [1], 'out': [12], 'sorted_id': 11}
        {
            Tensor::shape_type shape = {1};
            ZerosOp* op = new ZerosOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_ps', 'op': 'aten::to', 'in': [11, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [1], 'out': [13, 381], 'sorted_id': 12}
        {
            Tensor::shape_type shape = {1};
            ToOp* op = new ToOp();
            forward_result[12] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/log_p', 'op': 'aten::mean', 'in': [12, 3], 'output_id': 0, 'shape': [], 'out': [14], 'sorted_id': 13}
        {
            MeanOp* op = new MeanOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1398', 'op': 'aten::sub', 'in': [9, 13, 0], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            SubOp* op = new SubOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1005', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 15000.0, 'out': [16], 'sorted_id': 15}
        {
            Tensor c = (fprec)15000.0;
            forward_result[15] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1399', 'op': 'aten::div', 'in': [14, 15], 'output_id': 0, 'shape': [], 'out': [28], 'sorted_id': 16}
        {
            DivOp* op = new DivOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1026', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 4.0, 'out': [19], 'sorted_id': 17}
        {
            Tensor c = (fprec)4.0;
            forward_result[17] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1025', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 10.0, 'out': [19], 'sorted_id': 18}
        {
            Tensor c = (fprec)10.0;
            forward_result[18] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1028', 'op': 'prim::ListConstruct', 'in': [0, 17, 18], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 19}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1029', 'op': 'aten::zeros', 'in': [19, 2, 3, 4, 5], 'output_id': 0, 'shape': [1, 4, 10], 'out': [21], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {1,4,10};
            ZerosOp* op = new ZerosOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/outputs', 'op': 'aten::to', 'in': [20, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [1, 4, 10], 'out': [30, 23], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {1,4,10};
            ToOp* op = new ToOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1395', 'op': 'prim::ListConstruct', 'in': [7], 'output_id': 0, 'shape': [], 'out': [23], 'sorted_id': 22}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/input', 'op': 'aten::mean', 'in': [21, 22, 5, 3], 'output_id': 0, 'shape': [4, 10], 'out': [27], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {4,10};
            MeanOp* op = new MeanOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1007', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [4], 'constant_value': [5.0, 4.0, 9.0, 4.0], 'out': [27], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {4};
            Constant1.reshape( shape );
            forward_result[24] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1016', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [238, 317, 328, 210, 303, 319, 305, 254, 208, 183, 145, 71, 185, 360, 27, 194, 196, 221, 109, 129, 251, 219, 292, 100, 142, 347, 330, 132, 111, 86, 363, 68, 83, 350, 294, 241, 98], 'sorted_id': 25}
        {
            Tensor c = (fprec)2.0;
            forward_result[25] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1006', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [27], 'sorted_id': 26}
        {
            Tensor c = (fprec)-100.0;
            forward_result[26] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/nll', 'op': 'aten::nll_loss_nd', 'in': [23, 24, 3, 25, 26], 'output_id': 0, 'shape': [], 'out': [28], 'sorted_id': 27}
        {
            NLLLossOp* op = new NLLLossOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1400', 'op': 'aten::add', 'in': [16, 27, 0], 'output_id': 0, 'shape': [], 'out': [29], 'sorted_id': 28}
        {
            AddOp* op = new AddOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [28], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 29}
        {
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1383', 'op': 'aten::select', 'in': [21, 7, 7], 'output_id': 0, 'shape': [4, 10], 'out': [380], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {4,10};
            SelectOp* op = new SelectOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[21] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [4, 1, 28, 28], 'out': [35], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {4,1,28,28};
            forward_result[31] = &input_var;
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/1008', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -1.0, 'out': [34], 'sorted_id': 32}
        {
            Tensor c = (fprec)-1.0;
            forward_result[32] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/1009', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 784.0, 'out': [34], 'sorted_id': 33}
        {
            Tensor c = (fprec)784.0;
            forward_result[33] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/1040', 'op': 'prim::ListConstruct', 'in': [32, 33], 'output_id': 0, 'shape': [], 'out': [35], 'sorted_id': 34}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[33] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.1', 'op': 'aten::view', 'in': [31, 34], 'output_id': 0, 'shape': [4, 784], 'out': [65], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/weight_mu/weight_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [128, 51], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[36] = new VariableTensor( l1_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/weight_rho/weight_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [38, 123, 41, 42, 130], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[37] = new VariableTensor( l1_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1054', 'op': 'aten::exp', 'in': [37], 'output_id': 0, 'shape': [400, 784], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1055', 'op': 'aten::log1p', 'in': [38], 'output_id': 0, 'shape': [400, 784], 'out': [50], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1019', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [270, 283, 58, 44, 174, 161], 'sorted_id': 40}
        {
            Tensor c = (fprec)0.0;
            forward_result[40] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1046', 'op': 'aten::size', 'in': [37, 7], 'output_id': 0, 'shape': [], 'out': [43, 46], 'sorted_id': 41}
        {
            SizeOp* op = new SizeOp();
            forward_result[41] = op;
            
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1047', 'op': 'aten::size', 'in': [37, 0], 'output_id': 0, 'shape': [], 'out': [43, 46], 'sorted_id': 42}
        {
            SizeOp* op = new SizeOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1048', 'op': 'prim::ListConstruct', 'in': [41, 42], 'output_id': 0, 'shape': [], 'out': [44], 'sorted_id': 43}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1049', 'op': 'aten::expand', 'in': [40, 43, 5], 'output_id': 0, 'shape': [400, 784], 'out': [48], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1018', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [285, 176, 163, 60, 272, 47], 'sorted_id': 45}
        {
            Tensor c = (fprec)1.0;
            forward_result[45] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1050', 'op': 'prim::ListConstruct', 'in': [41, 42], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 46}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1051', 'op': 'aten::expand', 'in': [45, 46, 5], 'output_id': 0, 'shape': [400, 784], 'out': [48], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1052', 'op': 'aten::normal', 'in': [44, 47, 3], 'output_id': 0, 'shape': [400, 784], 'out': [49], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[48] = op;
            
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/epsilon.1', 'op': 'aten::to', 'in': [48, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [400, 784], 'out': [50], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {400,784};
            ToOp* op = new ToOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1056', 'op': 'aten::mul', 'in': [39, 49], 'output_id': 0, 'shape': [400, 784], 'out': [51], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/value.1', 'op': 'aten::add', 'in': [36, 50, 0], 'output_id': 0, 'shape': [400, 784], 'out': [128, 65, 67, 82], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/bias_mu/bias_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [141, 64], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[52] = new VariableTensor( l1_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l1]/bias_rho/bias_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [143, 56, 137, 54], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[53] = new VariableTensor( l1_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1065', 'op': 'aten::exp', 'in': [53], 'output_id': 0, 'shape': [400], 'out': [55], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1066', 'op': 'aten::log1p', 'in': [54], 'output_id': 0, 'shape': [400], 'out': [63], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1058', 'op': 'aten::size', 'in': [53, 7], 'output_id': 0, 'shape': [], 'out': [57, 59], 'sorted_id': 56}
        {
            SizeOp* op = new SizeOp();
            forward_result[56] = op;
            
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1059', 'op': 'prim::ListConstruct', 'in': [56], 'output_id': 0, 'shape': [], 'out': [58], 'sorted_id': 57}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1060', 'op': 'aten::expand', 'in': [40, 57, 5], 'output_id': 0, 'shape': [400], 'out': [61], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[58] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1061', 'op': 'prim::ListConstruct', 'in': [56], 'output_id': 0, 'shape': [], 'out': [60], 'sorted_id': 59}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1062', 'op': 'aten::expand', 'in': [45, 59, 5], 'output_id': 0, 'shape': [400], 'out': [61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[60] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1063', 'op': 'aten::normal', 'in': [58, 60, 3], 'output_id': 0, 'shape': [400], 'out': [62], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/epsilon.3', 'op': 'aten::to', 'in': [61, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [400], 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1067', 'op': 'aten::mul', 'in': [55, 62], 'output_id': 0, 'shape': [400], 'out': [64], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/value.3', 'op': 'aten::add', 'in': [52, 63, 0], 'output_id': 0, 'shape': [400], 'out': [65, 141, 97, 108], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/input.3', 'op': 'aten::linear', 'in': [35, 51, 64], 'output_id': 0, 'shape': [4, 400], 'out': [151], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1015', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [97, 302, 82, 327, 291, 218, 316, 108, 182, 67, 207, 193], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {1};
            Constant2.reshape( shape );
            forward_result[66] = new VariableTensor( Constant2 );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1071', 'op': 'aten::sub', 'in': [51, 66, 0], 'output_id': 0, 'shape': [400, 784], 'out': [68], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1072', 'op': 'aten::pow', 'in': [67, 25], 'output_id': 0, 'shape': [400, 784], 'out': [69], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[67] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1073', 'op': 'aten::neg', 'in': [68], 'output_id': 0, 'shape': [400, 784], 'out': [74], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1017', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [322, 210, 213, 75, 71, 188, 294, 103, 319, 100, 185, 297], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {1};
            Constant3.reshape( shape );
            forward_result[70] = new VariableTensor( Constant3 );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.1', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [73], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1014', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [242, 222, 295, 351, 255, 73, 101, 146, 112, 197, 186, 211, 133, 87, 331, 364, 306, 320], 'sorted_id': 72}
        {
            Tensor c = (fprec)2.0;
            forward_result[72] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1074', 'op': 'aten::mul', 'in': [71, 72], 'output_id': 0, 'shape': [1], 'out': [74], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1075', 'op': 'aten::div', 'in': [69, 73], 'output_id': 0, 'shape': [400, 784], 'out': [76], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.1', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [76], 'sorted_id': 75}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[75] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1076', 'op': 'aten::sub', 'in': [74, 75, 0], 'output_id': 0, 'shape': [400, 784], 'out': [78], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1013', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.9189, 'out': [91, 190, 299, 324, 105, 335, 78, 215, 116, 226, 310, 201], 'sorted_id': 77}
        {
            Tensor c = (fprec)0.9189;
            forward_result[77] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1077', 'op': 'aten::sub', 'in': [76, 77, 0], 'output_id': 0, 'shape': [400, 784], 'out': [79], 'sorted_id': 78}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[78] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob1.1', 'op': 'aten::exp', 'in': [78], 'output_id': 0, 'shape': [400, 784], 'out': [81], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[78] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1011', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [192, 107, 312, 337, 203, 217, 228, 118, 93, 81, 301, 326], 'sorted_id': 80}
        {
            Tensor c = (fprec)0.5;
            forward_result[80] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1089', 'op': 'aten::mul', 'in': [79, 80], 'output_id': 0, 'shape': [400, 784], 'out': [94], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1081', 'op': 'aten::sub', 'in': [51, 66, 0], 'output_id': 0, 'shape': [400, 784], 'out': [83], 'sorted_id': 82}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[82] = op;
            
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1082', 'op': 'aten::pow', 'in': [82, 25], 'output_id': 0, 'shape': [400, 784], 'out': [84], 'sorted_id': 83}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[83] = op;
            
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1083', 'op': 'aten::neg', 'in': [83], 'output_id': 0, 'shape': [400, 784], 'out': [88], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[83] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1012', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [89, 330, 224, 333, 196, 111, 86, 199, 221, 308, 305, 114], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {1};
            Constant4.reshape( shape );
            forward_result[85] = new VariableTensor( Constant4 );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.3', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [87], 'sorted_id': 86}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[86] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1084', 'op': 'aten::mul', 'in': [86, 72], 'output_id': 0, 'shape': [1], 'out': [88], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[86] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1085', 'op': 'aten::div', 'in': [84, 87], 'output_id': 0, 'shape': [400, 784], 'out': [90], 'sorted_id': 88}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[88] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[87] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.3', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [90], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1086', 'op': 'aten::sub', 'in': [88, 89, 0], 'output_id': 0, 'shape': [400, 784], 'out': [91], 'sorted_id': 90}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[90] = op;
            
            op->set_inputs( forward_result[88] );
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1087', 'op': 'aten::sub', 'in': [90, 77, 0], 'output_id': 0, 'shape': [400, 784], 'out': [92], 'sorted_id': 91}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[91] = op;
            
            op->set_inputs( forward_result[90] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob2.1', 'op': 'aten::exp', 'in': [91], 'output_id': 0, 'shape': [400, 784], 'out': [93], 'sorted_id': 92}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[92] = op;
            
            op->set_inputs( forward_result[91] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1090', 'op': 'aten::mul', 'in': [92, 80], 'output_id': 0, 'shape': [400, 784], 'out': [94], 'sorted_id': 93}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[92] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1091', 'op': 'aten::add', 'in': [81, 93, 0], 'output_id': 0, 'shape': [400, 784], 'out': [95], 'sorted_id': 94}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[94] = op;
            
            op->set_inputs( forward_result[81] );
            op->set_inputs( forward_result[93] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1092', 'op': 'aten::log', 'in': [94], 'output_id': 0, 'shape': [400, 784], 'out': [96], 'sorted_id': 95}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[95] = op;
            
            op->set_inputs( forward_result[94] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1093', 'op': 'aten::sum', 'in': [95, 3], 'output_id': 0, 'shape': [], 'out': [122], 'sorted_id': 96}
        {
            SumOp* op = new SumOp();
            forward_result[96] = op;
            
            op->set_inputs( forward_result[95] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1096', 'op': 'aten::sub', 'in': [64, 66, 0], 'output_id': 0, 'shape': [400], 'out': [98], 'sorted_id': 97}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[97] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1097', 'op': 'aten::pow', 'in': [97, 25], 'output_id': 0, 'shape': [400], 'out': [99], 'sorted_id': 98}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[98] = op;
            
            op->set_inputs( forward_result[97] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1098', 'op': 'aten::neg', 'in': [98], 'output_id': 0, 'shape': [400], 'out': [102], 'sorted_id': 99}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[99] = op;
            
            op->set_inputs( forward_result[98] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.5', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [101], 'sorted_id': 100}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[100] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1099', 'op': 'aten::mul', 'in': [100, 72], 'output_id': 0, 'shape': [1], 'out': [102], 'sorted_id': 101}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[101] = op;
            
            op->set_inputs( forward_result[100] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1100', 'op': 'aten::div', 'in': [99, 101], 'output_id': 0, 'shape': [400], 'out': [104], 'sorted_id': 102}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[102] = op;
            
            op->set_inputs( forward_result[99] );
            op->set_inputs( forward_result[101] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.5', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [104], 'sorted_id': 103}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[103] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1101', 'op': 'aten::sub', 'in': [102, 103, 0], 'output_id': 0, 'shape': [400], 'out': [105], 'sorted_id': 104}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[104] = op;
            
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[103] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1102', 'op': 'aten::sub', 'in': [104, 77, 0], 'output_id': 0, 'shape': [400], 'out': [106], 'sorted_id': 105}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[105] = op;
            
            op->set_inputs( forward_result[104] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob1.3', 'op': 'aten::exp', 'in': [105], 'output_id': 0, 'shape': [400], 'out': [107], 'sorted_id': 106}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[106] = op;
            
            op->set_inputs( forward_result[105] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1114', 'op': 'aten::mul', 'in': [106, 80], 'output_id': 0, 'shape': [400], 'out': [119], 'sorted_id': 107}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[107] = op;
            
            op->set_inputs( forward_result[106] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1106', 'op': 'aten::sub', 'in': [64, 66, 0], 'output_id': 0, 'shape': [400], 'out': [109], 'sorted_id': 108}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[108] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1107', 'op': 'aten::pow', 'in': [108, 25], 'output_id': 0, 'shape': [400], 'out': [110], 'sorted_id': 109}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[109] = op;
            
            op->set_inputs( forward_result[108] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1108', 'op': 'aten::neg', 'in': [109], 'output_id': 0, 'shape': [400], 'out': [113], 'sorted_id': 110}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[110] = op;
            
            op->set_inputs( forward_result[109] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/var.7', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [112], 'sorted_id': 111}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[111] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1109', 'op': 'aten::mul', 'in': [111, 72], 'output_id': 0, 'shape': [1], 'out': [113], 'sorted_id': 112}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[112] = op;
            
            op->set_inputs( forward_result[111] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1110', 'op': 'aten::div', 'in': [110, 112], 'output_id': 0, 'shape': [400], 'out': [115], 'sorted_id': 113}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[113] = op;
            
            op->set_inputs( forward_result[110] );
            op->set_inputs( forward_result[112] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/log_scale.7', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [115], 'sorted_id': 114}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[114] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1111', 'op': 'aten::sub', 'in': [113, 114, 0], 'output_id': 0, 'shape': [400], 'out': [116], 'sorted_id': 115}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[115] = op;
            
            op->set_inputs( forward_result[113] );
            op->set_inputs( forward_result[114] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1112', 'op': 'aten::sub', 'in': [115, 77, 0], 'output_id': 0, 'shape': [400], 'out': [117], 'sorted_id': 116}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[116] = op;
            
            op->set_inputs( forward_result[115] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/prob2.3', 'op': 'aten::exp', 'in': [116], 'output_id': 0, 'shape': [400], 'out': [118], 'sorted_id': 117}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[117] = op;
            
            op->set_inputs( forward_result[116] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1115', 'op': 'aten::mul', 'in': [117, 80], 'output_id': 0, 'shape': [400], 'out': [119], 'sorted_id': 118}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[118] = op;
            
            op->set_inputs( forward_result[117] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1116', 'op': 'aten::add', 'in': [107, 118, 0], 'output_id': 0, 'shape': [400], 'out': [120], 'sorted_id': 119}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[119] = op;
            
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[118] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1117', 'op': 'aten::log', 'in': [119], 'output_id': 0, 'shape': [400], 'out': [121], 'sorted_id': 120}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[120] = op;
            
            op->set_inputs( forward_result[119] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1118', 'op': 'aten::sum', 'in': [120, 3], 'output_id': 0, 'shape': [], 'out': [122], 'sorted_id': 121}
        {
            SumOp* op = new SumOp();
            forward_result[121] = op;
            
            op->set_inputs( forward_result[120] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1119', 'op': 'aten::add', 'in': [96, 121, 0], 'output_id': 0, 'shape': [], 'out': [151], 'sorted_id': 122}
        {
            AddOp* op = new AddOp();
            forward_result[122] = op;
            
            op->set_inputs( forward_result[96] );
            op->set_inputs( forward_result[121] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1120', 'op': 'aten::exp', 'in': [37], 'output_id': 0, 'shape': [400, 784], 'out': [124], 'sorted_id': 123}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[123] = op;
            
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1121', 'op': 'aten::log1p', 'in': [123], 'output_id': 0, 'shape': [400, 784], 'out': [125], 'sorted_id': 124}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[124] = op;
            
            op->set_inputs( forward_result[123] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1122', 'op': 'aten::log', 'in': [124], 'output_id': 0, 'shape': [400, 784], 'out': [127], 'sorted_id': 125}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[125] = op;
            
            op->set_inputs( forward_result[124] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1010', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -0.9189385332046727, 'out': [345, 236, 249, 358, 127, 140], 'sorted_id': 126}
        {
            Tensor c = (fprec)-0.9189385332046727;
            forward_result[126] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1123', 'op': 'aten::rsub', 'in': [125, 126, 0], 'output_id': 0, 'shape': [400, 784], 'out': [135], 'sorted_id': 127}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[127] = op;
            
            op->set_inputs( forward_result[125] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1124', 'op': 'aten::sub', 'in': [51, 36, 0], 'output_id': 0, 'shape': [400, 784], 'out': [129], 'sorted_id': 128}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[128] = op;
            
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1125', 'op': 'aten::pow', 'in': [128, 25], 'output_id': 0, 'shape': [400, 784], 'out': [134], 'sorted_id': 129}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[129] = op;
            
            op->set_inputs( forward_result[128] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1126', 'op': 'aten::exp', 'in': [37], 'output_id': 0, 'shape': [400, 784], 'out': [131], 'sorted_id': 130}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[130] = op;
            
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1127', 'op': 'aten::log1p', 'in': [130], 'output_id': 0, 'shape': [400, 784], 'out': [132], 'sorted_id': 131}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[131] = op;
            
            op->set_inputs( forward_result[130] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1128', 'op': 'aten::pow', 'in': [131, 25], 'output_id': 0, 'shape': [400, 784], 'out': [133], 'sorted_id': 132}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[132] = op;
            
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1129', 'op': 'aten::mul', 'in': [132, 72], 'output_id': 0, 'shape': [400, 784], 'out': [134], 'sorted_id': 133}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[133] = op;
            
            op->set_inputs( forward_result[132] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1130', 'op': 'aten::div', 'in': [129, 133], 'output_id': 0, 'shape': [400, 784], 'out': [135], 'sorted_id': 134}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[134] = op;
            
            op->set_inputs( forward_result[129] );
            op->set_inputs( forward_result[133] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1131', 'op': 'aten::sub', 'in': [127, 134, 0], 'output_id': 0, 'shape': [400, 784], 'out': [136], 'sorted_id': 135}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[135] = op;
            
            op->set_inputs( forward_result[127] );
            op->set_inputs( forward_result[134] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1132', 'op': 'aten::sum', 'in': [135, 3], 'output_id': 0, 'shape': [], 'out': [150], 'sorted_id': 136}
        {
            SumOp* op = new SumOp();
            forward_result[136] = op;
            
            op->set_inputs( forward_result[135] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1133', 'op': 'aten::exp', 'in': [53], 'output_id': 0, 'shape': [400], 'out': [138], 'sorted_id': 137}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[137] = op;
            
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1134', 'op': 'aten::log1p', 'in': [137], 'output_id': 0, 'shape': [400], 'out': [139], 'sorted_id': 138}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[138] = op;
            
            op->set_inputs( forward_result[137] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1135', 'op': 'aten::log', 'in': [138], 'output_id': 0, 'shape': [400], 'out': [140], 'sorted_id': 139}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[139] = op;
            
            op->set_inputs( forward_result[138] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1136', 'op': 'aten::rsub', 'in': [139, 126, 0], 'output_id': 0, 'shape': [400], 'out': [148], 'sorted_id': 140}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[140] = op;
            
            op->set_inputs( forward_result[139] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1137', 'op': 'aten::sub', 'in': [64, 52, 0], 'output_id': 0, 'shape': [400], 'out': [142], 'sorted_id': 141}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[141] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1138', 'op': 'aten::pow', 'in': [141, 25], 'output_id': 0, 'shape': [400], 'out': [147], 'sorted_id': 142}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[142] = op;
            
            op->set_inputs( forward_result[141] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1139', 'op': 'aten::exp', 'in': [53], 'output_id': 0, 'shape': [400], 'out': [144], 'sorted_id': 143}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[143] = op;
            
            op->set_inputs( forward_result[53] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1140', 'op': 'aten::log1p', 'in': [143], 'output_id': 0, 'shape': [400], 'out': [145], 'sorted_id': 144}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[144] = op;
            
            op->set_inputs( forward_result[143] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1141', 'op': 'aten::pow', 'in': [144, 25], 'output_id': 0, 'shape': [400], 'out': [146], 'sorted_id': 145}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[145] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1142', 'op': 'aten::mul', 'in': [145, 72], 'output_id': 0, 'shape': [400], 'out': [147], 'sorted_id': 146}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[146] = op;
            
            op->set_inputs( forward_result[145] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1143', 'op': 'aten::div', 'in': [142, 146], 'output_id': 0, 'shape': [400], 'out': [148], 'sorted_id': 147}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[147] = op;
            
            op->set_inputs( forward_result[142] );
            op->set_inputs( forward_result[146] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1144', 'op': 'aten::sub', 'in': [140, 147, 0], 'output_id': 0, 'shape': [400], 'out': [149], 'sorted_id': 148}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[148] = op;
            
            op->set_inputs( forward_result[140] );
            op->set_inputs( forward_result[147] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1145', 'op': 'aten::sum', 'in': [148, 3], 'output_id': 0, 'shape': [], 'out': [150], 'sorted_id': 149}
        {
            SumOp* op = new SumOp();
            forward_result[149] = op;
            
            op->set_inputs( forward_result[148] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l1]/1146', 'op': 'aten::add', 'in': [136, 149, 0], 'output_id': 0, 'shape': [], 'out': [151], 'sorted_id': 150}
        {
            AddOp* op = new AddOp();
            forward_result[150] = op;
            
            op->set_inputs( forward_result[136] );
            op->set_inputs( forward_result[149] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/1148', 'op': 'prim::TupleConstruct', 'in': [65, 122, 150], 'output_id': 0, 'shape': [], 'out': [375, 152, 372], 'sorted_id': 151}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[151] = op;
            
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[122] );
            op->set_inputs( forward_result[150] );
        }
        
        // {'name': 'Net/1149', 'op': 'prim::TupleUnpack', 'in': [151], 'output_id': 0, 'shape': [4, 400], 'out': [153], 'sorted_id': 152}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[152] = op;
            
            op->set_inputs( forward_result[151] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.5', 'op': 'aten::relu', 'in': [152], 'output_id': 0, 'shape': [4, 400], 'out': [181], 'sorted_id': 153}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[153] = op;
            
            op->set_inputs( forward_result[152] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/weight_mu/weight_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [237, 167], 'sorted_id': 154}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[154] = new VariableTensor( l2_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/weight_rho/weight_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [159, 158, 239, 156, 233], 'sorted_id': 155}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[155] = new VariableTensor( l2_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1165', 'op': 'aten::exp', 'in': [155], 'output_id': 0, 'shape': [400, 400], 'out': [157], 'sorted_id': 156}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[156] = op;
            
            op->set_inputs( forward_result[155] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1166', 'op': 'aten::log1p', 'in': [156], 'output_id': 0, 'shape': [400, 400], 'out': [166], 'sorted_id': 157}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[157] = op;
            
            op->set_inputs( forward_result[156] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1157', 'op': 'aten::size', 'in': [155, 7], 'output_id': 0, 'shape': [], 'out': [160, 162], 'sorted_id': 158}
        {
            SizeOp* op = new SizeOp();
            forward_result[158] = op;
            
            op->set_inputs( forward_result[155] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1158', 'op': 'aten::size', 'in': [155, 0], 'output_id': 0, 'shape': [], 'out': [160, 162], 'sorted_id': 159}
        {
            SizeOp* op = new SizeOp();
            forward_result[159] = op;
            
            op->set_inputs( forward_result[155] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1159', 'op': 'prim::ListConstruct', 'in': [158, 159], 'output_id': 0, 'shape': [], 'out': [161], 'sorted_id': 160}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[160] = op;
            
            op->set_inputs( forward_result[158] );
            op->set_inputs( forward_result[159] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1160', 'op': 'aten::expand', 'in': [40, 160, 5], 'output_id': 0, 'shape': [400, 400], 'out': [164], 'sorted_id': 161}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[161] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[160] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1161', 'op': 'prim::ListConstruct', 'in': [158, 159], 'output_id': 0, 'shape': [], 'out': [163], 'sorted_id': 162}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[162] = op;
            
            op->set_inputs( forward_result[158] );
            op->set_inputs( forward_result[159] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1162', 'op': 'aten::expand', 'in': [45, 162, 5], 'output_id': 0, 'shape': [400, 400], 'out': [164], 'sorted_id': 163}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[163] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[162] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1163', 'op': 'aten::normal', 'in': [161, 163, 3], 'output_id': 0, 'shape': [400, 400], 'out': [165], 'sorted_id': 164}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[164] = op;
            
            op->set_inputs( forward_result[161] );
            op->set_inputs( forward_result[163] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/epsilon.5', 'op': 'aten::to', 'in': [164, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [400, 400], 'out': [166], 'sorted_id': 165}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[165] = op;
            
            op->set_inputs( forward_result[164] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1167', 'op': 'aten::mul', 'in': [157, 165], 'output_id': 0, 'shape': [400, 400], 'out': [167], 'sorted_id': 166}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[166] = op;
            
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[165] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/value.5', 'op': 'aten::add', 'in': [154, 166, 0], 'output_id': 0, 'shape': [400, 400], 'out': [237, 182, 181, 193], 'sorted_id': 167}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[167] = op;
            
            op->set_inputs( forward_result[154] );
            op->set_inputs( forward_result[166] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/bias_mu/bias_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [250, 180], 'sorted_id': 168}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[168] = new VariableTensor( l2_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l2]/bias_rho/bias_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [172, 246, 252, 170], 'sorted_id': 169}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[169] = new VariableTensor( l2_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1176', 'op': 'aten::exp', 'in': [169], 'output_id': 0, 'shape': [400], 'out': [171], 'sorted_id': 170}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[170] = op;
            
            op->set_inputs( forward_result[169] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1177', 'op': 'aten::log1p', 'in': [170], 'output_id': 0, 'shape': [400], 'out': [179], 'sorted_id': 171}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[171] = op;
            
            op->set_inputs( forward_result[170] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1169', 'op': 'aten::size', 'in': [169, 7], 'output_id': 0, 'shape': [], 'out': [173, 175], 'sorted_id': 172}
        {
            SizeOp* op = new SizeOp();
            forward_result[172] = op;
            
            op->set_inputs( forward_result[169] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1170', 'op': 'prim::ListConstruct', 'in': [172], 'output_id': 0, 'shape': [], 'out': [174], 'sorted_id': 173}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[173] = op;
            
            op->set_inputs( forward_result[172] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1171', 'op': 'aten::expand', 'in': [40, 173, 5], 'output_id': 0, 'shape': [400], 'out': [177], 'sorted_id': 174}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[174] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[173] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1172', 'op': 'prim::ListConstruct', 'in': [172], 'output_id': 0, 'shape': [], 'out': [176], 'sorted_id': 175}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[175] = op;
            
            op->set_inputs( forward_result[172] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1173', 'op': 'aten::expand', 'in': [45, 175, 5], 'output_id': 0, 'shape': [400], 'out': [177], 'sorted_id': 176}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[176] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[175] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1174', 'op': 'aten::normal', 'in': [174, 176, 3], 'output_id': 0, 'shape': [400], 'out': [178], 'sorted_id': 177}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[177] = op;
            
            op->set_inputs( forward_result[174] );
            op->set_inputs( forward_result[176] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/epsilon.7', 'op': 'aten::to', 'in': [177, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [400], 'out': [179], 'sorted_id': 178}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[178] = op;
            
            op->set_inputs( forward_result[177] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1178', 'op': 'aten::mul', 'in': [171, 178], 'output_id': 0, 'shape': [400], 'out': [180], 'sorted_id': 179}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[179] = op;
            
            op->set_inputs( forward_result[171] );
            op->set_inputs( forward_result[178] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/value.7', 'op': 'aten::add', 'in': [168, 179, 0], 'output_id': 0, 'shape': [400], 'out': [181, 250, 207, 218], 'sorted_id': 180}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[180] = op;
            
            op->set_inputs( forward_result[168] );
            op->set_inputs( forward_result[179] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/input.7', 'op': 'aten::linear', 'in': [153, 167, 180], 'output_id': 0, 'shape': [4, 400], 'out': [260], 'sorted_id': 181}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[181] = op;
            
            op->set_inputs( forward_result[153] );
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[180] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1182', 'op': 'aten::sub', 'in': [167, 66, 0], 'output_id': 0, 'shape': [400, 400], 'out': [183], 'sorted_id': 182}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[182] = op;
            
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1183', 'op': 'aten::pow', 'in': [182, 25], 'output_id': 0, 'shape': [400, 400], 'out': [184], 'sorted_id': 183}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[183] = op;
            
            op->set_inputs( forward_result[182] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1184', 'op': 'aten::neg', 'in': [183], 'output_id': 0, 'shape': [400, 400], 'out': [187], 'sorted_id': 184}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[184] = op;
            
            op->set_inputs( forward_result[183] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.9', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [186], 'sorted_id': 185}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[185] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1185', 'op': 'aten::mul', 'in': [185, 72], 'output_id': 0, 'shape': [1], 'out': [187], 'sorted_id': 186}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[186] = op;
            
            op->set_inputs( forward_result[185] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1186', 'op': 'aten::div', 'in': [184, 186], 'output_id': 0, 'shape': [400, 400], 'out': [189], 'sorted_id': 187}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[187] = op;
            
            op->set_inputs( forward_result[184] );
            op->set_inputs( forward_result[186] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.9', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [189], 'sorted_id': 188}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[188] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1187', 'op': 'aten::sub', 'in': [187, 188, 0], 'output_id': 0, 'shape': [400, 400], 'out': [190], 'sorted_id': 189}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[189] = op;
            
            op->set_inputs( forward_result[187] );
            op->set_inputs( forward_result[188] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1188', 'op': 'aten::sub', 'in': [189, 77, 0], 'output_id': 0, 'shape': [400, 400], 'out': [191], 'sorted_id': 190}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[190] = op;
            
            op->set_inputs( forward_result[189] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob1.5', 'op': 'aten::exp', 'in': [190], 'output_id': 0, 'shape': [400, 400], 'out': [192], 'sorted_id': 191}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[191] = op;
            
            op->set_inputs( forward_result[190] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1200', 'op': 'aten::mul', 'in': [191, 80], 'output_id': 0, 'shape': [400, 400], 'out': [204], 'sorted_id': 192}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[192] = op;
            
            op->set_inputs( forward_result[191] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1192', 'op': 'aten::sub', 'in': [167, 66, 0], 'output_id': 0, 'shape': [400, 400], 'out': [194], 'sorted_id': 193}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[193] = op;
            
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1193', 'op': 'aten::pow', 'in': [193, 25], 'output_id': 0, 'shape': [400, 400], 'out': [195], 'sorted_id': 194}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[194] = op;
            
            op->set_inputs( forward_result[193] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1194', 'op': 'aten::neg', 'in': [194], 'output_id': 0, 'shape': [400, 400], 'out': [198], 'sorted_id': 195}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[195] = op;
            
            op->set_inputs( forward_result[194] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.11', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [197], 'sorted_id': 196}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[196] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1195', 'op': 'aten::mul', 'in': [196, 72], 'output_id': 0, 'shape': [1], 'out': [198], 'sorted_id': 197}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[197] = op;
            
            op->set_inputs( forward_result[196] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1196', 'op': 'aten::div', 'in': [195, 197], 'output_id': 0, 'shape': [400, 400], 'out': [200], 'sorted_id': 198}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[198] = op;
            
            op->set_inputs( forward_result[195] );
            op->set_inputs( forward_result[197] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.11', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [200], 'sorted_id': 199}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[199] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1197', 'op': 'aten::sub', 'in': [198, 199, 0], 'output_id': 0, 'shape': [400, 400], 'out': [201], 'sorted_id': 200}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[200] = op;
            
            op->set_inputs( forward_result[198] );
            op->set_inputs( forward_result[199] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1198', 'op': 'aten::sub', 'in': [200, 77, 0], 'output_id': 0, 'shape': [400, 400], 'out': [202], 'sorted_id': 201}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[201] = op;
            
            op->set_inputs( forward_result[200] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob2.5', 'op': 'aten::exp', 'in': [201], 'output_id': 0, 'shape': [400, 400], 'out': [203], 'sorted_id': 202}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[202] = op;
            
            op->set_inputs( forward_result[201] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1201', 'op': 'aten::mul', 'in': [202, 80], 'output_id': 0, 'shape': [400, 400], 'out': [204], 'sorted_id': 203}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[203] = op;
            
            op->set_inputs( forward_result[202] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1202', 'op': 'aten::add', 'in': [192, 203, 0], 'output_id': 0, 'shape': [400, 400], 'out': [205], 'sorted_id': 204}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[204] = op;
            
            op->set_inputs( forward_result[192] );
            op->set_inputs( forward_result[203] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1203', 'op': 'aten::log', 'in': [204], 'output_id': 0, 'shape': [400, 400], 'out': [206], 'sorted_id': 205}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[205] = op;
            
            op->set_inputs( forward_result[204] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1204', 'op': 'aten::sum', 'in': [205, 3], 'output_id': 0, 'shape': [], 'out': [232], 'sorted_id': 206}
        {
            SumOp* op = new SumOp();
            forward_result[206] = op;
            
            op->set_inputs( forward_result[205] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1207', 'op': 'aten::sub', 'in': [180, 66, 0], 'output_id': 0, 'shape': [400], 'out': [208], 'sorted_id': 207}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[207] = op;
            
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1208', 'op': 'aten::pow', 'in': [207, 25], 'output_id': 0, 'shape': [400], 'out': [209], 'sorted_id': 208}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[208] = op;
            
            op->set_inputs( forward_result[207] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1209', 'op': 'aten::neg', 'in': [208], 'output_id': 0, 'shape': [400], 'out': [212], 'sorted_id': 209}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[209] = op;
            
            op->set_inputs( forward_result[208] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.13', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [211], 'sorted_id': 210}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[210] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1210', 'op': 'aten::mul', 'in': [210, 72], 'output_id': 0, 'shape': [1], 'out': [212], 'sorted_id': 211}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[211] = op;
            
            op->set_inputs( forward_result[210] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1211', 'op': 'aten::div', 'in': [209, 211], 'output_id': 0, 'shape': [400], 'out': [214], 'sorted_id': 212}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[212] = op;
            
            op->set_inputs( forward_result[209] );
            op->set_inputs( forward_result[211] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.13', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [214], 'sorted_id': 213}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[213] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1212', 'op': 'aten::sub', 'in': [212, 213, 0], 'output_id': 0, 'shape': [400], 'out': [215], 'sorted_id': 214}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[214] = op;
            
            op->set_inputs( forward_result[212] );
            op->set_inputs( forward_result[213] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1213', 'op': 'aten::sub', 'in': [214, 77, 0], 'output_id': 0, 'shape': [400], 'out': [216], 'sorted_id': 215}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[215] = op;
            
            op->set_inputs( forward_result[214] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob1.7', 'op': 'aten::exp', 'in': [215], 'output_id': 0, 'shape': [400], 'out': [217], 'sorted_id': 216}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[216] = op;
            
            op->set_inputs( forward_result[215] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1225', 'op': 'aten::mul', 'in': [216, 80], 'output_id': 0, 'shape': [400], 'out': [229], 'sorted_id': 217}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[217] = op;
            
            op->set_inputs( forward_result[216] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1217', 'op': 'aten::sub', 'in': [180, 66, 0], 'output_id': 0, 'shape': [400], 'out': [219], 'sorted_id': 218}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[218] = op;
            
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1218', 'op': 'aten::pow', 'in': [218, 25], 'output_id': 0, 'shape': [400], 'out': [220], 'sorted_id': 219}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[219] = op;
            
            op->set_inputs( forward_result[218] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1219', 'op': 'aten::neg', 'in': [219], 'output_id': 0, 'shape': [400], 'out': [223], 'sorted_id': 220}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[220] = op;
            
            op->set_inputs( forward_result[219] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/var.15', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [222], 'sorted_id': 221}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[221] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1220', 'op': 'aten::mul', 'in': [221, 72], 'output_id': 0, 'shape': [1], 'out': [223], 'sorted_id': 222}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[222] = op;
            
            op->set_inputs( forward_result[221] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1221', 'op': 'aten::div', 'in': [220, 222], 'output_id': 0, 'shape': [400], 'out': [225], 'sorted_id': 223}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[223] = op;
            
            op->set_inputs( forward_result[220] );
            op->set_inputs( forward_result[222] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/log_scale.15', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [225], 'sorted_id': 224}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[224] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1222', 'op': 'aten::sub', 'in': [223, 224, 0], 'output_id': 0, 'shape': [400], 'out': [226], 'sorted_id': 225}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[225] = op;
            
            op->set_inputs( forward_result[223] );
            op->set_inputs( forward_result[224] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1223', 'op': 'aten::sub', 'in': [225, 77, 0], 'output_id': 0, 'shape': [400], 'out': [227], 'sorted_id': 226}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[226] = op;
            
            op->set_inputs( forward_result[225] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/prob2.7', 'op': 'aten::exp', 'in': [226], 'output_id': 0, 'shape': [400], 'out': [228], 'sorted_id': 227}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[227] = op;
            
            op->set_inputs( forward_result[226] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1226', 'op': 'aten::mul', 'in': [227, 80], 'output_id': 0, 'shape': [400], 'out': [229], 'sorted_id': 228}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[228] = op;
            
            op->set_inputs( forward_result[227] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1227', 'op': 'aten::add', 'in': [217, 228, 0], 'output_id': 0, 'shape': [400], 'out': [230], 'sorted_id': 229}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[229] = op;
            
            op->set_inputs( forward_result[217] );
            op->set_inputs( forward_result[228] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1228', 'op': 'aten::log', 'in': [229], 'output_id': 0, 'shape': [400], 'out': [231], 'sorted_id': 230}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[230] = op;
            
            op->set_inputs( forward_result[229] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1229', 'op': 'aten::sum', 'in': [230, 3], 'output_id': 0, 'shape': [], 'out': [232], 'sorted_id': 231}
        {
            SumOp* op = new SumOp();
            forward_result[231] = op;
            
            op->set_inputs( forward_result[230] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1230', 'op': 'aten::add', 'in': [206, 231, 0], 'output_id': 0, 'shape': [], 'out': [260], 'sorted_id': 232}
        {
            AddOp* op = new AddOp();
            forward_result[232] = op;
            
            op->set_inputs( forward_result[206] );
            op->set_inputs( forward_result[231] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1231', 'op': 'aten::exp', 'in': [155], 'output_id': 0, 'shape': [400, 400], 'out': [234], 'sorted_id': 233}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[233] = op;
            
            op->set_inputs( forward_result[155] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1232', 'op': 'aten::log1p', 'in': [233], 'output_id': 0, 'shape': [400, 400], 'out': [235], 'sorted_id': 234}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[234] = op;
            
            op->set_inputs( forward_result[233] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1233', 'op': 'aten::log', 'in': [234], 'output_id': 0, 'shape': [400, 400], 'out': [236], 'sorted_id': 235}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[235] = op;
            
            op->set_inputs( forward_result[234] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1234', 'op': 'aten::rsub', 'in': [235, 126, 0], 'output_id': 0, 'shape': [400, 400], 'out': [244], 'sorted_id': 236}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[236] = op;
            
            op->set_inputs( forward_result[235] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1235', 'op': 'aten::sub', 'in': [167, 154, 0], 'output_id': 0, 'shape': [400, 400], 'out': [238], 'sorted_id': 237}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[237] = op;
            
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[154] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1236', 'op': 'aten::pow', 'in': [237, 25], 'output_id': 0, 'shape': [400, 400], 'out': [243], 'sorted_id': 238}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[238] = op;
            
            op->set_inputs( forward_result[237] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1237', 'op': 'aten::exp', 'in': [155], 'output_id': 0, 'shape': [400, 400], 'out': [240], 'sorted_id': 239}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[239] = op;
            
            op->set_inputs( forward_result[155] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1238', 'op': 'aten::log1p', 'in': [239], 'output_id': 0, 'shape': [400, 400], 'out': [241], 'sorted_id': 240}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[240] = op;
            
            op->set_inputs( forward_result[239] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1239', 'op': 'aten::pow', 'in': [240, 25], 'output_id': 0, 'shape': [400, 400], 'out': [242], 'sorted_id': 241}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[241] = op;
            
            op->set_inputs( forward_result[240] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1240', 'op': 'aten::mul', 'in': [241, 72], 'output_id': 0, 'shape': [400, 400], 'out': [243], 'sorted_id': 242}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[242] = op;
            
            op->set_inputs( forward_result[241] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1241', 'op': 'aten::div', 'in': [238, 242], 'output_id': 0, 'shape': [400, 400], 'out': [244], 'sorted_id': 243}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[243] = op;
            
            op->set_inputs( forward_result[238] );
            op->set_inputs( forward_result[242] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1242', 'op': 'aten::sub', 'in': [236, 243, 0], 'output_id': 0, 'shape': [400, 400], 'out': [245], 'sorted_id': 244}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[244] = op;
            
            op->set_inputs( forward_result[236] );
            op->set_inputs( forward_result[243] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1243', 'op': 'aten::sum', 'in': [244, 3], 'output_id': 0, 'shape': [], 'out': [259], 'sorted_id': 245}
        {
            SumOp* op = new SumOp();
            forward_result[245] = op;
            
            op->set_inputs( forward_result[244] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1244', 'op': 'aten::exp', 'in': [169], 'output_id': 0, 'shape': [400], 'out': [247], 'sorted_id': 246}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[246] = op;
            
            op->set_inputs( forward_result[169] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1245', 'op': 'aten::log1p', 'in': [246], 'output_id': 0, 'shape': [400], 'out': [248], 'sorted_id': 247}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[247] = op;
            
            op->set_inputs( forward_result[246] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1246', 'op': 'aten::log', 'in': [247], 'output_id': 0, 'shape': [400], 'out': [249], 'sorted_id': 248}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[248] = op;
            
            op->set_inputs( forward_result[247] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1247', 'op': 'aten::rsub', 'in': [248, 126, 0], 'output_id': 0, 'shape': [400], 'out': [257], 'sorted_id': 249}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[249] = op;
            
            op->set_inputs( forward_result[248] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1248', 'op': 'aten::sub', 'in': [180, 168, 0], 'output_id': 0, 'shape': [400], 'out': [251], 'sorted_id': 250}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[250] = op;
            
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[168] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1249', 'op': 'aten::pow', 'in': [250, 25], 'output_id': 0, 'shape': [400], 'out': [256], 'sorted_id': 251}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[251] = op;
            
            op->set_inputs( forward_result[250] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1250', 'op': 'aten::exp', 'in': [169], 'output_id': 0, 'shape': [400], 'out': [253], 'sorted_id': 252}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[252] = op;
            
            op->set_inputs( forward_result[169] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1251', 'op': 'aten::log1p', 'in': [252], 'output_id': 0, 'shape': [400], 'out': [254], 'sorted_id': 253}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[253] = op;
            
            op->set_inputs( forward_result[252] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1252', 'op': 'aten::pow', 'in': [253, 25], 'output_id': 0, 'shape': [400], 'out': [255], 'sorted_id': 254}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[254] = op;
            
            op->set_inputs( forward_result[253] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1253', 'op': 'aten::mul', 'in': [254, 72], 'output_id': 0, 'shape': [400], 'out': [256], 'sorted_id': 255}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[255] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1254', 'op': 'aten::div', 'in': [251, 255], 'output_id': 0, 'shape': [400], 'out': [257], 'sorted_id': 256}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[256] = op;
            
            op->set_inputs( forward_result[251] );
            op->set_inputs( forward_result[255] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1255', 'op': 'aten::sub', 'in': [249, 256, 0], 'output_id': 0, 'shape': [400], 'out': [258], 'sorted_id': 257}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[257] = op;
            
            op->set_inputs( forward_result[249] );
            op->set_inputs( forward_result[256] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1256', 'op': 'aten::sum', 'in': [257, 3], 'output_id': 0, 'shape': [], 'out': [259], 'sorted_id': 258}
        {
            SumOp* op = new SumOp();
            forward_result[258] = op;
            
            op->set_inputs( forward_result[257] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l2]/1257', 'op': 'aten::add', 'in': [245, 258, 0], 'output_id': 0, 'shape': [], 'out': [260], 'sorted_id': 259}
        {
            AddOp* op = new AddOp();
            forward_result[259] = op;
            
            op->set_inputs( forward_result[245] );
            op->set_inputs( forward_result[258] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/1259', 'op': 'prim::TupleConstruct', 'in': [181, 232, 259], 'output_id': 0, 'shape': [], 'out': [373, 376, 261], 'sorted_id': 260}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[260] = op;
            
            op->set_inputs( forward_result[181] );
            op->set_inputs( forward_result[232] );
            op->set_inputs( forward_result[259] );
        }
        
        // {'name': 'Net/1260', 'op': 'prim::TupleUnpack', 'in': [260], 'output_id': 0, 'shape': [4, 400], 'out': [262], 'sorted_id': 261}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[261] = op;
            
            op->set_inputs( forward_result[260] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/input.9', 'op': 'aten::relu', 'in': [261], 'output_id': 0, 'shape': [4, 400], 'out': [290], 'sorted_id': 262}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[262] = op;
            
            op->set_inputs( forward_result[261] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/weight_mu/weight_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [346, 276], 'sorted_id': 263}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[263] = new VariableTensor( l3_weight_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/weight_rho/weight_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [348, 268, 342, 265, 267], 'sorted_id': 264}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[264] = new VariableTensor( l3_weight_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1276', 'op': 'aten::exp', 'in': [264], 'output_id': 0, 'shape': [10, 400], 'out': [266], 'sorted_id': 265}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[265] = op;
            
            op->set_inputs( forward_result[264] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1277', 'op': 'aten::log1p', 'in': [265], 'output_id': 0, 'shape': [10, 400], 'out': [275], 'sorted_id': 266}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[266] = op;
            
            op->set_inputs( forward_result[265] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1268', 'op': 'aten::size', 'in': [264, 7], 'output_id': 0, 'shape': [], 'out': [271, 269], 'sorted_id': 267}
        {
            SizeOp* op = new SizeOp();
            forward_result[267] = op;
            
            op->set_inputs( forward_result[264] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1269', 'op': 'aten::size', 'in': [264, 0], 'output_id': 0, 'shape': [], 'out': [271, 269], 'sorted_id': 268}
        {
            SizeOp* op = new SizeOp();
            forward_result[268] = op;
            
            op->set_inputs( forward_result[264] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1270', 'op': 'prim::ListConstruct', 'in': [267, 268], 'output_id': 0, 'shape': [], 'out': [270], 'sorted_id': 269}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[269] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[268] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1271', 'op': 'aten::expand', 'in': [40, 269, 5], 'output_id': 0, 'shape': [10, 400], 'out': [273], 'sorted_id': 270}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[270] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1272', 'op': 'prim::ListConstruct', 'in': [267, 268], 'output_id': 0, 'shape': [], 'out': [272], 'sorted_id': 271}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[271] = op;
            
            op->set_inputs( forward_result[267] );
            op->set_inputs( forward_result[268] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1273', 'op': 'aten::expand', 'in': [45, 271, 5], 'output_id': 0, 'shape': [10, 400], 'out': [273], 'sorted_id': 272}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[272] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[271] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1274', 'op': 'aten::normal', 'in': [270, 272, 3], 'output_id': 0, 'shape': [10, 400], 'out': [274], 'sorted_id': 273}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[273] = op;
            
            op->set_inputs( forward_result[270] );
            op->set_inputs( forward_result[272] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/epsilon.9', 'op': 'aten::to', 'in': [273, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [10, 400], 'out': [275], 'sorted_id': 274}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[274] = op;
            
            op->set_inputs( forward_result[273] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1278', 'op': 'aten::mul', 'in': [266, 274], 'output_id': 0, 'shape': [10, 400], 'out': [276], 'sorted_id': 275}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[275] = op;
            
            op->set_inputs( forward_result[266] );
            op->set_inputs( forward_result[274] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/value.9', 'op': 'aten::add', 'in': [263, 275, 0], 'output_id': 0, 'shape': [10, 400], 'out': [346, 291, 290, 302], 'sorted_id': 276}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[276] = op;
            
            op->set_inputs( forward_result[263] );
            op->set_inputs( forward_result[275] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/bias_mu/bias_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [289, 359], 'sorted_id': 277}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[277] = new VariableTensor( l3_bias_mu );
        }
        
        // {'name': 'Net/BayesianNetwork[net]/BayesianLinear[l3]/bias_rho/bias_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [361, 355, 281, 279], 'sorted_id': 278}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[278] = new VariableTensor( l3_bias_rho );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1287', 'op': 'aten::exp', 'in': [278], 'output_id': 0, 'shape': [10], 'out': [280], 'sorted_id': 279}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[279] = op;
            
            op->set_inputs( forward_result[278] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1288', 'op': 'aten::log1p', 'in': [279], 'output_id': 0, 'shape': [10], 'out': [288], 'sorted_id': 280}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[280] = op;
            
            op->set_inputs( forward_result[279] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1280', 'op': 'aten::size', 'in': [278, 7], 'output_id': 0, 'shape': [], 'out': [282, 284], 'sorted_id': 281}
        {
            SizeOp* op = new SizeOp();
            forward_result[281] = op;
            
            op->set_inputs( forward_result[278] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1281', 'op': 'prim::ListConstruct', 'in': [281], 'output_id': 0, 'shape': [], 'out': [283], 'sorted_id': 282}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[282] = op;
            
            op->set_inputs( forward_result[281] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1282', 'op': 'aten::expand', 'in': [40, 282, 5], 'output_id': 0, 'shape': [10], 'out': [286], 'sorted_id': 283}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[283] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[282] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1283', 'op': 'prim::ListConstruct', 'in': [281], 'output_id': 0, 'shape': [], 'out': [285], 'sorted_id': 284}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[284] = op;
            
            op->set_inputs( forward_result[281] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1284', 'op': 'aten::expand', 'in': [45, 284, 5], 'output_id': 0, 'shape': [10], 'out': [286], 'sorted_id': 285}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[285] = op;
            
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[284] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1285', 'op': 'aten::normal', 'in': [283, 285, 3], 'output_id': 0, 'shape': [10], 'out': [287], 'sorted_id': 286}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[286] = op;
            
            op->set_inputs( forward_result[283] );
            op->set_inputs( forward_result[285] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/epsilon', 'op': 'aten::to', 'in': [286, 2, 7, 4, 3, 5, 5, 3], 'output_id': 0, 'shape': [10], 'out': [288], 'sorted_id': 287}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[287] = op;
            
            op->set_inputs( forward_result[286] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1289', 'op': 'aten::mul', 'in': [280, 287], 'output_id': 0, 'shape': [10], 'out': [289], 'sorted_id': 288}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[288] = op;
            
            op->set_inputs( forward_result[280] );
            op->set_inputs( forward_result[287] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/value', 'op': 'aten::add', 'in': [277, 288, 0], 'output_id': 0, 'shape': [10], 'out': [290, 359, 327, 316], 'sorted_id': 289}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[289] = op;
            
            op->set_inputs( forward_result[277] );
            op->set_inputs( forward_result[288] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/input.11', 'op': 'aten::linear', 'in': [262, 276, 289], 'output_id': 0, 'shape': [4, 10], 'out': [369], 'sorted_id': 290}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[290] = op;
            
            op->set_inputs( forward_result[262] );
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[289] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1293', 'op': 'aten::sub', 'in': [276, 66, 0], 'output_id': 0, 'shape': [10, 400], 'out': [292], 'sorted_id': 291}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[291] = op;
            
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1294', 'op': 'aten::pow', 'in': [291, 25], 'output_id': 0, 'shape': [10, 400], 'out': [293], 'sorted_id': 292}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[292] = op;
            
            op->set_inputs( forward_result[291] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1295', 'op': 'aten::neg', 'in': [292], 'output_id': 0, 'shape': [10, 400], 'out': [296], 'sorted_id': 293}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[293] = op;
            
            op->set_inputs( forward_result[292] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.17', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [295], 'sorted_id': 294}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[294] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1296', 'op': 'aten::mul', 'in': [294, 72], 'output_id': 0, 'shape': [1], 'out': [296], 'sorted_id': 295}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[295] = op;
            
            op->set_inputs( forward_result[294] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1297', 'op': 'aten::div', 'in': [293, 295], 'output_id': 0, 'shape': [10, 400], 'out': [298], 'sorted_id': 296}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[296] = op;
            
            op->set_inputs( forward_result[293] );
            op->set_inputs( forward_result[295] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.17', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [298], 'sorted_id': 297}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[297] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1298', 'op': 'aten::sub', 'in': [296, 297, 0], 'output_id': 0, 'shape': [10, 400], 'out': [299], 'sorted_id': 298}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[298] = op;
            
            op->set_inputs( forward_result[296] );
            op->set_inputs( forward_result[297] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1299', 'op': 'aten::sub', 'in': [298, 77, 0], 'output_id': 0, 'shape': [10, 400], 'out': [300], 'sorted_id': 299}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[299] = op;
            
            op->set_inputs( forward_result[298] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob1.9', 'op': 'aten::exp', 'in': [299], 'output_id': 0, 'shape': [10, 400], 'out': [301], 'sorted_id': 300}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[300] = op;
            
            op->set_inputs( forward_result[299] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1311', 'op': 'aten::mul', 'in': [300, 80], 'output_id': 0, 'shape': [10, 400], 'out': [313], 'sorted_id': 301}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[301] = op;
            
            op->set_inputs( forward_result[300] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1303', 'op': 'aten::sub', 'in': [276, 66, 0], 'output_id': 0, 'shape': [10, 400], 'out': [303], 'sorted_id': 302}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[302] = op;
            
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1304', 'op': 'aten::pow', 'in': [302, 25], 'output_id': 0, 'shape': [10, 400], 'out': [304], 'sorted_id': 303}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[303] = op;
            
            op->set_inputs( forward_result[302] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1305', 'op': 'aten::neg', 'in': [303], 'output_id': 0, 'shape': [10, 400], 'out': [307], 'sorted_id': 304}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[304] = op;
            
            op->set_inputs( forward_result[303] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.19', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [306], 'sorted_id': 305}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[305] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1306', 'op': 'aten::mul', 'in': [305, 72], 'output_id': 0, 'shape': [1], 'out': [307], 'sorted_id': 306}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[306] = op;
            
            op->set_inputs( forward_result[305] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1307', 'op': 'aten::div', 'in': [304, 306], 'output_id': 0, 'shape': [10, 400], 'out': [309], 'sorted_id': 307}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[307] = op;
            
            op->set_inputs( forward_result[304] );
            op->set_inputs( forward_result[306] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.19', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [309], 'sorted_id': 308}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[308] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1308', 'op': 'aten::sub', 'in': [307, 308, 0], 'output_id': 0, 'shape': [10, 400], 'out': [310], 'sorted_id': 309}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[309] = op;
            
            op->set_inputs( forward_result[307] );
            op->set_inputs( forward_result[308] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1309', 'op': 'aten::sub', 'in': [309, 77, 0], 'output_id': 0, 'shape': [10, 400], 'out': [311], 'sorted_id': 310}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[310] = op;
            
            op->set_inputs( forward_result[309] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob2.9', 'op': 'aten::exp', 'in': [310], 'output_id': 0, 'shape': [10, 400], 'out': [312], 'sorted_id': 311}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[311] = op;
            
            op->set_inputs( forward_result[310] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1312', 'op': 'aten::mul', 'in': [311, 80], 'output_id': 0, 'shape': [10, 400], 'out': [313], 'sorted_id': 312}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[312] = op;
            
            op->set_inputs( forward_result[311] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1313', 'op': 'aten::add', 'in': [301, 312, 0], 'output_id': 0, 'shape': [10, 400], 'out': [314], 'sorted_id': 313}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[313] = op;
            
            op->set_inputs( forward_result[301] );
            op->set_inputs( forward_result[312] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1314', 'op': 'aten::log', 'in': [313], 'output_id': 0, 'shape': [10, 400], 'out': [315], 'sorted_id': 314}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[314] = op;
            
            op->set_inputs( forward_result[313] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1315', 'op': 'aten::sum', 'in': [314, 3], 'output_id': 0, 'shape': [], 'out': [341], 'sorted_id': 315}
        {
            SumOp* op = new SumOp();
            forward_result[315] = op;
            
            op->set_inputs( forward_result[314] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1318', 'op': 'aten::sub', 'in': [289, 66, 0], 'output_id': 0, 'shape': [10], 'out': [317], 'sorted_id': 316}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[316] = op;
            
            op->set_inputs( forward_result[289] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1319', 'op': 'aten::pow', 'in': [316, 25], 'output_id': 0, 'shape': [10], 'out': [318], 'sorted_id': 317}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[317] = op;
            
            op->set_inputs( forward_result[316] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1320', 'op': 'aten::neg', 'in': [317], 'output_id': 0, 'shape': [10], 'out': [321], 'sorted_id': 318}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[318] = op;
            
            op->set_inputs( forward_result[317] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var.21', 'op': 'aten::pow', 'in': [70, 25], 'output_id': 0, 'shape': [1], 'out': [320], 'sorted_id': 319}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[319] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1321', 'op': 'aten::mul', 'in': [319, 72], 'output_id': 0, 'shape': [1], 'out': [321], 'sorted_id': 320}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[320] = op;
            
            op->set_inputs( forward_result[319] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1322', 'op': 'aten::div', 'in': [318, 320], 'output_id': 0, 'shape': [10], 'out': [323], 'sorted_id': 321}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[321] = op;
            
            op->set_inputs( forward_result[318] );
            op->set_inputs( forward_result[320] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale.21', 'op': 'aten::log', 'in': [70], 'output_id': 0, 'shape': [1], 'out': [323], 'sorted_id': 322}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[322] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1323', 'op': 'aten::sub', 'in': [321, 322, 0], 'output_id': 0, 'shape': [10], 'out': [324], 'sorted_id': 323}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[323] = op;
            
            op->set_inputs( forward_result[321] );
            op->set_inputs( forward_result[322] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1324', 'op': 'aten::sub', 'in': [323, 77, 0], 'output_id': 0, 'shape': [10], 'out': [325], 'sorted_id': 324}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[324] = op;
            
            op->set_inputs( forward_result[323] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob1', 'op': 'aten::exp', 'in': [324], 'output_id': 0, 'shape': [10], 'out': [326], 'sorted_id': 325}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[325] = op;
            
            op->set_inputs( forward_result[324] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1336', 'op': 'aten::mul', 'in': [325, 80], 'output_id': 0, 'shape': [10], 'out': [338], 'sorted_id': 326}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[326] = op;
            
            op->set_inputs( forward_result[325] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1328', 'op': 'aten::sub', 'in': [289, 66, 0], 'output_id': 0, 'shape': [10], 'out': [328], 'sorted_id': 327}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[327] = op;
            
            op->set_inputs( forward_result[289] );
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1329', 'op': 'aten::pow', 'in': [327, 25], 'output_id': 0, 'shape': [10], 'out': [329], 'sorted_id': 328}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[328] = op;
            
            op->set_inputs( forward_result[327] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1330', 'op': 'aten::neg', 'in': [328], 'output_id': 0, 'shape': [10], 'out': [332], 'sorted_id': 329}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[329] = op;
            
            op->set_inputs( forward_result[328] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/var', 'op': 'aten::pow', 'in': [85, 25], 'output_id': 0, 'shape': [1], 'out': [331], 'sorted_id': 330}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[330] = op;
            
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1331', 'op': 'aten::mul', 'in': [330, 72], 'output_id': 0, 'shape': [1], 'out': [332], 'sorted_id': 331}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[331] = op;
            
            op->set_inputs( forward_result[330] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1332', 'op': 'aten::div', 'in': [329, 331], 'output_id': 0, 'shape': [10], 'out': [334], 'sorted_id': 332}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[332] = op;
            
            op->set_inputs( forward_result[329] );
            op->set_inputs( forward_result[331] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/log_scale', 'op': 'aten::log', 'in': [85], 'output_id': 0, 'shape': [1], 'out': [334], 'sorted_id': 333}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[333] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1333', 'op': 'aten::sub', 'in': [332, 333, 0], 'output_id': 0, 'shape': [10], 'out': [335], 'sorted_id': 334}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[334] = op;
            
            op->set_inputs( forward_result[332] );
            op->set_inputs( forward_result[333] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1334', 'op': 'aten::sub', 'in': [334, 77, 0], 'output_id': 0, 'shape': [10], 'out': [336], 'sorted_id': 335}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[335] = op;
            
            op->set_inputs( forward_result[334] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/prob2', 'op': 'aten::exp', 'in': [335], 'output_id': 0, 'shape': [10], 'out': [337], 'sorted_id': 336}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[336] = op;
            
            op->set_inputs( forward_result[335] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1337', 'op': 'aten::mul', 'in': [336, 80], 'output_id': 0, 'shape': [10], 'out': [338], 'sorted_id': 337}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[337] = op;
            
            op->set_inputs( forward_result[336] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1338', 'op': 'aten::add', 'in': [326, 337, 0], 'output_id': 0, 'shape': [10], 'out': [339], 'sorted_id': 338}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[338] = op;
            
            op->set_inputs( forward_result[326] );
            op->set_inputs( forward_result[337] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1339', 'op': 'aten::log', 'in': [338], 'output_id': 0, 'shape': [10], 'out': [340], 'sorted_id': 339}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[339] = op;
            
            op->set_inputs( forward_result[338] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1340', 'op': 'aten::sum', 'in': [339, 3], 'output_id': 0, 'shape': [], 'out': [341], 'sorted_id': 340}
        {
            SumOp* op = new SumOp();
            forward_result[340] = op;
            
            op->set_inputs( forward_result[339] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1341', 'op': 'aten::add', 'in': [315, 340, 0], 'output_id': 0, 'shape': [], 'out': [369], 'sorted_id': 341}
        {
            AddOp* op = new AddOp();
            forward_result[341] = op;
            
            op->set_inputs( forward_result[315] );
            op->set_inputs( forward_result[340] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1342', 'op': 'aten::exp', 'in': [264], 'output_id': 0, 'shape': [10, 400], 'out': [343], 'sorted_id': 342}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[342] = op;
            
            op->set_inputs( forward_result[264] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1343', 'op': 'aten::log1p', 'in': [342], 'output_id': 0, 'shape': [10, 400], 'out': [344], 'sorted_id': 343}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[343] = op;
            
            op->set_inputs( forward_result[342] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1344', 'op': 'aten::log', 'in': [343], 'output_id': 0, 'shape': [10, 400], 'out': [345], 'sorted_id': 344}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[344] = op;
            
            op->set_inputs( forward_result[343] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1345', 'op': 'aten::rsub', 'in': [344, 126, 0], 'output_id': 0, 'shape': [10, 400], 'out': [353], 'sorted_id': 345}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[345] = op;
            
            op->set_inputs( forward_result[344] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1346', 'op': 'aten::sub', 'in': [276, 263, 0], 'output_id': 0, 'shape': [10, 400], 'out': [347], 'sorted_id': 346}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[346] = op;
            
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[263] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1347', 'op': 'aten::pow', 'in': [346, 25], 'output_id': 0, 'shape': [10, 400], 'out': [352], 'sorted_id': 347}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[347] = op;
            
            op->set_inputs( forward_result[346] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1348', 'op': 'aten::exp', 'in': [264], 'output_id': 0, 'shape': [10, 400], 'out': [349], 'sorted_id': 348}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[348] = op;
            
            op->set_inputs( forward_result[264] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1349', 'op': 'aten::log1p', 'in': [348], 'output_id': 0, 'shape': [10, 400], 'out': [350], 'sorted_id': 349}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[349] = op;
            
            op->set_inputs( forward_result[348] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1350', 'op': 'aten::pow', 'in': [349, 25], 'output_id': 0, 'shape': [10, 400], 'out': [351], 'sorted_id': 350}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[350] = op;
            
            op->set_inputs( forward_result[349] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1351', 'op': 'aten::mul', 'in': [350, 72], 'output_id': 0, 'shape': [10, 400], 'out': [352], 'sorted_id': 351}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[351] = op;
            
            op->set_inputs( forward_result[350] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1352', 'op': 'aten::div', 'in': [347, 351], 'output_id': 0, 'shape': [10, 400], 'out': [353], 'sorted_id': 352}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[352] = op;
            
            op->set_inputs( forward_result[347] );
            op->set_inputs( forward_result[351] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1353', 'op': 'aten::sub', 'in': [345, 352, 0], 'output_id': 0, 'shape': [10, 400], 'out': [354], 'sorted_id': 353}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[353] = op;
            
            op->set_inputs( forward_result[345] );
            op->set_inputs( forward_result[352] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1354', 'op': 'aten::sum', 'in': [353, 3], 'output_id': 0, 'shape': [], 'out': [368], 'sorted_id': 354}
        {
            SumOp* op = new SumOp();
            forward_result[354] = op;
            
            op->set_inputs( forward_result[353] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1355', 'op': 'aten::exp', 'in': [278], 'output_id': 0, 'shape': [10], 'out': [356], 'sorted_id': 355}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[355] = op;
            
            op->set_inputs( forward_result[278] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1356', 'op': 'aten::log1p', 'in': [355], 'output_id': 0, 'shape': [10], 'out': [357], 'sorted_id': 356}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[356] = op;
            
            op->set_inputs( forward_result[355] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1357', 'op': 'aten::log', 'in': [356], 'output_id': 0, 'shape': [10], 'out': [358], 'sorted_id': 357}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[357] = op;
            
            op->set_inputs( forward_result[356] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1358', 'op': 'aten::rsub', 'in': [357, 126, 0], 'output_id': 0, 'shape': [10], 'out': [366], 'sorted_id': 358}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[358] = op;
            
            op->set_inputs( forward_result[357] );
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1359', 'op': 'aten::sub', 'in': [289, 277, 0], 'output_id': 0, 'shape': [10], 'out': [360], 'sorted_id': 359}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[359] = op;
            
            op->set_inputs( forward_result[289] );
            op->set_inputs( forward_result[277] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1360', 'op': 'aten::pow', 'in': [359, 25], 'output_id': 0, 'shape': [10], 'out': [365], 'sorted_id': 360}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[360] = op;
            
            op->set_inputs( forward_result[359] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1361', 'op': 'aten::exp', 'in': [278], 'output_id': 0, 'shape': [10], 'out': [362], 'sorted_id': 361}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[361] = op;
            
            op->set_inputs( forward_result[278] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1362', 'op': 'aten::log1p', 'in': [361], 'output_id': 0, 'shape': [10], 'out': [363], 'sorted_id': 362}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[362] = op;
            
            op->set_inputs( forward_result[361] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1363', 'op': 'aten::pow', 'in': [362, 25], 'output_id': 0, 'shape': [10], 'out': [364], 'sorted_id': 363}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[363] = op;
            
            op->set_inputs( forward_result[362] );
            op->set_inputs( forward_result[25] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1364', 'op': 'aten::mul', 'in': [363, 72], 'output_id': 0, 'shape': [10], 'out': [365], 'sorted_id': 364}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[364] = op;
            
            op->set_inputs( forward_result[363] );
            op->set_inputs( forward_result[72] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1365', 'op': 'aten::div', 'in': [360, 364], 'output_id': 0, 'shape': [10], 'out': [366], 'sorted_id': 365}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[365] = op;
            
            op->set_inputs( forward_result[360] );
            op->set_inputs( forward_result[364] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1366', 'op': 'aten::sub', 'in': [358, 365, 0], 'output_id': 0, 'shape': [10], 'out': [367], 'sorted_id': 366}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[366] = op;
            
            op->set_inputs( forward_result[358] );
            op->set_inputs( forward_result[365] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1367', 'op': 'aten::sum', 'in': [366, 3], 'output_id': 0, 'shape': [], 'out': [368], 'sorted_id': 367}
        {
            SumOp* op = new SumOp();
            forward_result[367] = op;
            
            op->set_inputs( forward_result[366] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/BayesianLinear[l3]/1368', 'op': 'aten::add', 'in': [354, 367, 0], 'output_id': 0, 'shape': [], 'out': [369], 'sorted_id': 368}
        {
            AddOp* op = new AddOp();
            forward_result[368] = op;
            
            op->set_inputs( forward_result[354] );
            op->set_inputs( forward_result[367] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/1370', 'op': 'prim::TupleConstruct', 'in': [290, 341, 368], 'output_id': 0, 'shape': [], 'out': [374, 377, 370], 'sorted_id': 369}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[369] = op;
            
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[341] );
            op->set_inputs( forward_result[368] );
        }
        
        // {'name': 'Net/1371', 'op': 'prim::TupleUnpack', 'in': [369], 'output_id': 0, 'shape': [4, 10], 'out': [371], 'sorted_id': 370}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[370] = op;
            
            op->set_inputs( forward_result[369] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/BayesianNetwork[net]/1374', 'op': 'aten::log_softmax', 'in': [370, 0, 3], 'output_id': 0, 'shape': [4, 10], 'out': [378], 'sorted_id': 371}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[371] = op;
            
            op->set_inputs( forward_result[370] );
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/1150', 'op': 'prim::TupleUnpack', 'in': [151], 'output_id': 1, 'shape': [], 'out': [378], 'sorted_id': 372}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[372] = op;
            
            op->set_inputs( forward_result[151] );
        }
        
        // {'name': 'Net/1261', 'op': 'prim::TupleUnpack', 'in': [260], 'output_id': 1, 'shape': [], 'out': [378], 'sorted_id': 373}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[373] = op;
            
            op->set_inputs( forward_result[260] );
        }
        
        // {'name': 'Net/1372', 'op': 'prim::TupleUnpack', 'in': [369], 'output_id': 1, 'shape': [], 'out': [378], 'sorted_id': 374}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[374] = op;
            
            op->set_inputs( forward_result[369] );
        }
        
        // {'name': 'Net/1151', 'op': 'prim::TupleUnpack', 'in': [151], 'output_id': 2, 'shape': [], 'out': [378], 'sorted_id': 375}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[375] = op;
            
            op->set_inputs( forward_result[151] );
        }
        
        // {'name': 'Net/1262', 'op': 'prim::TupleUnpack', 'in': [260], 'output_id': 2, 'shape': [], 'out': [378], 'sorted_id': 376}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[376] = op;
            
            op->set_inputs( forward_result[260] );
        }
        
        // {'name': 'Net/1373', 'op': 'prim::TupleUnpack', 'in': [369], 'output_id': 2, 'shape': [], 'out': [378], 'sorted_id': 377}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[377] = op;
            
            op->set_inputs( forward_result[369] );
        }
        
        // {'name': 'Net/1375', 'op': 'prim::TupleConstruct', 'in': [371, 372, 373, 374, 375, 376, 377], 'output_id': 0, 'shape': [], 'out': [385, 392, 382, 390, 383, 389, 379], 'sorted_id': 378}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[378] = op;
            
            op->set_inputs( forward_result[371] );
            op->set_inputs( forward_result[372] );
            op->set_inputs( forward_result[373] );
            op->set_inputs( forward_result[374] );
            op->set_inputs( forward_result[375] );
            op->set_inputs( forward_result[376] );
            op->set_inputs( forward_result[377] );
        }
        
        // {'name': 'Net/1376', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 0, 'shape': [4, 10], 'out': [380], 'sorted_id': 379}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[379] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1384', 'op': 'aten::copy_', 'in': [30, 379, 5], 'output_id': 0, 'shape': [4, 10], 'out': [], 'sorted_id': 380}
        {
            Tensor::shape_type shape = {4,10};
            Copy_Op* op = new Copy_Op();
            forward_result[380] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[379] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1387', 'op': 'aten::select', 'in': [12, 7, 7], 'output_id': 0, 'shape': [], 'out': [387], 'sorted_id': 381}
        {
            SelectOp* op = new SelectOp();
            forward_result[381] = op;
            
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/1377', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 1, 'shape': [], 'out': [384], 'sorted_id': 382}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[382] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/1378', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 2, 'shape': [], 'out': [384], 'sorted_id': 383}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[383] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1385', 'op': 'aten::add', 'in': [382, 383, 0], 'output_id': 0, 'shape': [], 'out': [386], 'sorted_id': 384}
        {
            AddOp* op = new AddOp();
            forward_result[384] = op;
            
            op->set_inputs( forward_result[382] );
            op->set_inputs( forward_result[383] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/1379', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 3, 'shape': [], 'out': [386], 'sorted_id': 385}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[385] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1386', 'op': 'aten::add', 'in': [384, 385, 0], 'output_id': 0, 'shape': [], 'out': [387], 'sorted_id': 386}
        {
            AddOp* op = new AddOp();
            forward_result[386] = op;
            
            op->set_inputs( forward_result[384] );
            op->set_inputs( forward_result[385] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1388', 'op': 'aten::copy_', 'in': [381, 386, 5], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 387}
        {
            Copy_Op* op = new Copy_Op();
            forward_result[387] = op;
            
            op->set_inputs( forward_result[381] );
            op->set_inputs( forward_result[386] );
            op->set_inputs( forward_result[5] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1391', 'op': 'aten::select', 'in': [8, 7, 7], 'output_id': 0, 'shape': [], 'out': [394], 'sorted_id': 388}
        {
            SelectOp* op = new SelectOp();
            forward_result[388] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Net/1380', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 4, 'shape': [], 'out': [391], 'sorted_id': 389}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[389] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/1381', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 5, 'shape': [], 'out': [391], 'sorted_id': 390}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[390] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1389', 'op': 'aten::add', 'in': [389, 390, 0], 'output_id': 0, 'shape': [], 'out': [393], 'sorted_id': 391}
        {
            AddOp* op = new AddOp();
            forward_result[391] = op;
            
            op->set_inputs( forward_result[389] );
            op->set_inputs( forward_result[390] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/1382', 'op': 'prim::TupleUnpack', 'in': [378], 'output_id': 6, 'shape': [], 'out': [393], 'sorted_id': 392}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 6 );
            forward_result[392] = op;
            
            op->set_inputs( forward_result[378] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1390', 'op': 'aten::add', 'in': [391, 392, 0], 'output_id': 0, 'shape': [], 'out': [394], 'sorted_id': 393}
        {
            AddOp* op = new AddOp();
            forward_result[393] = op;
            
            op->set_inputs( forward_result[391] );
            op->set_inputs( forward_result[392] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/BBBLoss[loss_func]/1392', 'op': 'aten::copy_', 'in': [388, 393, 5], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 394}
        {
            Copy_Op* op = new Copy_Op();
            forward_result[394] = op;
            
            op->set_inputs( forward_result[388] );
            op->set_inputs( forward_result[393] );
            op->set_inputs( forward_result[5] );
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
        vector<MCTNode*> forward_result(395);
    
        // input data
        Tensor::shape_type shape = {4,1,28,28};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 28 );
    #else
        do_train1( forward_result, input_var, 28 );
    #endif
        
        return 0;
    }
    