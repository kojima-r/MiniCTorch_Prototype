
    //
    //  bbb3
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
    extern Tensor  Constant4;
    extern Tensor  Constant5;
    extern Tensor  Constant6;
    extern Tensor  l2_weight_mu;
    extern Tensor  l2_weight_rho;
    extern Tensor  l2_bias_mu;
    extern Tensor  l2_bias_rho;
    extern Tensor  Constant7;
    extern Tensor  Constant8;
    extern Tensor  Constant9;
    extern Tensor  Constant10;
    extern Tensor  Constant11;
    extern Tensor  Constant12;
    extern Tensor  l3_weight_mu;
    extern Tensor  l3_weight_rho;
    extern Tensor  l3_bias_mu;
    extern Tensor  l3_bias_rho;
    extern Tensor  Constant13;
    extern Tensor  Constant14;
    extern Tensor  Constant15;
    extern Tensor  Constant16;
    extern Tensor  Constant17;
    extern Tensor  Constant18;
    extern Tensor  Constant19;
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [4, 1, 28, 28], 'out': [374, 733, 4], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {4,1,28,28};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Model/Net[net]/2693', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -1.0, 'out': [3, 732, 373], 'sorted_id': 1}
        {
            Tensor c = (fprec)-1.0;
            forward_result[1] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/2692', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 784.0, 'out': [3, 732, 373], 'sorted_id': 2}
        {
            Tensor c = (fprec)784.0;
            forward_result[2] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/2697', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [4], 'sorted_id': 3}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Model/Net[net]/input.1', 'op': 'aten::view', 'in': [0, 3], 'output_id': 0, 'shape': [4, 784], 'out': [42], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_mu/weight_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [26, 109], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[5] = new VariableTensor( l1_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_rho/weight_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [104, 111, 13, 7, 11], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[6] = new VariableTensor( l1_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2711', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2712', 'op': 'aten::log1p', 'in': [7], 'output_id': 0, 'shape': [400, 784], 'out': [25], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[8] = op;
            
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2678', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [16, 132], 'sorted_id': 9}
        {
            Tensor c = (fprec)0.0;
            forward_result[9] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2676', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [24, 1081, 1063, 156, 267, 641, 259, 163, 847, 275, 380, 388, 148, 11, 403, 867, 499, 618, 626, 861, 522, 396, 634, 745, 976, 738, 956, 752, 507, 140, 32, 854, 1099, 282, 970, 39, 515, 963, 1062, 758], 'sorted_id': 10}
        {
            Tensor c = (fprec)0.0;
            forward_result[10] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2703', 'op': 'aten::size', 'in': [6, 10], 'output_id': 0, 'shape': [], 'out': [18, 14], 'sorted_id': 11}
        {
            SizeOp* op = new SizeOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2677', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [77, 926, 324, 682, 830, 1013, 1035, 235, 770, 978, 579, 879, 1060, 1055, 109, 205, 536, 300, 655, 805, 576, 360, 594, 460, 837, 209, 1074, 701, 216, 190, 706, 586, 429, 430, 671, 719, 769, 787, 829, 248, 893, 524, 646, 675, 335, 914, 1069, 1042, 475, 991, 848, 824, 987, 381, 359, 713, 488, 1030, 957, 108, 798, 856, 240, 780, 1087, 628, 918, 1012, 556, 433, 643, 687, 731, 609, 1027, 509, 871, 998, 347, 339, 408, 1072, 599, 241, 882, 654, 938, 1016, 773, 457, 189, 390, 587, 999, 61, 527, 667, 907, 1077, 1090, 116, 269, 121, 417, 668, 939, 122, 296, 342, 223, 915, 13, 747, 193, 946, 903, 659, 220, 480, 70, 563, 728, 463, 890, 1057, 96, 600, 948, 1097, 1092, 100, 481, 346, 85, 141, 988, 1023, 1085, 150, 812, 217, 312, 421, 878, 227, 287, 284, 762, 56, 369, 328, 449, 309, 839, 795, 619, 1002, 549, 467, 695, 683, 1079, 367, 705, 228, 535, 204, 739, 698, 129, 564, 726, 784, 181, 809, 176, 372, 250, 323, 165, 500, 980, 168, 131, 44, 1005, 1024, 405, 490, 794, 582, 354, 41, 416, 437, 694, 965, 806, 568, 444, 904, 260, 781, 295, 89, 69, 540, 548, 97, 73, 607, 1048, 1067, 1047, 308, 103, 84, 456, 575, 54, 336, 1095, 552, 889, 197, 26, 177, 468, 445, 1034, 316, 896, 817, 760, 933, 718, 869, 816, 925, 921], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2704', 'op': 'aten::size', 'in': [6, 12], 'output_id': 0, 'shape': [], 'out': [18, 14], 'sorted_id': 13}
        {
            SizeOp* op = new SizeOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2705', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[14] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2679', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [24, 16, 961, 267, 641, 624, 163, 852, 143, 401, 865, 520, 388, 148, 161, 403, 867, 158, 626, 37, 754, 522, 636, 756, 959, 262, 972, 398, 741, 745, 976, 383, 505, 280, 517, 1064, 386, 974, 507, 854, 850, 863, 743, 146, 282, 621, 265, 639, 277, 19, 39, 34, 963, 502, 758], 'sorted_id': 15}
        {
            Tensor c = (fprec)0.0;
            forward_result[15] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2706', 'op': 'aten::expand', 'in': [9, 14, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2680', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [19, 132], 'sorted_id': 17}
        {
            Tensor c = (fprec)1.0;
            forward_result[17] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2707', 'op': 'prim::ListConstruct', 'in': [11, 13], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[13] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2708', 'op': 'aten::expand', 'in': [17, 18, 15], 'output_id': 0, 'shape': [400, 784], 'out': [21], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2681', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [267, 163, 700, 1060, 388, 403, 195, 920, 522, 1029, 1064, 368, 39, 1004, 758, 786, 731, 1056, 148, 402, 811, 162, 757, 745, 314, 476, 744, 117, 727, 435, 38, 24, 641, 1100, 75, 1043, 867, 521, 236, 976, 506, 895, 222, 825, 282, 934, 608, 975, 595, 372, 147, 581, 266, 387, 866, 130, 962, 626, 838, 21, 554, 853, 489, 1082, 102, 507, 854, 355, 673, 341, 281, 625, 714, 249, 947, 462, 963, 640], 'sorted_id': 20}
        {
            forward_result[20] = NULL;
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2709', 'op': 'aten::normal', 'in': [16, 19, 20], 'output_id': 0, 'shape': [400, 784], 'out': [24], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2682', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [745, 24, 282, 403, 867, 976, 267, 641, 626, 963, 163, 39, 522, 507, 854, 388, 148, 758], 'sorted_id': 22}
        {
            Tensor c = (fprec)6.0;
            forward_result[22] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2683', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [745, 24, 282, 403, 867, 976, 267, 641, 626, 963, 163, 39, 522, 507, 854, 388, 148, 758], 'sorted_id': 23}
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
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2713', 'op': 'aten::mul', 'in': [8, 24], 'output_id': 0, 'shape': [400, 784], 'out': [26], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.1', 'op': 'aten::add', 'in': [5, 25, 12], 'output_id': 0, 'shape': [400, 784], 'out': [42, 44, 109, 61], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[26] = op;
            
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_mu/bias_mu.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [122, 41], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[27] = new VariableTensor( l1_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_rho/bias_rho.1', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [32, 124, 29, 118], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[28] = new VariableTensor( l1_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2724', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [30], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2725', 'op': 'aten::log1p', 'in': [29], 'output_id': 0, 'shape': [400], 'out': [40], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2716', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [34, 132], 'sorted_id': 31}
        {
            Tensor c = (fprec)0.0;
            forward_result[31] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2715', 'op': 'aten::size', 'in': [28, 10], 'output_id': 0, 'shape': [], 'out': [33, 36], 'sorted_id': 32}
        {
            SizeOp* op = new SizeOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2717', 'op': 'prim::ListConstruct', 'in': [32], 'output_id': 0, 'shape': [], 'out': [34], 'sorted_id': 33}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2718', 'op': 'aten::expand', 'in': [31, 33, 15], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2719', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [37, 132], 'sorted_id': 35}
        {
            Tensor c = (fprec)1.0;
            forward_result[35] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2720', 'op': 'prim::ListConstruct', 'in': [32], 'output_id': 0, 'shape': [], 'out': [37], 'sorted_id': 36}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2721', 'op': 'aten::expand', 'in': [35, 36, 15], 'output_id': 0, 'shape': [400], 'out': [38], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2722', 'op': 'aten::normal', 'in': [34, 37, 20], 'output_id': 0, 'shape': [400], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.3', 'op': 'aten::to', 'in': [38, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [40], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[38] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2726', 'op': 'aten::mul', 'in': [30, 39], 'output_id': 0, 'shape': [400], 'out': [41], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.3', 'op': 'aten::add', 'in': [27, 40, 12], 'output_id': 0, 'shape': [400], 'out': [77, 89, 42, 122], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[41] = op;
            
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/input.3', 'op': 'aten::linear', 'in': [4, 26, 41], 'output_id': 0, 'shape': [4, 400], 'out': [132], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[41] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2686', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [44, 132], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {1};
            Constant1.reshape( shape );
            forward_result[43] = new VariableTensor( Constant1, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2730', 'op': 'aten::sub', 'in': [26, 43, 12], 'output_id': 0, 'shape': [400, 784], 'out': [46], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2685', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [604, 482, 46, 788, 198, 200, 872, 329, 818, 110, 927, 994, 422, 317, 245, 331, 49, 92, 885, 591, 1039, 776, 899, 440, 229, 1019, 663, 569, 472, 660, 943, 883, 304, 450, 232, 940, 288, 1017, 544, 80, 801, 821, 983, 763, 834, 361, 1049, 242, 676, 528, 723, 469, 210, 62, 1052, 123, 291, 425, 720, 412, 707, 588, 438, 601, 78, 1006, 690, 678, 541, 185, 409, 930, 790, 212, 647, 799, 874, 90, 1036, 897, 910, 364, 65, 348, 908, 319, 1008, 182, 831, 172, 169, 126, 992, 452, 485, 571, 301, 765, 981, 710, 531, 688, 113, 559, 351, 650, 774, 557], 'sorted_id': 45}
        {
            Tensor c = (fprec)2.0;
            forward_result[45] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2731', 'op': 'aten::pow', 'in': [44, 45], 'output_id': 0, 'shape': [400, 784], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2732', 'op': 'aten::neg', 'in': [46], 'output_id': 0, 'shape': [400, 784], 'out': [52], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2684', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [53, 83, 80, 49, 132], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {1};
            Constant2.reshape( shape );
            forward_result[48] = new VariableTensor( Constant2, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.1', 'op': 'aten::pow', 'in': [48, 45], 'output_id': 0, 'shape': [1], 'out': [51], 'sorted_id': 49}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2687', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [66, 691, 995, 572, 213, 984, 664, 51, 233, 545, 93, 365, 724, 711, 605, 791, 875, 835, 486, 944, 1053, 802, 473, 532, 332, 246, 453, 886, 320, 292, 114, 822, 931, 1040, 305, 900, 127, 352, 651, 186, 173, 592, 560, 766, 201, 441, 1009, 679, 1020, 911, 426, 413, 81, 777], 'sorted_id': 50}
        {
            Tensor c = (fprec)2.0;
            forward_result[50] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2733', 'op': 'aten::mul', 'in': [49, 50], 'output_id': 0, 'shape': [1], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[49] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2734', 'op': 'aten::div', 'in': [47, 51], 'output_id': 0, 'shape': [400, 784], 'out': [54], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.1', 'op': 'aten::log', 'in': [48], 'output_id': 0, 'shape': [1], 'out': [54], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[53] = op;
            
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2735', 'op': 'aten::sub', 'in': [52, 53, 12], 'output_id': 0, 'shape': [400, 784], 'out': [56], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[54] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2688', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.9189, 'out': [324, 890, 70, 1013, 1024, 770, 879, 85, 806, 988, 904, 205, 536, 217, 781, 655, 576, 97, 457, 56, 999, 190, 309, 336, 795, 430, 549, 695, 683, 417, 668, 177, 445, 296, 564, 915], 'sorted_id': 55}
        {
            Tensor c = (fprec)0.9189;
            forward_result[55] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2736', 'op': 'aten::sub', 'in': [54, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [57], 'sorted_id': 56}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[56] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.1', 'op': 'aten::exp', 'in': [56], 'output_id': 0, 'shape': [400, 784], 'out': [59], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[56] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2690', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [881, 87, 311, 459, 578, 783, 432, 657, 906, 338, 99, 670, 326, 990, 685, 797, 551, 219, 419, 538, 192, 207, 179, 566, 72, 697, 892, 59, 1026, 1015, 1001, 298, 772, 808, 447, 917], 'sorted_id': 58}
        {
            Tensor c = (fprec)0.5;
            forward_result[58] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2749', 'op': 'aten::mul', 'in': [57, 58], 'output_id': 0, 'shape': [400, 784], 'out': [73], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2740', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [132, 61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {1};
            Constant3.reshape( shape );
            forward_result[60] = new VariableTensor( Constant3, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2741', 'op': 'aten::sub', 'in': [26, 60, 12], 'output_id': 0, 'shape': [400, 784], 'out': [62], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2742', 'op': 'aten::pow', 'in': [61, 45], 'output_id': 0, 'shape': [400, 784], 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2743', 'op': 'aten::neg', 'in': [62], 'output_id': 0, 'shape': [400, 784], 'out': [67], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2689', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [68, 65, 92, 95, 132], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {1};
            Constant4.reshape( shape );
            forward_result[64] = new VariableTensor( Constant4, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.3', 'op': 'aten::pow', 'in': [64, 45], 'output_id': 0, 'shape': [1], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2744', 'op': 'aten::mul', 'in': [65, 50], 'output_id': 0, 'shape': [1], 'out': [67], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2745', 'op': 'aten::div', 'in': [63, 66], 'output_id': 0, 'shape': [400, 784], 'out': [69], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[66] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.3', 'op': 'aten::log', 'in': [64], 'output_id': 0, 'shape': [1], 'out': [69], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[68] = op;
            
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2746', 'op': 'aten::sub', 'in': [67, 68, 12], 'output_id': 0, 'shape': [400, 784], 'out': [70], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[67] );
            op->set_inputs( forward_result[68] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2747', 'op': 'aten::sub', 'in': [69, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [71], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[70] = op;
            
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.1', 'op': 'aten::exp', 'in': [70], 'output_id': 0, 'shape': [400, 784], 'out': [72], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[70] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2750', 'op': 'aten::mul', 'in': [71, 58], 'output_id': 0, 'shape': [400, 784], 'out': [73], 'sorted_id': 72}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2751', 'op': 'aten::add', 'in': [59, 72, 12], 'output_id': 0, 'shape': [400, 784], 'out': [74], 'sorted_id': 73}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[73] = op;
            
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2752', 'op': 'aten::log', 'in': [73], 'output_id': 0, 'shape': [400, 784], 'out': [75], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2753', 'op': 'aten::sum', 'in': [74, 20], 'output_id': 0, 'shape': [], 'out': [103], 'sorted_id': 75}
        {
            SumOp* op = new SumOp();
            forward_result[75] = op;
            
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2756', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [77, 132], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {1};
            Constant5.reshape( shape );
            forward_result[76] = new VariableTensor( Constant5, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2757', 'op': 'aten::sub', 'in': [41, 76, 12], 'output_id': 0, 'shape': [400], 'out': [78], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2758', 'op': 'aten::pow', 'in': [77, 45], 'output_id': 0, 'shape': [400], 'out': [79], 'sorted_id': 78}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[78] = op;
            
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2759', 'op': 'aten::neg', 'in': [78], 'output_id': 0, 'shape': [400], 'out': [82], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[78] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.5', 'op': 'aten::pow', 'in': [48, 45], 'output_id': 0, 'shape': [1], 'out': [81], 'sorted_id': 80}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[80] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2760', 'op': 'aten::mul', 'in': [80, 50], 'output_id': 0, 'shape': [1], 'out': [82], 'sorted_id': 81}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[80] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2761', 'op': 'aten::div', 'in': [79, 81], 'output_id': 0, 'shape': [400], 'out': [84], 'sorted_id': 82}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[82] = op;
            
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[81] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.5', 'op': 'aten::log', 'in': [48], 'output_id': 0, 'shape': [1], 'out': [84], 'sorted_id': 83}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[83] = op;
            
            op->set_inputs( forward_result[48] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2762', 'op': 'aten::sub', 'in': [82, 83, 12], 'output_id': 0, 'shape': [400], 'out': [85], 'sorted_id': 84}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[84] = op;
            
            op->set_inputs( forward_result[82] );
            op->set_inputs( forward_result[83] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2763', 'op': 'aten::sub', 'in': [84, 55, 12], 'output_id': 0, 'shape': [400], 'out': [86], 'sorted_id': 85}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[84] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.3', 'op': 'aten::exp', 'in': [85], 'output_id': 0, 'shape': [400], 'out': [87], 'sorted_id': 86}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[86] = op;
            
            op->set_inputs( forward_result[85] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2776', 'op': 'aten::mul', 'in': [86, 58], 'output_id': 0, 'shape': [400], 'out': [100], 'sorted_id': 87}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[86] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2767', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [89, 132], 'sorted_id': 88}
        {
            Tensor::shape_type shape = {1};
            Constant6.reshape( shape );
            forward_result[88] = new VariableTensor( Constant6, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2768', 'op': 'aten::sub', 'in': [41, 88, 12], 'output_id': 0, 'shape': [400], 'out': [90], 'sorted_id': 89}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[89] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[88] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2769', 'op': 'aten::pow', 'in': [89, 45], 'output_id': 0, 'shape': [400], 'out': [91], 'sorted_id': 90}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[90] = op;
            
            op->set_inputs( forward_result[89] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2770', 'op': 'aten::neg', 'in': [90], 'output_id': 0, 'shape': [400], 'out': [94], 'sorted_id': 91}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[91] = op;
            
            op->set_inputs( forward_result[90] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.7', 'op': 'aten::pow', 'in': [64, 45], 'output_id': 0, 'shape': [1], 'out': [93], 'sorted_id': 92}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[92] = op;
            
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2771', 'op': 'aten::mul', 'in': [92, 50], 'output_id': 0, 'shape': [1], 'out': [94], 'sorted_id': 93}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[93] = op;
            
            op->set_inputs( forward_result[92] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2772', 'op': 'aten::div', 'in': [91, 93], 'output_id': 0, 'shape': [400], 'out': [96], 'sorted_id': 94}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[94] = op;
            
            op->set_inputs( forward_result[91] );
            op->set_inputs( forward_result[93] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.7', 'op': 'aten::log', 'in': [64], 'output_id': 0, 'shape': [1], 'out': [96], 'sorted_id': 95}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[95] = op;
            
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2773', 'op': 'aten::sub', 'in': [94, 95, 12], 'output_id': 0, 'shape': [400], 'out': [97], 'sorted_id': 96}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[96] = op;
            
            op->set_inputs( forward_result[94] );
            op->set_inputs( forward_result[95] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2774', 'op': 'aten::sub', 'in': [96, 55, 12], 'output_id': 0, 'shape': [400], 'out': [98], 'sorted_id': 97}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[97] = op;
            
            op->set_inputs( forward_result[96] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.3', 'op': 'aten::exp', 'in': [97], 'output_id': 0, 'shape': [400], 'out': [99], 'sorted_id': 98}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[98] = op;
            
            op->set_inputs( forward_result[97] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2777', 'op': 'aten::mul', 'in': [98, 58], 'output_id': 0, 'shape': [400], 'out': [100], 'sorted_id': 99}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[99] = op;
            
            op->set_inputs( forward_result[98] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2778', 'op': 'aten::add', 'in': [87, 99, 12], 'output_id': 0, 'shape': [400], 'out': [101], 'sorted_id': 100}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[100] = op;
            
            op->set_inputs( forward_result[87] );
            op->set_inputs( forward_result[99] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2779', 'op': 'aten::log', 'in': [100], 'output_id': 0, 'shape': [400], 'out': [102], 'sorted_id': 101}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[101] = op;
            
            op->set_inputs( forward_result[100] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2780', 'op': 'aten::sum', 'in': [101, 20], 'output_id': 0, 'shape': [], 'out': [103], 'sorted_id': 102}
        {
            SumOp* op = new SumOp();
            forward_result[102] = op;
            
            op->set_inputs( forward_result[101] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2781', 'op': 'aten::add', 'in': [75, 102, 12], 'output_id': 0, 'shape': [], 'out': [132], 'sorted_id': 103}
        {
            AddOp* op = new AddOp();
            forward_result[103] = op;
            
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[102] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2782', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [105], 'sorted_id': 104}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[104] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2783', 'op': 'aten::log1p', 'in': [104], 'output_id': 0, 'shape': [400, 784], 'out': [106], 'sorted_id': 105}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[105] = op;
            
            op->set_inputs( forward_result[104] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2784', 'op': 'aten::log', 'in': [105], 'output_id': 0, 'shape': [400, 784], 'out': [108], 'sorted_id': 106}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[106] = op;
            
            op->set_inputs( forward_result[105] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2691', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -0.9189385332046727, 'out': [1047, 108, 1034, 240, 480, 586, 718, 599, 829, 467, 816, 925, 359, 121, 346, 227, 705, 938], 'sorted_id': 107}
        {
            Tensor c = (fprec)-0.9189385332046727;
            forward_result[107] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2785', 'op': 'aten::rsub', 'in': [106, 107, 12], 'output_id': 0, 'shape': [400, 784], 'out': [116], 'sorted_id': 108}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[108] = op;
            
            op->set_inputs( forward_result[106] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2786', 'op': 'aten::sub', 'in': [26, 5, 12], 'output_id': 0, 'shape': [400, 784], 'out': [110], 'sorted_id': 109}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[109] = op;
            
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2787', 'op': 'aten::pow', 'in': [109, 45], 'output_id': 0, 'shape': [400, 784], 'out': [115], 'sorted_id': 110}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[110] = op;
            
            op->set_inputs( forward_result[109] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2788', 'op': 'aten::exp', 'in': [6], 'output_id': 0, 'shape': [400, 784], 'out': [112], 'sorted_id': 111}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[111] = op;
            
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2789', 'op': 'aten::log1p', 'in': [111], 'output_id': 0, 'shape': [400, 784], 'out': [113], 'sorted_id': 112}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[112] = op;
            
            op->set_inputs( forward_result[111] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2790', 'op': 'aten::pow', 'in': [112, 45], 'output_id': 0, 'shape': [400, 784], 'out': [114], 'sorted_id': 113}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[113] = op;
            
            op->set_inputs( forward_result[112] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2791', 'op': 'aten::mul', 'in': [113, 50], 'output_id': 0, 'shape': [400, 784], 'out': [115], 'sorted_id': 114}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[114] = op;
            
            op->set_inputs( forward_result[113] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2792', 'op': 'aten::div', 'in': [110, 114], 'output_id': 0, 'shape': [400, 784], 'out': [116], 'sorted_id': 115}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[115] = op;
            
            op->set_inputs( forward_result[110] );
            op->set_inputs( forward_result[114] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2793', 'op': 'aten::sub', 'in': [108, 115, 12], 'output_id': 0, 'shape': [400, 784], 'out': [117], 'sorted_id': 116}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[116] = op;
            
            op->set_inputs( forward_result[108] );
            op->set_inputs( forward_result[115] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2794', 'op': 'aten::sum', 'in': [116, 20], 'output_id': 0, 'shape': [], 'out': [131], 'sorted_id': 117}
        {
            SumOp* op = new SumOp();
            forward_result[117] = op;
            
            op->set_inputs( forward_result[116] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2795', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [119], 'sorted_id': 118}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[118] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2796', 'op': 'aten::log1p', 'in': [118], 'output_id': 0, 'shape': [400], 'out': [120], 'sorted_id': 119}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[119] = op;
            
            op->set_inputs( forward_result[118] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2797', 'op': 'aten::log', 'in': [119], 'output_id': 0, 'shape': [400], 'out': [121], 'sorted_id': 120}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[120] = op;
            
            op->set_inputs( forward_result[119] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2798', 'op': 'aten::rsub', 'in': [120, 107, 12], 'output_id': 0, 'shape': [400], 'out': [129], 'sorted_id': 121}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[121] = op;
            
            op->set_inputs( forward_result[120] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2799', 'op': 'aten::sub', 'in': [41, 27, 12], 'output_id': 0, 'shape': [400], 'out': [123], 'sorted_id': 122}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[122] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2800', 'op': 'aten::pow', 'in': [122, 45], 'output_id': 0, 'shape': [400], 'out': [128], 'sorted_id': 123}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[123] = op;
            
            op->set_inputs( forward_result[122] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2801', 'op': 'aten::exp', 'in': [28], 'output_id': 0, 'shape': [400], 'out': [125], 'sorted_id': 124}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[124] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2802', 'op': 'aten::log1p', 'in': [124], 'output_id': 0, 'shape': [400], 'out': [126], 'sorted_id': 125}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[125] = op;
            
            op->set_inputs( forward_result[124] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2803', 'op': 'aten::pow', 'in': [125, 45], 'output_id': 0, 'shape': [400], 'out': [127], 'sorted_id': 126}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[126] = op;
            
            op->set_inputs( forward_result[125] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2804', 'op': 'aten::mul', 'in': [126, 50], 'output_id': 0, 'shape': [400], 'out': [128], 'sorted_id': 127}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[127] = op;
            
            op->set_inputs( forward_result[126] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2805', 'op': 'aten::div', 'in': [123, 127], 'output_id': 0, 'shape': [400], 'out': [129], 'sorted_id': 128}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[128] = op;
            
            op->set_inputs( forward_result[123] );
            op->set_inputs( forward_result[127] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2806', 'op': 'aten::sub', 'in': [121, 128, 12], 'output_id': 0, 'shape': [400], 'out': [130], 'sorted_id': 129}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[129] = op;
            
            op->set_inputs( forward_result[121] );
            op->set_inputs( forward_result[128] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2807', 'op': 'aten::sum', 'in': [129, 20], 'output_id': 0, 'shape': [], 'out': [131], 'sorted_id': 130}
        {
            SumOp* op = new SumOp();
            forward_result[130] = op;
            
            op->set_inputs( forward_result[129] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/2808', 'op': 'aten::add', 'in': [117, 130, 12], 'output_id': 0, 'shape': [], 'out': [132], 'sorted_id': 131}
        {
            AddOp* op = new AddOp();
            forward_result[131] = op;
            
            op->set_inputs( forward_result[117] );
            op->set_inputs( forward_result[130] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/2810', 'op': 'prim::TupleConstruct', 'in': [42, 103, 131, 9, 17, 31, 35, 48, 43, 64, 60, 76, 88], 'output_id': 0, 'shape': [], 'out': [379, 384, 395, 411, 436, 448, 1065, 399, 1083, 420, 133, 424, 407], 'sorted_id': 132}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[132] = op;
            
            op->set_inputs( forward_result[42] );
            op->set_inputs( forward_result[103] );
            op->set_inputs( forward_result[131] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[88] );
        }
        
        // {'name': 'Model/2811', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 0, 'shape': [4, 400], 'out': [134], 'sorted_id': 133}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[133] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/input.5', 'op': 'aten::relu', 'in': [133], 'output_id': 0, 'shape': [4, 400], 'out': [166], 'sorted_id': 134}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[134] = op;
            
            op->set_inputs( forward_result[133] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_mu/weight_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [150, 228], 'sorted_id': 135}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[135] = new VariableTensor( l2_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_rho/weight_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [230, 137, 224, 140, 141], 'sorted_id': 136}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[136] = new VariableTensor( l2_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2842', 'op': 'aten::exp', 'in': [136], 'output_id': 0, 'shape': [400, 400], 'out': [138], 'sorted_id': 137}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[137] = op;
            
            op->set_inputs( forward_result[136] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2843', 'op': 'aten::log1p', 'in': [137], 'output_id': 0, 'shape': [400, 400], 'out': [149], 'sorted_id': 138}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[138] = op;
            
            op->set_inputs( forward_result[137] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2829', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [251, 143], 'sorted_id': 139}
        {
            Tensor c = (fprec)0.0;
            forward_result[139] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2834', 'op': 'aten::size', 'in': [136, 10], 'output_id': 0, 'shape': [], 'out': [145, 142], 'sorted_id': 140}
        {
            SizeOp* op = new SizeOp();
            forward_result[140] = op;
            
            op->set_inputs( forward_result[136] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2835', 'op': 'aten::size', 'in': [136, 12], 'output_id': 0, 'shape': [], 'out': [145, 142], 'sorted_id': 141}
        {
            SizeOp* op = new SizeOp();
            forward_result[141] = op;
            
            op->set_inputs( forward_result[136] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2836', 'op': 'prim::ListConstruct', 'in': [140, 141], 'output_id': 0, 'shape': [], 'out': [143], 'sorted_id': 142}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[142] = op;
            
            op->set_inputs( forward_result[140] );
            op->set_inputs( forward_result[141] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2837', 'op': 'aten::expand', 'in': [139, 142, 15], 'output_id': 0, 'shape': [400, 400], 'out': [147], 'sorted_id': 143}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[143] = op;
            
            op->set_inputs( forward_result[139] );
            op->set_inputs( forward_result[142] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2828', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [146, 251], 'sorted_id': 144}
        {
            Tensor c = (fprec)1.0;
            forward_result[144] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2838', 'op': 'prim::ListConstruct', 'in': [140, 141], 'output_id': 0, 'shape': [], 'out': [146], 'sorted_id': 145}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[145] = op;
            
            op->set_inputs( forward_result[140] );
            op->set_inputs( forward_result[141] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2839', 'op': 'aten::expand', 'in': [144, 145, 15], 'output_id': 0, 'shape': [400, 400], 'out': [147], 'sorted_id': 146}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[146] = op;
            
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[145] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2840', 'op': 'aten::normal', 'in': [143, 146, 20], 'output_id': 0, 'shape': [400, 400], 'out': [148], 'sorted_id': 147}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[147] = op;
            
            op->set_inputs( forward_result[143] );
            op->set_inputs( forward_result[146] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.5', 'op': 'aten::to', 'in': [147, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 400], 'out': [149], 'sorted_id': 148}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[148] = op;
            
            op->set_inputs( forward_result[147] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2844', 'op': 'aten::mul', 'in': [138, 148], 'output_id': 0, 'shape': [400, 400], 'out': [150], 'sorted_id': 149}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[149] = op;
            
            op->set_inputs( forward_result[138] );
            op->set_inputs( forward_result[148] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.5', 'op': 'aten::add', 'in': [135, 149, 12], 'output_id': 0, 'shape': [400, 400], 'out': [168, 181, 166, 228], 'sorted_id': 150}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[150] = op;
            
            op->set_inputs( forward_result[135] );
            op->set_inputs( forward_result[149] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_mu/bias_mu.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [241, 165], 'sorted_id': 151}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[151] = new VariableTensor( l2_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_rho/bias_rho.3', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [237, 243, 153, 156], 'sorted_id': 152}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[152] = new VariableTensor( l2_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2855', 'op': 'aten::exp', 'in': [152], 'output_id': 0, 'shape': [400], 'out': [154], 'sorted_id': 153}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[153] = op;
            
            op->set_inputs( forward_result[152] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2856', 'op': 'aten::log1p', 'in': [153], 'output_id': 0, 'shape': [400], 'out': [164], 'sorted_id': 154}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[154] = op;
            
            op->set_inputs( forward_result[153] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2847', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [251, 158], 'sorted_id': 155}
        {
            Tensor c = (fprec)0.0;
            forward_result[155] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2846', 'op': 'aten::size', 'in': [152, 10], 'output_id': 0, 'shape': [], 'out': [157, 160], 'sorted_id': 156}
        {
            SizeOp* op = new SizeOp();
            forward_result[156] = op;
            
            op->set_inputs( forward_result[152] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2848', 'op': 'prim::ListConstruct', 'in': [156], 'output_id': 0, 'shape': [], 'out': [158], 'sorted_id': 157}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[157] = op;
            
            op->set_inputs( forward_result[156] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2849', 'op': 'aten::expand', 'in': [155, 157, 15], 'output_id': 0, 'shape': [400], 'out': [162], 'sorted_id': 158}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[158] = op;
            
            op->set_inputs( forward_result[155] );
            op->set_inputs( forward_result[157] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2850', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [251, 161], 'sorted_id': 159}
        {
            Tensor c = (fprec)1.0;
            forward_result[159] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2851', 'op': 'prim::ListConstruct', 'in': [156], 'output_id': 0, 'shape': [], 'out': [161], 'sorted_id': 160}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[160] = op;
            
            op->set_inputs( forward_result[156] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2852', 'op': 'aten::expand', 'in': [159, 160, 15], 'output_id': 0, 'shape': [400], 'out': [162], 'sorted_id': 161}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[161] = op;
            
            op->set_inputs( forward_result[159] );
            op->set_inputs( forward_result[160] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2853', 'op': 'aten::normal', 'in': [158, 161, 20], 'output_id': 0, 'shape': [400], 'out': [163], 'sorted_id': 162}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[162] = op;
            
            op->set_inputs( forward_result[158] );
            op->set_inputs( forward_result[161] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.7', 'op': 'aten::to', 'in': [162, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [164], 'sorted_id': 163}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[163] = op;
            
            op->set_inputs( forward_result[162] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2857', 'op': 'aten::mul', 'in': [154, 163], 'output_id': 0, 'shape': [400], 'out': [165], 'sorted_id': 164}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[164] = op;
            
            op->set_inputs( forward_result[154] );
            op->set_inputs( forward_result[163] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.7', 'op': 'aten::add', 'in': [151, 164, 12], 'output_id': 0, 'shape': [400], 'out': [241, 166, 197, 209], 'sorted_id': 165}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[165] = op;
            
            op->set_inputs( forward_result[151] );
            op->set_inputs( forward_result[164] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/input.7', 'op': 'aten::linear', 'in': [134, 150, 165], 'output_id': 0, 'shape': [4, 400], 'out': [251], 'sorted_id': 166}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[166] = op;
            
            op->set_inputs( forward_result[134] );
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[165] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2826', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [168, 251], 'sorted_id': 167}
        {
            Tensor::shape_type shape = {1};
            Constant7.reshape( shape );
            forward_result[167] = new VariableTensor( Constant7, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2861', 'op': 'aten::sub', 'in': [150, 167, 12], 'output_id': 0, 'shape': [400, 400], 'out': [169], 'sorted_id': 168}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[168] = op;
            
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2862', 'op': 'aten::pow', 'in': [168, 45], 'output_id': 0, 'shape': [400, 400], 'out': [170], 'sorted_id': 169}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[169] = op;
            
            op->set_inputs( forward_result[168] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2863', 'op': 'aten::neg', 'in': [169], 'output_id': 0, 'shape': [400, 400], 'out': [174], 'sorted_id': 170}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[170] = op;
            
            op->set_inputs( forward_result[169] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2827', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [200, 172, 175, 251, 203], 'sorted_id': 171}
        {
            Tensor::shape_type shape = {1};
            Constant8.reshape( shape );
            forward_result[171] = new VariableTensor( Constant8, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.9', 'op': 'aten::pow', 'in': [171, 45], 'output_id': 0, 'shape': [1], 'out': [173], 'sorted_id': 172}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[172] = op;
            
            op->set_inputs( forward_result[171] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2864', 'op': 'aten::mul', 'in': [172, 50], 'output_id': 0, 'shape': [1], 'out': [174], 'sorted_id': 173}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[173] = op;
            
            op->set_inputs( forward_result[172] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2865', 'op': 'aten::div', 'in': [170, 173], 'output_id': 0, 'shape': [400, 400], 'out': [176], 'sorted_id': 174}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[174] = op;
            
            op->set_inputs( forward_result[170] );
            op->set_inputs( forward_result[173] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.9', 'op': 'aten::log', 'in': [171], 'output_id': 0, 'shape': [1], 'out': [176], 'sorted_id': 175}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[175] = op;
            
            op->set_inputs( forward_result[171] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2866', 'op': 'aten::sub', 'in': [174, 175, 12], 'output_id': 0, 'shape': [400, 400], 'out': [177], 'sorted_id': 176}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[176] = op;
            
            op->set_inputs( forward_result[174] );
            op->set_inputs( forward_result[175] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2867', 'op': 'aten::sub', 'in': [176, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [178], 'sorted_id': 177}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[177] = op;
            
            op->set_inputs( forward_result[176] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.5', 'op': 'aten::exp', 'in': [177], 'output_id': 0, 'shape': [400, 400], 'out': [179], 'sorted_id': 178}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[178] = op;
            
            op->set_inputs( forward_result[177] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2880', 'op': 'aten::mul', 'in': [178, 58], 'output_id': 0, 'shape': [400, 400], 'out': [193], 'sorted_id': 179}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[179] = op;
            
            op->set_inputs( forward_result[178] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2871', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [251, 181], 'sorted_id': 180}
        {
            Tensor::shape_type shape = {1};
            Constant9.reshape( shape );
            forward_result[180] = new VariableTensor( Constant9, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2872', 'op': 'aten::sub', 'in': [150, 180, 12], 'output_id': 0, 'shape': [400, 400], 'out': [182], 'sorted_id': 181}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[181] = op;
            
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2873', 'op': 'aten::pow', 'in': [181, 45], 'output_id': 0, 'shape': [400, 400], 'out': [183], 'sorted_id': 182}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[182] = op;
            
            op->set_inputs( forward_result[181] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2874', 'op': 'aten::neg', 'in': [182], 'output_id': 0, 'shape': [400, 400], 'out': [187], 'sorted_id': 183}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[183] = op;
            
            op->set_inputs( forward_result[182] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2825', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [185, 188, 215, 212, 251], 'sorted_id': 184}
        {
            Tensor::shape_type shape = {1};
            Constant10.reshape( shape );
            forward_result[184] = new VariableTensor( Constant10, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.11', 'op': 'aten::pow', 'in': [184, 45], 'output_id': 0, 'shape': [1], 'out': [186], 'sorted_id': 185}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[185] = op;
            
            op->set_inputs( forward_result[184] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2875', 'op': 'aten::mul', 'in': [185, 50], 'output_id': 0, 'shape': [1], 'out': [187], 'sorted_id': 186}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[186] = op;
            
            op->set_inputs( forward_result[185] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2876', 'op': 'aten::div', 'in': [183, 186], 'output_id': 0, 'shape': [400, 400], 'out': [189], 'sorted_id': 187}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[187] = op;
            
            op->set_inputs( forward_result[183] );
            op->set_inputs( forward_result[186] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.11', 'op': 'aten::log', 'in': [184], 'output_id': 0, 'shape': [1], 'out': [189], 'sorted_id': 188}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[188] = op;
            
            op->set_inputs( forward_result[184] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2877', 'op': 'aten::sub', 'in': [187, 188, 12], 'output_id': 0, 'shape': [400, 400], 'out': [190], 'sorted_id': 189}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[189] = op;
            
            op->set_inputs( forward_result[187] );
            op->set_inputs( forward_result[188] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2878', 'op': 'aten::sub', 'in': [189, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [191], 'sorted_id': 190}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[190] = op;
            
            op->set_inputs( forward_result[189] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.5', 'op': 'aten::exp', 'in': [190], 'output_id': 0, 'shape': [400, 400], 'out': [192], 'sorted_id': 191}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[191] = op;
            
            op->set_inputs( forward_result[190] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2881', 'op': 'aten::mul', 'in': [191, 58], 'output_id': 0, 'shape': [400, 400], 'out': [193], 'sorted_id': 192}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[192] = op;
            
            op->set_inputs( forward_result[191] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2882', 'op': 'aten::add', 'in': [179, 192, 12], 'output_id': 0, 'shape': [400, 400], 'out': [194], 'sorted_id': 193}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[193] = op;
            
            op->set_inputs( forward_result[179] );
            op->set_inputs( forward_result[192] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2883', 'op': 'aten::log', 'in': [193], 'output_id': 0, 'shape': [400, 400], 'out': [195], 'sorted_id': 194}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[194] = op;
            
            op->set_inputs( forward_result[193] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2884', 'op': 'aten::sum', 'in': [194, 20], 'output_id': 0, 'shape': [], 'out': [223], 'sorted_id': 195}
        {
            SumOp* op = new SumOp();
            forward_result[195] = op;
            
            op->set_inputs( forward_result[194] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2887', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [251, 197], 'sorted_id': 196}
        {
            Tensor::shape_type shape = {1};
            Constant11.reshape( shape );
            forward_result[196] = new VariableTensor( Constant11, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2888', 'op': 'aten::sub', 'in': [165, 196, 12], 'output_id': 0, 'shape': [400], 'out': [198], 'sorted_id': 197}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[197] = op;
            
            op->set_inputs( forward_result[165] );
            op->set_inputs( forward_result[196] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2889', 'op': 'aten::pow', 'in': [197, 45], 'output_id': 0, 'shape': [400], 'out': [199], 'sorted_id': 198}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[198] = op;
            
            op->set_inputs( forward_result[197] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2890', 'op': 'aten::neg', 'in': [198], 'output_id': 0, 'shape': [400], 'out': [202], 'sorted_id': 199}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[199] = op;
            
            op->set_inputs( forward_result[198] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.13', 'op': 'aten::pow', 'in': [171, 45], 'output_id': 0, 'shape': [1], 'out': [201], 'sorted_id': 200}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[200] = op;
            
            op->set_inputs( forward_result[171] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2891', 'op': 'aten::mul', 'in': [200, 50], 'output_id': 0, 'shape': [1], 'out': [202], 'sorted_id': 201}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[201] = op;
            
            op->set_inputs( forward_result[200] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2892', 'op': 'aten::div', 'in': [199, 201], 'output_id': 0, 'shape': [400], 'out': [204], 'sorted_id': 202}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[202] = op;
            
            op->set_inputs( forward_result[199] );
            op->set_inputs( forward_result[201] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.13', 'op': 'aten::log', 'in': [171], 'output_id': 0, 'shape': [1], 'out': [204], 'sorted_id': 203}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[203] = op;
            
            op->set_inputs( forward_result[171] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2893', 'op': 'aten::sub', 'in': [202, 203, 12], 'output_id': 0, 'shape': [400], 'out': [205], 'sorted_id': 204}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[204] = op;
            
            op->set_inputs( forward_result[202] );
            op->set_inputs( forward_result[203] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2894', 'op': 'aten::sub', 'in': [204, 55, 12], 'output_id': 0, 'shape': [400], 'out': [206], 'sorted_id': 205}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[205] = op;
            
            op->set_inputs( forward_result[204] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.7', 'op': 'aten::exp', 'in': [205], 'output_id': 0, 'shape': [400], 'out': [207], 'sorted_id': 206}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[206] = op;
            
            op->set_inputs( forward_result[205] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2907', 'op': 'aten::mul', 'in': [206, 58], 'output_id': 0, 'shape': [400], 'out': [220], 'sorted_id': 207}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[207] = op;
            
            op->set_inputs( forward_result[206] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2898', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [251, 209], 'sorted_id': 208}
        {
            Tensor::shape_type shape = {1};
            Constant12.reshape( shape );
            forward_result[208] = new VariableTensor( Constant12, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2899', 'op': 'aten::sub', 'in': [165, 208, 12], 'output_id': 0, 'shape': [400], 'out': [210], 'sorted_id': 209}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[209] = op;
            
            op->set_inputs( forward_result[165] );
            op->set_inputs( forward_result[208] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2900', 'op': 'aten::pow', 'in': [209, 45], 'output_id': 0, 'shape': [400], 'out': [211], 'sorted_id': 210}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[210] = op;
            
            op->set_inputs( forward_result[209] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2901', 'op': 'aten::neg', 'in': [210], 'output_id': 0, 'shape': [400], 'out': [214], 'sorted_id': 211}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[211] = op;
            
            op->set_inputs( forward_result[210] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.15', 'op': 'aten::pow', 'in': [184, 45], 'output_id': 0, 'shape': [1], 'out': [213], 'sorted_id': 212}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[212] = op;
            
            op->set_inputs( forward_result[184] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2902', 'op': 'aten::mul', 'in': [212, 50], 'output_id': 0, 'shape': [1], 'out': [214], 'sorted_id': 213}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[213] = op;
            
            op->set_inputs( forward_result[212] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2903', 'op': 'aten::div', 'in': [211, 213], 'output_id': 0, 'shape': [400], 'out': [216], 'sorted_id': 214}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[214] = op;
            
            op->set_inputs( forward_result[211] );
            op->set_inputs( forward_result[213] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.15', 'op': 'aten::log', 'in': [184], 'output_id': 0, 'shape': [1], 'out': [216], 'sorted_id': 215}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[215] = op;
            
            op->set_inputs( forward_result[184] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2904', 'op': 'aten::sub', 'in': [214, 215, 12], 'output_id': 0, 'shape': [400], 'out': [217], 'sorted_id': 216}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[216] = op;
            
            op->set_inputs( forward_result[214] );
            op->set_inputs( forward_result[215] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2905', 'op': 'aten::sub', 'in': [216, 55, 12], 'output_id': 0, 'shape': [400], 'out': [218], 'sorted_id': 217}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[217] = op;
            
            op->set_inputs( forward_result[216] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.7', 'op': 'aten::exp', 'in': [217], 'output_id': 0, 'shape': [400], 'out': [219], 'sorted_id': 218}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[218] = op;
            
            op->set_inputs( forward_result[217] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2908', 'op': 'aten::mul', 'in': [218, 58], 'output_id': 0, 'shape': [400], 'out': [220], 'sorted_id': 219}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[219] = op;
            
            op->set_inputs( forward_result[218] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2909', 'op': 'aten::add', 'in': [207, 219, 12], 'output_id': 0, 'shape': [400], 'out': [221], 'sorted_id': 220}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[220] = op;
            
            op->set_inputs( forward_result[207] );
            op->set_inputs( forward_result[219] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2910', 'op': 'aten::log', 'in': [220], 'output_id': 0, 'shape': [400], 'out': [222], 'sorted_id': 221}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[221] = op;
            
            op->set_inputs( forward_result[220] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2911', 'op': 'aten::sum', 'in': [221, 20], 'output_id': 0, 'shape': [], 'out': [223], 'sorted_id': 222}
        {
            SumOp* op = new SumOp();
            forward_result[222] = op;
            
            op->set_inputs( forward_result[221] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2912', 'op': 'aten::add', 'in': [195, 222, 12], 'output_id': 0, 'shape': [], 'out': [251], 'sorted_id': 223}
        {
            AddOp* op = new AddOp();
            forward_result[223] = op;
            
            op->set_inputs( forward_result[195] );
            op->set_inputs( forward_result[222] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2913', 'op': 'aten::exp', 'in': [136], 'output_id': 0, 'shape': [400, 400], 'out': [225], 'sorted_id': 224}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[224] = op;
            
            op->set_inputs( forward_result[136] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2914', 'op': 'aten::log1p', 'in': [224], 'output_id': 0, 'shape': [400, 400], 'out': [226], 'sorted_id': 225}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[225] = op;
            
            op->set_inputs( forward_result[224] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2915', 'op': 'aten::log', 'in': [225], 'output_id': 0, 'shape': [400, 400], 'out': [227], 'sorted_id': 226}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[226] = op;
            
            op->set_inputs( forward_result[225] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2916', 'op': 'aten::rsub', 'in': [226, 107, 12], 'output_id': 0, 'shape': [400, 400], 'out': [235], 'sorted_id': 227}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[227] = op;
            
            op->set_inputs( forward_result[226] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2917', 'op': 'aten::sub', 'in': [150, 135, 12], 'output_id': 0, 'shape': [400, 400], 'out': [229], 'sorted_id': 228}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[228] = op;
            
            op->set_inputs( forward_result[150] );
            op->set_inputs( forward_result[135] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2918', 'op': 'aten::pow', 'in': [228, 45], 'output_id': 0, 'shape': [400, 400], 'out': [234], 'sorted_id': 229}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[229] = op;
            
            op->set_inputs( forward_result[228] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2919', 'op': 'aten::exp', 'in': [136], 'output_id': 0, 'shape': [400, 400], 'out': [231], 'sorted_id': 230}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[230] = op;
            
            op->set_inputs( forward_result[136] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2920', 'op': 'aten::log1p', 'in': [230], 'output_id': 0, 'shape': [400, 400], 'out': [232], 'sorted_id': 231}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[231] = op;
            
            op->set_inputs( forward_result[230] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2921', 'op': 'aten::pow', 'in': [231, 45], 'output_id': 0, 'shape': [400, 400], 'out': [233], 'sorted_id': 232}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[232] = op;
            
            op->set_inputs( forward_result[231] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2922', 'op': 'aten::mul', 'in': [232, 50], 'output_id': 0, 'shape': [400, 400], 'out': [234], 'sorted_id': 233}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[233] = op;
            
            op->set_inputs( forward_result[232] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2923', 'op': 'aten::div', 'in': [229, 233], 'output_id': 0, 'shape': [400, 400], 'out': [235], 'sorted_id': 234}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[234] = op;
            
            op->set_inputs( forward_result[229] );
            op->set_inputs( forward_result[233] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2924', 'op': 'aten::sub', 'in': [227, 234, 12], 'output_id': 0, 'shape': [400, 400], 'out': [236], 'sorted_id': 235}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[235] = op;
            
            op->set_inputs( forward_result[227] );
            op->set_inputs( forward_result[234] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2925', 'op': 'aten::sum', 'in': [235, 20], 'output_id': 0, 'shape': [], 'out': [250], 'sorted_id': 236}
        {
            SumOp* op = new SumOp();
            forward_result[236] = op;
            
            op->set_inputs( forward_result[235] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2926', 'op': 'aten::exp', 'in': [152], 'output_id': 0, 'shape': [400], 'out': [238], 'sorted_id': 237}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[237] = op;
            
            op->set_inputs( forward_result[152] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2927', 'op': 'aten::log1p', 'in': [237], 'output_id': 0, 'shape': [400], 'out': [239], 'sorted_id': 238}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[238] = op;
            
            op->set_inputs( forward_result[237] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2928', 'op': 'aten::log', 'in': [238], 'output_id': 0, 'shape': [400], 'out': [240], 'sorted_id': 239}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[239] = op;
            
            op->set_inputs( forward_result[238] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2929', 'op': 'aten::rsub', 'in': [239, 107, 12], 'output_id': 0, 'shape': [400], 'out': [248], 'sorted_id': 240}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[240] = op;
            
            op->set_inputs( forward_result[239] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2930', 'op': 'aten::sub', 'in': [165, 151, 12], 'output_id': 0, 'shape': [400], 'out': [242], 'sorted_id': 241}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[241] = op;
            
            op->set_inputs( forward_result[165] );
            op->set_inputs( forward_result[151] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2931', 'op': 'aten::pow', 'in': [241, 45], 'output_id': 0, 'shape': [400], 'out': [247], 'sorted_id': 242}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[242] = op;
            
            op->set_inputs( forward_result[241] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2932', 'op': 'aten::exp', 'in': [152], 'output_id': 0, 'shape': [400], 'out': [244], 'sorted_id': 243}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[243] = op;
            
            op->set_inputs( forward_result[152] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2933', 'op': 'aten::log1p', 'in': [243], 'output_id': 0, 'shape': [400], 'out': [245], 'sorted_id': 244}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[244] = op;
            
            op->set_inputs( forward_result[243] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2934', 'op': 'aten::pow', 'in': [244, 45], 'output_id': 0, 'shape': [400], 'out': [246], 'sorted_id': 245}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[245] = op;
            
            op->set_inputs( forward_result[244] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2935', 'op': 'aten::mul', 'in': [245, 50], 'output_id': 0, 'shape': [400], 'out': [247], 'sorted_id': 246}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[246] = op;
            
            op->set_inputs( forward_result[245] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2936', 'op': 'aten::div', 'in': [242, 246], 'output_id': 0, 'shape': [400], 'out': [248], 'sorted_id': 247}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[247] = op;
            
            op->set_inputs( forward_result[242] );
            op->set_inputs( forward_result[246] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2937', 'op': 'aten::sub', 'in': [240, 247, 12], 'output_id': 0, 'shape': [400], 'out': [249], 'sorted_id': 248}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[248] = op;
            
            op->set_inputs( forward_result[240] );
            op->set_inputs( forward_result[247] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2938', 'op': 'aten::sum', 'in': [248, 20], 'output_id': 0, 'shape': [], 'out': [250], 'sorted_id': 249}
        {
            SumOp* op = new SumOp();
            forward_result[249] = op;
            
            op->set_inputs( forward_result[248] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/2939', 'op': 'aten::add', 'in': [236, 249, 12], 'output_id': 0, 'shape': [], 'out': [251], 'sorted_id': 250}
        {
            AddOp* op = new AddOp();
            forward_result[250] = op;
            
            op->set_inputs( forward_result[236] );
            op->set_inputs( forward_result[249] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/2941', 'op': 'prim::TupleConstruct', 'in': [166, 223, 250, 139, 144, 155, 159, 171, 167, 184, 180, 196, 208], 'output_id': 0, 'shape': [], 'out': [252, 567, 526, 503, 514, 543, 518, 539, 498, 555, 1066, 530, 1084], 'sorted_id': 251}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[251] = op;
            
            op->set_inputs( forward_result[166] );
            op->set_inputs( forward_result[223] );
            op->set_inputs( forward_result[250] );
            op->set_inputs( forward_result[139] );
            op->set_inputs( forward_result[144] );
            op->set_inputs( forward_result[155] );
            op->set_inputs( forward_result[159] );
            op->set_inputs( forward_result[171] );
            op->set_inputs( forward_result[167] );
            op->set_inputs( forward_result[184] );
            op->set_inputs( forward_result[180] );
            op->set_inputs( forward_result[196] );
            op->set_inputs( forward_result[208] );
        }
        
        // {'name': 'Model/2942', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 0, 'shape': [4, 400], 'out': [253], 'sorted_id': 252}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[252] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/input.9', 'op': 'aten::relu', 'in': [252], 'output_id': 0, 'shape': [4, 400], 'out': [285], 'sorted_id': 253}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[253] = op;
            
            op->set_inputs( forward_result[252] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_mu/weight_mu.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [269, 347], 'sorted_id': 254}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[254] = new VariableTensor( l3_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_rho/weight_rho.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [256, 343, 349, 260, 259], 'sorted_id': 255}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[255] = new VariableTensor( l3_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2973', 'op': 'aten::exp', 'in': [255], 'output_id': 0, 'shape': [10, 400], 'out': [257], 'sorted_id': 256}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[256] = op;
            
            op->set_inputs( forward_result[255] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2974', 'op': 'aten::log1p', 'in': [256], 'output_id': 0, 'shape': [10, 400], 'out': [268], 'sorted_id': 257}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[257] = op;
            
            op->set_inputs( forward_result[256] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2960', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [370, 262], 'sorted_id': 258}
        {
            Tensor c = (fprec)0.0;
            forward_result[258] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2965', 'op': 'aten::size', 'in': [255, 10], 'output_id': 0, 'shape': [], 'out': [264, 261], 'sorted_id': 259}
        {
            SizeOp* op = new SizeOp();
            forward_result[259] = op;
            
            op->set_inputs( forward_result[255] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2966', 'op': 'aten::size', 'in': [255, 12], 'output_id': 0, 'shape': [], 'out': [264, 261], 'sorted_id': 260}
        {
            SizeOp* op = new SizeOp();
            forward_result[260] = op;
            
            op->set_inputs( forward_result[255] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2967', 'op': 'prim::ListConstruct', 'in': [259, 260], 'output_id': 0, 'shape': [], 'out': [262], 'sorted_id': 261}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[261] = op;
            
            op->set_inputs( forward_result[259] );
            op->set_inputs( forward_result[260] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2968', 'op': 'aten::expand', 'in': [258, 261, 15], 'output_id': 0, 'shape': [10, 400], 'out': [266], 'sorted_id': 262}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[262] = op;
            
            op->set_inputs( forward_result[258] );
            op->set_inputs( forward_result[261] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2959', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [370, 265], 'sorted_id': 263}
        {
            Tensor c = (fprec)1.0;
            forward_result[263] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2969', 'op': 'prim::ListConstruct', 'in': [259, 260], 'output_id': 0, 'shape': [], 'out': [265], 'sorted_id': 264}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[264] = op;
            
            op->set_inputs( forward_result[259] );
            op->set_inputs( forward_result[260] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2970', 'op': 'aten::expand', 'in': [263, 264, 15], 'output_id': 0, 'shape': [10, 400], 'out': [266], 'sorted_id': 265}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[265] = op;
            
            op->set_inputs( forward_result[263] );
            op->set_inputs( forward_result[264] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2971', 'op': 'aten::normal', 'in': [262, 265, 20], 'output_id': 0, 'shape': [10, 400], 'out': [267], 'sorted_id': 266}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[266] = op;
            
            op->set_inputs( forward_result[262] );
            op->set_inputs( forward_result[265] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.9', 'op': 'aten::to', 'in': [266, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10, 400], 'out': [268], 'sorted_id': 267}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[267] = op;
            
            op->set_inputs( forward_result[266] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2975', 'op': 'aten::mul', 'in': [257, 267], 'output_id': 0, 'shape': [10, 400], 'out': [269], 'sorted_id': 268}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[268] = op;
            
            op->set_inputs( forward_result[257] );
            op->set_inputs( forward_result[267] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.9', 'op': 'aten::add', 'in': [254, 268, 12], 'output_id': 0, 'shape': [10, 400], 'out': [285, 287, 300, 347], 'sorted_id': 269}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[269] = op;
            
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[268] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_mu/bias_mu.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [284, 360], 'sorted_id': 270}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[270] = new VariableTensor( l3_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_rho/bias_rho.5', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [272, 356, 362, 275], 'sorted_id': 271}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[271] = new VariableTensor( l3_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2986', 'op': 'aten::exp', 'in': [271], 'output_id': 0, 'shape': [10], 'out': [273], 'sorted_id': 272}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[272] = op;
            
            op->set_inputs( forward_result[271] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2987', 'op': 'aten::log1p', 'in': [272], 'output_id': 0, 'shape': [10], 'out': [283], 'sorted_id': 273}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[273] = op;
            
            op->set_inputs( forward_result[272] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2978', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [370, 277], 'sorted_id': 274}
        {
            Tensor c = (fprec)0.0;
            forward_result[274] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2977', 'op': 'aten::size', 'in': [271, 10], 'output_id': 0, 'shape': [], 'out': [279, 276], 'sorted_id': 275}
        {
            SizeOp* op = new SizeOp();
            forward_result[275] = op;
            
            op->set_inputs( forward_result[271] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2979', 'op': 'prim::ListConstruct', 'in': [275], 'output_id': 0, 'shape': [], 'out': [277], 'sorted_id': 276}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[276] = op;
            
            op->set_inputs( forward_result[275] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2980', 'op': 'aten::expand', 'in': [274, 276, 15], 'output_id': 0, 'shape': [10], 'out': [281], 'sorted_id': 277}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[277] = op;
            
            op->set_inputs( forward_result[274] );
            op->set_inputs( forward_result[276] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2981', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [370, 280], 'sorted_id': 278}
        {
            Tensor c = (fprec)1.0;
            forward_result[278] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2982', 'op': 'prim::ListConstruct', 'in': [275], 'output_id': 0, 'shape': [], 'out': [280], 'sorted_id': 279}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[279] = op;
            
            op->set_inputs( forward_result[275] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2983', 'op': 'aten::expand', 'in': [278, 279, 15], 'output_id': 0, 'shape': [10], 'out': [281], 'sorted_id': 280}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[280] = op;
            
            op->set_inputs( forward_result[278] );
            op->set_inputs( forward_result[279] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2984', 'op': 'aten::normal', 'in': [277, 280, 20], 'output_id': 0, 'shape': [10], 'out': [282], 'sorted_id': 281}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[281] = op;
            
            op->set_inputs( forward_result[277] );
            op->set_inputs( forward_result[280] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.11', 'op': 'aten::to', 'in': [281, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10], 'out': [283], 'sorted_id': 282}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[282] = op;
            
            op->set_inputs( forward_result[281] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2988', 'op': 'aten::mul', 'in': [273, 282], 'output_id': 0, 'shape': [10], 'out': [284], 'sorted_id': 283}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[283] = op;
            
            op->set_inputs( forward_result[273] );
            op->set_inputs( forward_result[282] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.11', 'op': 'aten::add', 'in': [270, 283, 12], 'output_id': 0, 'shape': [10], 'out': [285, 328, 316, 360], 'sorted_id': 284}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[284] = op;
            
            op->set_inputs( forward_result[270] );
            op->set_inputs( forward_result[283] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/input.11', 'op': 'aten::linear', 'in': [253, 269, 284], 'output_id': 0, 'shape': [4, 10], 'out': [370], 'sorted_id': 285}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[285] = op;
            
            op->set_inputs( forward_result[253] );
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[284] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2957', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [370, 287], 'sorted_id': 286}
        {
            Tensor::shape_type shape = {1};
            Constant13.reshape( shape );
            forward_result[286] = new VariableTensor( Constant13, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2992', 'op': 'aten::sub', 'in': [269, 286, 12], 'output_id': 0, 'shape': [10, 400], 'out': [288], 'sorted_id': 287}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[287] = op;
            
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[286] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2993', 'op': 'aten::pow', 'in': [287, 45], 'output_id': 0, 'shape': [10, 400], 'out': [289], 'sorted_id': 288}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[288] = op;
            
            op->set_inputs( forward_result[287] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2994', 'op': 'aten::neg', 'in': [288], 'output_id': 0, 'shape': [10, 400], 'out': [293], 'sorted_id': 289}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[289] = op;
            
            op->set_inputs( forward_result[288] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2958', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 1.0, 'out': [294, 322, 291, 370, 319], 'sorted_id': 290}
        {
            Tensor::shape_type shape = {1};
            Constant14.reshape( shape );
            forward_result[290] = new VariableTensor( Constant14, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.17', 'op': 'aten::pow', 'in': [290, 45], 'output_id': 0, 'shape': [1], 'out': [292], 'sorted_id': 291}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[291] = op;
            
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2995', 'op': 'aten::mul', 'in': [291, 50], 'output_id': 0, 'shape': [1], 'out': [293], 'sorted_id': 292}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[292] = op;
            
            op->set_inputs( forward_result[291] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2996', 'op': 'aten::div', 'in': [289, 292], 'output_id': 0, 'shape': [10, 400], 'out': [295], 'sorted_id': 293}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[293] = op;
            
            op->set_inputs( forward_result[289] );
            op->set_inputs( forward_result[292] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.17', 'op': 'aten::log', 'in': [290], 'output_id': 0, 'shape': [1], 'out': [295], 'sorted_id': 294}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[294] = op;
            
            op->set_inputs( forward_result[290] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2997', 'op': 'aten::sub', 'in': [293, 294, 12], 'output_id': 0, 'shape': [10, 400], 'out': [296], 'sorted_id': 295}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[295] = op;
            
            op->set_inputs( forward_result[293] );
            op->set_inputs( forward_result[294] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2998', 'op': 'aten::sub', 'in': [295, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [297], 'sorted_id': 296}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[296] = op;
            
            op->set_inputs( forward_result[295] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.9', 'op': 'aten::exp', 'in': [296], 'output_id': 0, 'shape': [10, 400], 'out': [298], 'sorted_id': 297}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[297] = op;
            
            op->set_inputs( forward_result[296] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3011', 'op': 'aten::mul', 'in': [297, 58], 'output_id': 0, 'shape': [10, 400], 'out': [312], 'sorted_id': 298}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[298] = op;
            
            op->set_inputs( forward_result[297] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3002', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [370, 300], 'sorted_id': 299}
        {
            Tensor::shape_type shape = {1};
            Constant15.reshape( shape );
            forward_result[299] = new VariableTensor( Constant15, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3003', 'op': 'aten::sub', 'in': [269, 299, 12], 'output_id': 0, 'shape': [10, 400], 'out': [301], 'sorted_id': 300}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[300] = op;
            
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[299] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3004', 'op': 'aten::pow', 'in': [300, 45], 'output_id': 0, 'shape': [10, 400], 'out': [302], 'sorted_id': 301}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[301] = op;
            
            op->set_inputs( forward_result[300] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3005', 'op': 'aten::neg', 'in': [301], 'output_id': 0, 'shape': [10, 400], 'out': [306], 'sorted_id': 302}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[302] = op;
            
            op->set_inputs( forward_result[301] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/2956', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0025, 'out': [331, 334, 370, 304, 307], 'sorted_id': 303}
        {
            Tensor::shape_type shape = {1};
            Constant16.reshape( shape );
            forward_result[303] = new VariableTensor( Constant16, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.19', 'op': 'aten::pow', 'in': [303, 45], 'output_id': 0, 'shape': [1], 'out': [305], 'sorted_id': 304}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[304] = op;
            
            op->set_inputs( forward_result[303] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3006', 'op': 'aten::mul', 'in': [304, 50], 'output_id': 0, 'shape': [1], 'out': [306], 'sorted_id': 305}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[305] = op;
            
            op->set_inputs( forward_result[304] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3007', 'op': 'aten::div', 'in': [302, 305], 'output_id': 0, 'shape': [10, 400], 'out': [308], 'sorted_id': 306}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[306] = op;
            
            op->set_inputs( forward_result[302] );
            op->set_inputs( forward_result[305] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.19', 'op': 'aten::log', 'in': [303], 'output_id': 0, 'shape': [1], 'out': [308], 'sorted_id': 307}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[307] = op;
            
            op->set_inputs( forward_result[303] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3008', 'op': 'aten::sub', 'in': [306, 307, 12], 'output_id': 0, 'shape': [10, 400], 'out': [309], 'sorted_id': 308}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[308] = op;
            
            op->set_inputs( forward_result[306] );
            op->set_inputs( forward_result[307] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3009', 'op': 'aten::sub', 'in': [308, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [310], 'sorted_id': 309}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[309] = op;
            
            op->set_inputs( forward_result[308] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.9', 'op': 'aten::exp', 'in': [309], 'output_id': 0, 'shape': [10, 400], 'out': [311], 'sorted_id': 310}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[310] = op;
            
            op->set_inputs( forward_result[309] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3012', 'op': 'aten::mul', 'in': [310, 58], 'output_id': 0, 'shape': [10, 400], 'out': [312], 'sorted_id': 311}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[311] = op;
            
            op->set_inputs( forward_result[310] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3013', 'op': 'aten::add', 'in': [298, 311, 12], 'output_id': 0, 'shape': [10, 400], 'out': [313], 'sorted_id': 312}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[312] = op;
            
            op->set_inputs( forward_result[298] );
            op->set_inputs( forward_result[311] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3014', 'op': 'aten::log', 'in': [312], 'output_id': 0, 'shape': [10, 400], 'out': [314], 'sorted_id': 313}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[313] = op;
            
            op->set_inputs( forward_result[312] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3015', 'op': 'aten::sum', 'in': [313, 20], 'output_id': 0, 'shape': [], 'out': [342], 'sorted_id': 314}
        {
            SumOp* op = new SumOp();
            forward_result[314] = op;
            
            op->set_inputs( forward_result[313] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3018', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [370, 316], 'sorted_id': 315}
        {
            Tensor::shape_type shape = {1};
            Constant17.reshape( shape );
            forward_result[315] = new VariableTensor( Constant17, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3019', 'op': 'aten::sub', 'in': [284, 315, 12], 'output_id': 0, 'shape': [10], 'out': [317], 'sorted_id': 316}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[316] = op;
            
            op->set_inputs( forward_result[284] );
            op->set_inputs( forward_result[315] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3020', 'op': 'aten::pow', 'in': [316, 45], 'output_id': 0, 'shape': [10], 'out': [318], 'sorted_id': 317}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[317] = op;
            
            op->set_inputs( forward_result[316] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3021', 'op': 'aten::neg', 'in': [317], 'output_id': 0, 'shape': [10], 'out': [321], 'sorted_id': 318}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[318] = op;
            
            op->set_inputs( forward_result[317] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.21', 'op': 'aten::pow', 'in': [290, 45], 'output_id': 0, 'shape': [1], 'out': [320], 'sorted_id': 319}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[319] = op;
            
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3022', 'op': 'aten::mul', 'in': [319, 50], 'output_id': 0, 'shape': [1], 'out': [321], 'sorted_id': 320}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[320] = op;
            
            op->set_inputs( forward_result[319] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3023', 'op': 'aten::div', 'in': [318, 320], 'output_id': 0, 'shape': [10], 'out': [323], 'sorted_id': 321}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[321] = op;
            
            op->set_inputs( forward_result[318] );
            op->set_inputs( forward_result[320] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.21', 'op': 'aten::log', 'in': [290], 'output_id': 0, 'shape': [1], 'out': [323], 'sorted_id': 322}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[322] = op;
            
            op->set_inputs( forward_result[290] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3024', 'op': 'aten::sub', 'in': [321, 322, 12], 'output_id': 0, 'shape': [10], 'out': [324], 'sorted_id': 323}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[323] = op;
            
            op->set_inputs( forward_result[321] );
            op->set_inputs( forward_result[322] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3025', 'op': 'aten::sub', 'in': [323, 55, 12], 'output_id': 0, 'shape': [10], 'out': [325], 'sorted_id': 324}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[324] = op;
            
            op->set_inputs( forward_result[323] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.11', 'op': 'aten::exp', 'in': [324], 'output_id': 0, 'shape': [10], 'out': [326], 'sorted_id': 325}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[325] = op;
            
            op->set_inputs( forward_result[324] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3038', 'op': 'aten::mul', 'in': [325, 58], 'output_id': 0, 'shape': [10], 'out': [339], 'sorted_id': 326}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[326] = op;
            
            op->set_inputs( forward_result[325] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3029', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [1], 'constant_value': 0.0, 'out': [370, 328], 'sorted_id': 327}
        {
            Tensor::shape_type shape = {1};
            Constant18.reshape( shape );
            forward_result[327] = new VariableTensor( Constant18, 1 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3030', 'op': 'aten::sub', 'in': [284, 327, 12], 'output_id': 0, 'shape': [10], 'out': [329], 'sorted_id': 328}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[328] = op;
            
            op->set_inputs( forward_result[284] );
            op->set_inputs( forward_result[327] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3031', 'op': 'aten::pow', 'in': [328, 45], 'output_id': 0, 'shape': [10], 'out': [330], 'sorted_id': 329}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[329] = op;
            
            op->set_inputs( forward_result[328] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3032', 'op': 'aten::neg', 'in': [329], 'output_id': 0, 'shape': [10], 'out': [333], 'sorted_id': 330}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[330] = op;
            
            op->set_inputs( forward_result[329] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.23', 'op': 'aten::pow', 'in': [303, 45], 'output_id': 0, 'shape': [1], 'out': [332], 'sorted_id': 331}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[331] = op;
            
            op->set_inputs( forward_result[303] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3033', 'op': 'aten::mul', 'in': [331, 50], 'output_id': 0, 'shape': [1], 'out': [333], 'sorted_id': 332}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[332] = op;
            
            op->set_inputs( forward_result[331] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3034', 'op': 'aten::div', 'in': [330, 332], 'output_id': 0, 'shape': [10], 'out': [335], 'sorted_id': 333}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[333] = op;
            
            op->set_inputs( forward_result[330] );
            op->set_inputs( forward_result[332] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.23', 'op': 'aten::log', 'in': [303], 'output_id': 0, 'shape': [1], 'out': [335], 'sorted_id': 334}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[334] = op;
            
            op->set_inputs( forward_result[303] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3035', 'op': 'aten::sub', 'in': [333, 334, 12], 'output_id': 0, 'shape': [10], 'out': [336], 'sorted_id': 335}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[335] = op;
            
            op->set_inputs( forward_result[333] );
            op->set_inputs( forward_result[334] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3036', 'op': 'aten::sub', 'in': [335, 55, 12], 'output_id': 0, 'shape': [10], 'out': [337], 'sorted_id': 336}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[336] = op;
            
            op->set_inputs( forward_result[335] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.11', 'op': 'aten::exp', 'in': [336], 'output_id': 0, 'shape': [10], 'out': [338], 'sorted_id': 337}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[337] = op;
            
            op->set_inputs( forward_result[336] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3039', 'op': 'aten::mul', 'in': [337, 58], 'output_id': 0, 'shape': [10], 'out': [339], 'sorted_id': 338}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[338] = op;
            
            op->set_inputs( forward_result[337] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3040', 'op': 'aten::add', 'in': [326, 338, 12], 'output_id': 0, 'shape': [10], 'out': [340], 'sorted_id': 339}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[339] = op;
            
            op->set_inputs( forward_result[326] );
            op->set_inputs( forward_result[338] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3041', 'op': 'aten::log', 'in': [339], 'output_id': 0, 'shape': [10], 'out': [341], 'sorted_id': 340}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[340] = op;
            
            op->set_inputs( forward_result[339] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3042', 'op': 'aten::sum', 'in': [340, 20], 'output_id': 0, 'shape': [], 'out': [342], 'sorted_id': 341}
        {
            SumOp* op = new SumOp();
            forward_result[341] = op;
            
            op->set_inputs( forward_result[340] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3043', 'op': 'aten::add', 'in': [314, 341, 12], 'output_id': 0, 'shape': [], 'out': [370], 'sorted_id': 342}
        {
            AddOp* op = new AddOp();
            forward_result[342] = op;
            
            op->set_inputs( forward_result[314] );
            op->set_inputs( forward_result[341] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3044', 'op': 'aten::exp', 'in': [255], 'output_id': 0, 'shape': [10, 400], 'out': [344], 'sorted_id': 343}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[343] = op;
            
            op->set_inputs( forward_result[255] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3045', 'op': 'aten::log1p', 'in': [343], 'output_id': 0, 'shape': [10, 400], 'out': [345], 'sorted_id': 344}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[344] = op;
            
            op->set_inputs( forward_result[343] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3046', 'op': 'aten::log', 'in': [344], 'output_id': 0, 'shape': [10, 400], 'out': [346], 'sorted_id': 345}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[345] = op;
            
            op->set_inputs( forward_result[344] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3047', 'op': 'aten::rsub', 'in': [345, 107, 12], 'output_id': 0, 'shape': [10, 400], 'out': [354], 'sorted_id': 346}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[346] = op;
            
            op->set_inputs( forward_result[345] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3048', 'op': 'aten::sub', 'in': [269, 254, 12], 'output_id': 0, 'shape': [10, 400], 'out': [348], 'sorted_id': 347}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[347] = op;
            
            op->set_inputs( forward_result[269] );
            op->set_inputs( forward_result[254] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3049', 'op': 'aten::pow', 'in': [347, 45], 'output_id': 0, 'shape': [10, 400], 'out': [353], 'sorted_id': 348}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[348] = op;
            
            op->set_inputs( forward_result[347] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3050', 'op': 'aten::exp', 'in': [255], 'output_id': 0, 'shape': [10, 400], 'out': [350], 'sorted_id': 349}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[349] = op;
            
            op->set_inputs( forward_result[255] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3051', 'op': 'aten::log1p', 'in': [349], 'output_id': 0, 'shape': [10, 400], 'out': [351], 'sorted_id': 350}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[350] = op;
            
            op->set_inputs( forward_result[349] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3052', 'op': 'aten::pow', 'in': [350, 45], 'output_id': 0, 'shape': [10, 400], 'out': [352], 'sorted_id': 351}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[351] = op;
            
            op->set_inputs( forward_result[350] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3053', 'op': 'aten::mul', 'in': [351, 50], 'output_id': 0, 'shape': [10, 400], 'out': [353], 'sorted_id': 352}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[352] = op;
            
            op->set_inputs( forward_result[351] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3054', 'op': 'aten::div', 'in': [348, 352], 'output_id': 0, 'shape': [10, 400], 'out': [354], 'sorted_id': 353}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[353] = op;
            
            op->set_inputs( forward_result[348] );
            op->set_inputs( forward_result[352] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3055', 'op': 'aten::sub', 'in': [346, 353, 12], 'output_id': 0, 'shape': [10, 400], 'out': [355], 'sorted_id': 354}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[354] = op;
            
            op->set_inputs( forward_result[346] );
            op->set_inputs( forward_result[353] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3056', 'op': 'aten::sum', 'in': [354, 20], 'output_id': 0, 'shape': [], 'out': [369], 'sorted_id': 355}
        {
            SumOp* op = new SumOp();
            forward_result[355] = op;
            
            op->set_inputs( forward_result[354] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3057', 'op': 'aten::exp', 'in': [271], 'output_id': 0, 'shape': [10], 'out': [357], 'sorted_id': 356}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[356] = op;
            
            op->set_inputs( forward_result[271] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3058', 'op': 'aten::log1p', 'in': [356], 'output_id': 0, 'shape': [10], 'out': [358], 'sorted_id': 357}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[357] = op;
            
            op->set_inputs( forward_result[356] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3059', 'op': 'aten::log', 'in': [357], 'output_id': 0, 'shape': [10], 'out': [359], 'sorted_id': 358}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[358] = op;
            
            op->set_inputs( forward_result[357] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3060', 'op': 'aten::rsub', 'in': [358, 107, 12], 'output_id': 0, 'shape': [10], 'out': [367], 'sorted_id': 359}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[359] = op;
            
            op->set_inputs( forward_result[358] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3061', 'op': 'aten::sub', 'in': [284, 270, 12], 'output_id': 0, 'shape': [10], 'out': [361], 'sorted_id': 360}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[360] = op;
            
            op->set_inputs( forward_result[284] );
            op->set_inputs( forward_result[270] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3062', 'op': 'aten::pow', 'in': [360, 45], 'output_id': 0, 'shape': [10], 'out': [366], 'sorted_id': 361}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[361] = op;
            
            op->set_inputs( forward_result[360] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3063', 'op': 'aten::exp', 'in': [271], 'output_id': 0, 'shape': [10], 'out': [363], 'sorted_id': 362}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[362] = op;
            
            op->set_inputs( forward_result[271] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3064', 'op': 'aten::log1p', 'in': [362], 'output_id': 0, 'shape': [10], 'out': [364], 'sorted_id': 363}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[363] = op;
            
            op->set_inputs( forward_result[362] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3065', 'op': 'aten::pow', 'in': [363, 45], 'output_id': 0, 'shape': [10], 'out': [365], 'sorted_id': 364}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[364] = op;
            
            op->set_inputs( forward_result[363] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3066', 'op': 'aten::mul', 'in': [364, 50], 'output_id': 0, 'shape': [10], 'out': [366], 'sorted_id': 365}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[365] = op;
            
            op->set_inputs( forward_result[364] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3067', 'op': 'aten::div', 'in': [361, 365], 'output_id': 0, 'shape': [10], 'out': [367], 'sorted_id': 366}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[366] = op;
            
            op->set_inputs( forward_result[361] );
            op->set_inputs( forward_result[365] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3068', 'op': 'aten::sub', 'in': [359, 366, 12], 'output_id': 0, 'shape': [10], 'out': [368], 'sorted_id': 367}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[367] = op;
            
            op->set_inputs( forward_result[359] );
            op->set_inputs( forward_result[366] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3069', 'op': 'aten::sum', 'in': [367, 20], 'output_id': 0, 'shape': [], 'out': [369], 'sorted_id': 368}
        {
            SumOp* op = new SumOp();
            forward_result[368] = op;
            
            op->set_inputs( forward_result[367] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3070', 'op': 'aten::add', 'in': [355, 368, 12], 'output_id': 0, 'shape': [], 'out': [370], 'sorted_id': 369}
        {
            AddOp* op = new AddOp();
            forward_result[369] = op;
            
            op->set_inputs( forward_result[355] );
            op->set_inputs( forward_result[368] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3072', 'op': 'prim::TupleConstruct', 'in': [285, 342, 369, 258, 263, 274, 278, 290, 286, 303, 299, 315, 327], 'output_id': 0, 'shape': [], 'out': [674, 1086, 649, 637, 662, 622, 1068, 633, 645, 686, 658, 617, 371], 'sorted_id': 370}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[370] = op;
            
            op->set_inputs( forward_result[285] );
            op->set_inputs( forward_result[342] );
            op->set_inputs( forward_result[369] );
            op->set_inputs( forward_result[258] );
            op->set_inputs( forward_result[263] );
            op->set_inputs( forward_result[274] );
            op->set_inputs( forward_result[278] );
            op->set_inputs( forward_result[290] );
            op->set_inputs( forward_result[286] );
            op->set_inputs( forward_result[303] );
            op->set_inputs( forward_result[299] );
            op->set_inputs( forward_result[315] );
            op->set_inputs( forward_result[327] );
        }
        
        // {'name': 'Model/3073', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 0, 'shape': [4, 10], 'out': [372], 'sorted_id': 371}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[371] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/3086', 'op': 'aten::log_softmax', 'in': [371, 12, 20], 'output_id': 0, 'shape': [4, 10], 'out': [1061], 'sorted_id': 372}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[372] = op;
            
            op->set_inputs( forward_result[371] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/3091', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [374], 'sorted_id': 373}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[373] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Model/Net[net]/input.13', 'op': 'aten::view', 'in': [0, 373], 'output_id': 0, 'shape': [4, 784], 'out': [406], 'sorted_id': 374}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[374] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[373] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_mu/weight_mu.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [390, 468], 'sorted_id': 375}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[375] = new VariableTensor( l1_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_rho/weight_rho.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [464, 377, 381, 470, 380], 'sorted_id': 376}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[376] = new VariableTensor( l1_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3105', 'op': 'aten::exp', 'in': [376], 'output_id': 0, 'shape': [400, 784], 'out': [378], 'sorted_id': 377}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[377] = op;
            
            op->set_inputs( forward_result[376] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3106', 'op': 'aten::log1p', 'in': [377], 'output_id': 0, 'shape': [400, 784], 'out': [389], 'sorted_id': 378}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[378] = op;
            
            op->set_inputs( forward_result[377] );
        }
        
        // {'name': 'Model/2814', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 3, 'shape': [], 'out': [741, 383], 'sorted_id': 379}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[379] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3097', 'op': 'aten::size', 'in': [376, 10], 'output_id': 0, 'shape': [], 'out': [385, 382], 'sorted_id': 380}
        {
            SizeOp* op = new SizeOp();
            forward_result[380] = op;
            
            op->set_inputs( forward_result[376] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3098', 'op': 'aten::size', 'in': [376, 12], 'output_id': 0, 'shape': [], 'out': [385, 382], 'sorted_id': 381}
        {
            SizeOp* op = new SizeOp();
            forward_result[381] = op;
            
            op->set_inputs( forward_result[376] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3099', 'op': 'prim::ListConstruct', 'in': [380, 381], 'output_id': 0, 'shape': [], 'out': [383], 'sorted_id': 382}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[382] = op;
            
            op->set_inputs( forward_result[380] );
            op->set_inputs( forward_result[381] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3100', 'op': 'aten::expand', 'in': [379, 382, 15], 'output_id': 0, 'shape': [400, 784], 'out': [387], 'sorted_id': 383}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[383] = op;
            
            op->set_inputs( forward_result[379] );
            op->set_inputs( forward_result[382] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/2815', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 4, 'shape': [], 'out': [386, 743], 'sorted_id': 384}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[384] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3101', 'op': 'prim::ListConstruct', 'in': [380, 381], 'output_id': 0, 'shape': [], 'out': [386], 'sorted_id': 385}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[385] = op;
            
            op->set_inputs( forward_result[380] );
            op->set_inputs( forward_result[381] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3102', 'op': 'aten::expand', 'in': [384, 385, 15], 'output_id': 0, 'shape': [400, 784], 'out': [387], 'sorted_id': 386}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[386] = op;
            
            op->set_inputs( forward_result[384] );
            op->set_inputs( forward_result[385] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3103', 'op': 'aten::normal', 'in': [383, 386, 20], 'output_id': 0, 'shape': [400, 784], 'out': [388], 'sorted_id': 387}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[387] = op;
            
            op->set_inputs( forward_result[383] );
            op->set_inputs( forward_result[386] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.13', 'op': 'aten::to', 'in': [387, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 784], 'out': [389], 'sorted_id': 388}
        {
            Tensor::shape_type shape = {400,784};
            ToOp* op = new ToOp();
            forward_result[388] = op;
            
            op->set_inputs( forward_result[387] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3107', 'op': 'aten::mul', 'in': [378, 388], 'output_id': 0, 'shape': [400, 784], 'out': [390], 'sorted_id': 389}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[389] = op;
            
            op->set_inputs( forward_result[378] );
            op->set_inputs( forward_result[388] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.13', 'op': 'aten::add', 'in': [375, 389, 12], 'output_id': 0, 'shape': [400, 784], 'out': [468, 406, 421, 408], 'sorted_id': 390}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[390] = op;
            
            op->set_inputs( forward_result[375] );
            op->set_inputs( forward_result[389] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_mu/bias_mu.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [405, 481], 'sorted_id': 391}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[391] = new VariableTensor( l1_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_rho/bias_rho.7', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [483, 393, 396, 477], 'sorted_id': 392}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[392] = new VariableTensor( l1_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3116', 'op': 'aten::exp', 'in': [392], 'output_id': 0, 'shape': [400], 'out': [394], 'sorted_id': 393}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[393] = op;
            
            op->set_inputs( forward_result[392] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3117', 'op': 'aten::log1p', 'in': [393], 'output_id': 0, 'shape': [400], 'out': [404], 'sorted_id': 394}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[394] = op;
            
            op->set_inputs( forward_result[393] );
        }
        
        // {'name': 'Model/2816', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 5, 'shape': [], 'out': [398, 754], 'sorted_id': 395}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[395] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3109', 'op': 'aten::size', 'in': [392, 10], 'output_id': 0, 'shape': [], 'out': [397, 400], 'sorted_id': 396}
        {
            SizeOp* op = new SizeOp();
            forward_result[396] = op;
            
            op->set_inputs( forward_result[392] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3110', 'op': 'prim::ListConstruct', 'in': [396], 'output_id': 0, 'shape': [], 'out': [398], 'sorted_id': 397}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[397] = op;
            
            op->set_inputs( forward_result[396] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3111', 'op': 'aten::expand', 'in': [395, 397, 15], 'output_id': 0, 'shape': [400], 'out': [402], 'sorted_id': 398}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[398] = op;
            
            op->set_inputs( forward_result[395] );
            op->set_inputs( forward_result[397] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/2817', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 6, 'shape': [], 'out': [401, 756], 'sorted_id': 399}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 6 );
            forward_result[399] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3112', 'op': 'prim::ListConstruct', 'in': [396], 'output_id': 0, 'shape': [], 'out': [401], 'sorted_id': 400}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[400] = op;
            
            op->set_inputs( forward_result[396] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3113', 'op': 'aten::expand', 'in': [399, 400, 15], 'output_id': 0, 'shape': [400], 'out': [402], 'sorted_id': 401}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[401] = op;
            
            op->set_inputs( forward_result[399] );
            op->set_inputs( forward_result[400] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3114', 'op': 'aten::normal', 'in': [398, 401, 20], 'output_id': 0, 'shape': [400], 'out': [403], 'sorted_id': 402}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[402] = op;
            
            op->set_inputs( forward_result[398] );
            op->set_inputs( forward_result[401] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.15', 'op': 'aten::to', 'in': [402, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [404], 'sorted_id': 403}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[403] = op;
            
            op->set_inputs( forward_result[402] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3118', 'op': 'aten::mul', 'in': [394, 403], 'output_id': 0, 'shape': [400], 'out': [405], 'sorted_id': 404}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[404] = op;
            
            op->set_inputs( forward_result[394] );
            op->set_inputs( forward_result[403] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.15', 'op': 'aten::add', 'in': [391, 404, 12], 'output_id': 0, 'shape': [400], 'out': [406, 449, 481, 437], 'sorted_id': 405}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[405] = op;
            
            op->set_inputs( forward_result[391] );
            op->set_inputs( forward_result[404] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/input.15', 'op': 'aten::linear', 'in': [374, 390, 405], 'output_id': 0, 'shape': [4, 400], 'out': [491], 'sorted_id': 406}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[406] = op;
            
            op->set_inputs( forward_result[374] );
            op->set_inputs( forward_result[390] );
            op->set_inputs( forward_result[405] );
        }
        
        // {'name': 'Model/2819', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 8, 'shape': [1], 'out': [762, 408], 'sorted_id': 407}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 8 );
            forward_result[407] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3122', 'op': 'aten::sub', 'in': [390, 407, 12], 'output_id': 0, 'shape': [400, 784], 'out': [409], 'sorted_id': 408}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[408] = op;
            
            op->set_inputs( forward_result[390] );
            op->set_inputs( forward_result[407] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3123', 'op': 'aten::pow', 'in': [408, 45], 'output_id': 0, 'shape': [400, 784], 'out': [410], 'sorted_id': 409}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[409] = op;
            
            op->set_inputs( forward_result[408] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3124', 'op': 'aten::neg', 'in': [409], 'output_id': 0, 'shape': [400, 784], 'out': [414], 'sorted_id': 410}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[410] = op;
            
            op->set_inputs( forward_result[409] );
        }
        
        // {'name': 'Model/2818', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 7, 'shape': [1], 'out': [440, 415, 443, 412, 790, 768, 765, 793], 'sorted_id': 411}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 7 );
            forward_result[411] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.25', 'op': 'aten::pow', 'in': [411, 45], 'output_id': 0, 'shape': [1], 'out': [413], 'sorted_id': 412}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[412] = op;
            
            op->set_inputs( forward_result[411] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3125', 'op': 'aten::mul', 'in': [412, 50], 'output_id': 0, 'shape': [1], 'out': [414], 'sorted_id': 413}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[413] = op;
            
            op->set_inputs( forward_result[412] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3126', 'op': 'aten::div', 'in': [410, 413], 'output_id': 0, 'shape': [400, 784], 'out': [416], 'sorted_id': 414}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[414] = op;
            
            op->set_inputs( forward_result[410] );
            op->set_inputs( forward_result[413] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.25', 'op': 'aten::log', 'in': [411], 'output_id': 0, 'shape': [1], 'out': [416], 'sorted_id': 415}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[415] = op;
            
            op->set_inputs( forward_result[411] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3127', 'op': 'aten::sub', 'in': [414, 415, 12], 'output_id': 0, 'shape': [400, 784], 'out': [417], 'sorted_id': 416}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[416] = op;
            
            op->set_inputs( forward_result[414] );
            op->set_inputs( forward_result[415] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3128', 'op': 'aten::sub', 'in': [416, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [418], 'sorted_id': 417}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[417] = op;
            
            op->set_inputs( forward_result[416] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.13', 'op': 'aten::exp', 'in': [417], 'output_id': 0, 'shape': [400, 784], 'out': [419], 'sorted_id': 418}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[418] = op;
            
            op->set_inputs( forward_result[417] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3140', 'op': 'aten::mul', 'in': [418, 58], 'output_id': 0, 'shape': [400, 784], 'out': [433], 'sorted_id': 419}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[419] = op;
            
            op->set_inputs( forward_result[418] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/2821', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 10, 'shape': [1], 'out': [421, 773], 'sorted_id': 420}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 10 );
            forward_result[420] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3132', 'op': 'aten::sub', 'in': [390, 420, 12], 'output_id': 0, 'shape': [400, 784], 'out': [422], 'sorted_id': 421}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[421] = op;
            
            op->set_inputs( forward_result[390] );
            op->set_inputs( forward_result[420] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3133', 'op': 'aten::pow', 'in': [421, 45], 'output_id': 0, 'shape': [400, 784], 'out': [423], 'sorted_id': 422}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[422] = op;
            
            op->set_inputs( forward_result[421] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3134', 'op': 'aten::neg', 'in': [422], 'output_id': 0, 'shape': [400, 784], 'out': [427], 'sorted_id': 423}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[423] = op;
            
            op->set_inputs( forward_result[422] );
        }
        
        // {'name': 'Model/2820', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 9, 'shape': [1], 'out': [804, 425, 428, 801, 452, 779, 455, 776], 'sorted_id': 424}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 9 );
            forward_result[424] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.27', 'op': 'aten::pow', 'in': [424, 45], 'output_id': 0, 'shape': [1], 'out': [426], 'sorted_id': 425}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[425] = op;
            
            op->set_inputs( forward_result[424] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3135', 'op': 'aten::mul', 'in': [425, 50], 'output_id': 0, 'shape': [1], 'out': [427], 'sorted_id': 426}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[426] = op;
            
            op->set_inputs( forward_result[425] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3136', 'op': 'aten::div', 'in': [423, 426], 'output_id': 0, 'shape': [400, 784], 'out': [429], 'sorted_id': 427}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[427] = op;
            
            op->set_inputs( forward_result[423] );
            op->set_inputs( forward_result[426] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.27', 'op': 'aten::log', 'in': [424], 'output_id': 0, 'shape': [1], 'out': [429], 'sorted_id': 428}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[428] = op;
            
            op->set_inputs( forward_result[424] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3137', 'op': 'aten::sub', 'in': [427, 428, 12], 'output_id': 0, 'shape': [400, 784], 'out': [430], 'sorted_id': 429}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[429] = op;
            
            op->set_inputs( forward_result[427] );
            op->set_inputs( forward_result[428] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3138', 'op': 'aten::sub', 'in': [429, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [431], 'sorted_id': 430}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[430] = op;
            
            op->set_inputs( forward_result[429] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.13', 'op': 'aten::exp', 'in': [430], 'output_id': 0, 'shape': [400, 784], 'out': [432], 'sorted_id': 431}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[431] = op;
            
            op->set_inputs( forward_result[430] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3141', 'op': 'aten::mul', 'in': [431, 58], 'output_id': 0, 'shape': [400, 784], 'out': [433], 'sorted_id': 432}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[432] = op;
            
            op->set_inputs( forward_result[431] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3142', 'op': 'aten::add', 'in': [419, 432, 12], 'output_id': 0, 'shape': [400, 784], 'out': [434], 'sorted_id': 433}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[433] = op;
            
            op->set_inputs( forward_result[419] );
            op->set_inputs( forward_result[432] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3143', 'op': 'aten::log', 'in': [433], 'output_id': 0, 'shape': [400, 784], 'out': [435], 'sorted_id': 434}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[434] = op;
            
            op->set_inputs( forward_result[433] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3144', 'op': 'aten::sum', 'in': [434, 20], 'output_id': 0, 'shape': [], 'out': [463], 'sorted_id': 435}
        {
            SumOp* op = new SumOp();
            forward_result[435] = op;
            
            op->set_inputs( forward_result[434] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/2822', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 11, 'shape': [1], 'out': [787, 437], 'sorted_id': 436}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 11 );
            forward_result[436] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3147', 'op': 'aten::sub', 'in': [405, 436, 12], 'output_id': 0, 'shape': [400], 'out': [438], 'sorted_id': 437}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[437] = op;
            
            op->set_inputs( forward_result[405] );
            op->set_inputs( forward_result[436] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3148', 'op': 'aten::pow', 'in': [437, 45], 'output_id': 0, 'shape': [400], 'out': [439], 'sorted_id': 438}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[438] = op;
            
            op->set_inputs( forward_result[437] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3149', 'op': 'aten::neg', 'in': [438], 'output_id': 0, 'shape': [400], 'out': [442], 'sorted_id': 439}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[439] = op;
            
            op->set_inputs( forward_result[438] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.29', 'op': 'aten::pow', 'in': [411, 45], 'output_id': 0, 'shape': [1], 'out': [441], 'sorted_id': 440}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[440] = op;
            
            op->set_inputs( forward_result[411] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3150', 'op': 'aten::mul', 'in': [440, 50], 'output_id': 0, 'shape': [1], 'out': [442], 'sorted_id': 441}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[441] = op;
            
            op->set_inputs( forward_result[440] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3151', 'op': 'aten::div', 'in': [439, 441], 'output_id': 0, 'shape': [400], 'out': [444], 'sorted_id': 442}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[442] = op;
            
            op->set_inputs( forward_result[439] );
            op->set_inputs( forward_result[441] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.29', 'op': 'aten::log', 'in': [411], 'output_id': 0, 'shape': [1], 'out': [444], 'sorted_id': 443}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[443] = op;
            
            op->set_inputs( forward_result[411] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3152', 'op': 'aten::sub', 'in': [442, 443, 12], 'output_id': 0, 'shape': [400], 'out': [445], 'sorted_id': 444}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[444] = op;
            
            op->set_inputs( forward_result[442] );
            op->set_inputs( forward_result[443] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3153', 'op': 'aten::sub', 'in': [444, 55, 12], 'output_id': 0, 'shape': [400], 'out': [446], 'sorted_id': 445}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[445] = op;
            
            op->set_inputs( forward_result[444] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.15', 'op': 'aten::exp', 'in': [445], 'output_id': 0, 'shape': [400], 'out': [447], 'sorted_id': 446}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[446] = op;
            
            op->set_inputs( forward_result[445] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3165', 'op': 'aten::mul', 'in': [446, 58], 'output_id': 0, 'shape': [400], 'out': [460], 'sorted_id': 447}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[447] = op;
            
            op->set_inputs( forward_result[446] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/2823', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 12, 'shape': [1], 'out': [449, 798], 'sorted_id': 448}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 12 );
            forward_result[448] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3157', 'op': 'aten::sub', 'in': [405, 448, 12], 'output_id': 0, 'shape': [400], 'out': [450], 'sorted_id': 449}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[449] = op;
            
            op->set_inputs( forward_result[405] );
            op->set_inputs( forward_result[448] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3158', 'op': 'aten::pow', 'in': [449, 45], 'output_id': 0, 'shape': [400], 'out': [451], 'sorted_id': 450}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[450] = op;
            
            op->set_inputs( forward_result[449] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3159', 'op': 'aten::neg', 'in': [450], 'output_id': 0, 'shape': [400], 'out': [454], 'sorted_id': 451}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[451] = op;
            
            op->set_inputs( forward_result[450] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.31', 'op': 'aten::pow', 'in': [424, 45], 'output_id': 0, 'shape': [1], 'out': [453], 'sorted_id': 452}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[452] = op;
            
            op->set_inputs( forward_result[424] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3160', 'op': 'aten::mul', 'in': [452, 50], 'output_id': 0, 'shape': [1], 'out': [454], 'sorted_id': 453}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[453] = op;
            
            op->set_inputs( forward_result[452] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3161', 'op': 'aten::div', 'in': [451, 453], 'output_id': 0, 'shape': [400], 'out': [456], 'sorted_id': 454}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[454] = op;
            
            op->set_inputs( forward_result[451] );
            op->set_inputs( forward_result[453] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.31', 'op': 'aten::log', 'in': [424], 'output_id': 0, 'shape': [1], 'out': [456], 'sorted_id': 455}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[455] = op;
            
            op->set_inputs( forward_result[424] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3162', 'op': 'aten::sub', 'in': [454, 455, 12], 'output_id': 0, 'shape': [400], 'out': [457], 'sorted_id': 456}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[456] = op;
            
            op->set_inputs( forward_result[454] );
            op->set_inputs( forward_result[455] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3163', 'op': 'aten::sub', 'in': [456, 55, 12], 'output_id': 0, 'shape': [400], 'out': [458], 'sorted_id': 457}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[457] = op;
            
            op->set_inputs( forward_result[456] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.15', 'op': 'aten::exp', 'in': [457], 'output_id': 0, 'shape': [400], 'out': [459], 'sorted_id': 458}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[458] = op;
            
            op->set_inputs( forward_result[457] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3166', 'op': 'aten::mul', 'in': [458, 58], 'output_id': 0, 'shape': [400], 'out': [460], 'sorted_id': 459}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[459] = op;
            
            op->set_inputs( forward_result[458] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3167', 'op': 'aten::add', 'in': [447, 459, 12], 'output_id': 0, 'shape': [400], 'out': [461], 'sorted_id': 460}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[460] = op;
            
            op->set_inputs( forward_result[447] );
            op->set_inputs( forward_result[459] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3168', 'op': 'aten::log', 'in': [460], 'output_id': 0, 'shape': [400], 'out': [462], 'sorted_id': 461}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[461] = op;
            
            op->set_inputs( forward_result[460] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3169', 'op': 'aten::sum', 'in': [461, 20], 'output_id': 0, 'shape': [], 'out': [463], 'sorted_id': 462}
        {
            SumOp* op = new SumOp();
            forward_result[462] = op;
            
            op->set_inputs( forward_result[461] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3170', 'op': 'aten::add', 'in': [435, 462, 12], 'output_id': 0, 'shape': [], 'out': [491], 'sorted_id': 463}
        {
            AddOp* op = new AddOp();
            forward_result[463] = op;
            
            op->set_inputs( forward_result[435] );
            op->set_inputs( forward_result[462] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3171', 'op': 'aten::exp', 'in': [376], 'output_id': 0, 'shape': [400, 784], 'out': [465], 'sorted_id': 464}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[464] = op;
            
            op->set_inputs( forward_result[376] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3172', 'op': 'aten::log1p', 'in': [464], 'output_id': 0, 'shape': [400, 784], 'out': [466], 'sorted_id': 465}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[465] = op;
            
            op->set_inputs( forward_result[464] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3173', 'op': 'aten::log', 'in': [465], 'output_id': 0, 'shape': [400, 784], 'out': [467], 'sorted_id': 466}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[466] = op;
            
            op->set_inputs( forward_result[465] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3174', 'op': 'aten::rsub', 'in': [466, 107, 12], 'output_id': 0, 'shape': [400, 784], 'out': [475], 'sorted_id': 467}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[467] = op;
            
            op->set_inputs( forward_result[466] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3175', 'op': 'aten::sub', 'in': [390, 375, 12], 'output_id': 0, 'shape': [400, 784], 'out': [469], 'sorted_id': 468}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[468] = op;
            
            op->set_inputs( forward_result[390] );
            op->set_inputs( forward_result[375] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3176', 'op': 'aten::pow', 'in': [468, 45], 'output_id': 0, 'shape': [400, 784], 'out': [474], 'sorted_id': 469}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[469] = op;
            
            op->set_inputs( forward_result[468] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3177', 'op': 'aten::exp', 'in': [376], 'output_id': 0, 'shape': [400, 784], 'out': [471], 'sorted_id': 470}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[470] = op;
            
            op->set_inputs( forward_result[376] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3178', 'op': 'aten::log1p', 'in': [470], 'output_id': 0, 'shape': [400, 784], 'out': [472], 'sorted_id': 471}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[471] = op;
            
            op->set_inputs( forward_result[470] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3179', 'op': 'aten::pow', 'in': [471, 45], 'output_id': 0, 'shape': [400, 784], 'out': [473], 'sorted_id': 472}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[472] = op;
            
            op->set_inputs( forward_result[471] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3180', 'op': 'aten::mul', 'in': [472, 50], 'output_id': 0, 'shape': [400, 784], 'out': [474], 'sorted_id': 473}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[473] = op;
            
            op->set_inputs( forward_result[472] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3181', 'op': 'aten::div', 'in': [469, 473], 'output_id': 0, 'shape': [400, 784], 'out': [475], 'sorted_id': 474}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[474] = op;
            
            op->set_inputs( forward_result[469] );
            op->set_inputs( forward_result[473] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3182', 'op': 'aten::sub', 'in': [467, 474, 12], 'output_id': 0, 'shape': [400, 784], 'out': [476], 'sorted_id': 475}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[475] = op;
            
            op->set_inputs( forward_result[467] );
            op->set_inputs( forward_result[474] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3183', 'op': 'aten::sum', 'in': [475, 20], 'output_id': 0, 'shape': [], 'out': [490], 'sorted_id': 476}
        {
            SumOp* op = new SumOp();
            forward_result[476] = op;
            
            op->set_inputs( forward_result[475] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3184', 'op': 'aten::exp', 'in': [392], 'output_id': 0, 'shape': [400], 'out': [478], 'sorted_id': 477}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[477] = op;
            
            op->set_inputs( forward_result[392] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3185', 'op': 'aten::log1p', 'in': [477], 'output_id': 0, 'shape': [400], 'out': [479], 'sorted_id': 478}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[478] = op;
            
            op->set_inputs( forward_result[477] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3186', 'op': 'aten::log', 'in': [478], 'output_id': 0, 'shape': [400], 'out': [480], 'sorted_id': 479}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[479] = op;
            
            op->set_inputs( forward_result[478] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3187', 'op': 'aten::rsub', 'in': [479, 107, 12], 'output_id': 0, 'shape': [400], 'out': [488], 'sorted_id': 480}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[480] = op;
            
            op->set_inputs( forward_result[479] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3188', 'op': 'aten::sub', 'in': [405, 391, 12], 'output_id': 0, 'shape': [400], 'out': [482], 'sorted_id': 481}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[481] = op;
            
            op->set_inputs( forward_result[405] );
            op->set_inputs( forward_result[391] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3189', 'op': 'aten::pow', 'in': [481, 45], 'output_id': 0, 'shape': [400], 'out': [487], 'sorted_id': 482}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[482] = op;
            
            op->set_inputs( forward_result[481] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3190', 'op': 'aten::exp', 'in': [392], 'output_id': 0, 'shape': [400], 'out': [484], 'sorted_id': 483}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[483] = op;
            
            op->set_inputs( forward_result[392] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3191', 'op': 'aten::log1p', 'in': [483], 'output_id': 0, 'shape': [400], 'out': [485], 'sorted_id': 484}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[484] = op;
            
            op->set_inputs( forward_result[483] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3192', 'op': 'aten::pow', 'in': [484, 45], 'output_id': 0, 'shape': [400], 'out': [486], 'sorted_id': 485}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[485] = op;
            
            op->set_inputs( forward_result[484] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3193', 'op': 'aten::mul', 'in': [485, 50], 'output_id': 0, 'shape': [400], 'out': [487], 'sorted_id': 486}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[486] = op;
            
            op->set_inputs( forward_result[485] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3194', 'op': 'aten::div', 'in': [482, 486], 'output_id': 0, 'shape': [400], 'out': [488], 'sorted_id': 487}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[487] = op;
            
            op->set_inputs( forward_result[482] );
            op->set_inputs( forward_result[486] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3195', 'op': 'aten::sub', 'in': [480, 487, 12], 'output_id': 0, 'shape': [400], 'out': [489], 'sorted_id': 488}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[488] = op;
            
            op->set_inputs( forward_result[480] );
            op->set_inputs( forward_result[487] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3196', 'op': 'aten::sum', 'in': [488, 20], 'output_id': 0, 'shape': [], 'out': [490], 'sorted_id': 489}
        {
            SumOp* op = new SumOp();
            forward_result[489] = op;
            
            op->set_inputs( forward_result[488] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3197', 'op': 'aten::add', 'in': [476, 489, 12], 'output_id': 0, 'shape': [], 'out': [491], 'sorted_id': 490}
        {
            AddOp* op = new AddOp();
            forward_result[490] = op;
            
            op->set_inputs( forward_result[476] );
            op->set_inputs( forward_result[489] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3199', 'op': 'prim::TupleConstruct', 'in': [406, 463, 490], 'output_id': 0, 'shape': [], 'out': [1088, 492, 1070], 'sorted_id': 491}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[491] = op;
            
            op->set_inputs( forward_result[406] );
            op->set_inputs( forward_result[463] );
            op->set_inputs( forward_result[490] );
        }
        
        // {'name': 'Model/3200', 'op': 'prim::TupleUnpack', 'in': [491], 'output_id': 0, 'shape': [4, 400], 'out': [493], 'sorted_id': 492}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[492] = op;
            
            op->set_inputs( forward_result[491] );
        }
        
        // {'name': 'Model/Net[net]/input.17', 'op': 'aten::relu', 'in': [492], 'output_id': 0, 'shape': [4, 400], 'out': [525], 'sorted_id': 493}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[493] = op;
            
            op->set_inputs( forward_result[492] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_mu/weight_mu.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [509, 587], 'sorted_id': 494}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[494] = new VariableTensor( l2_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_rho/weight_rho.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [499, 583, 496, 589, 500], 'sorted_id': 495}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[495] = new VariableTensor( l2_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3216', 'op': 'aten::exp', 'in': [495], 'output_id': 0, 'shape': [400, 400], 'out': [497], 'sorted_id': 496}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[496] = op;
            
            op->set_inputs( forward_result[495] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3217', 'op': 'aten::log1p', 'in': [496], 'output_id': 0, 'shape': [400, 400], 'out': [508], 'sorted_id': 497}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[497] = op;
            
            op->set_inputs( forward_result[496] );
        }
        
        // {'name': 'Model/2945', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 3, 'shape': [], 'out': [850, 502], 'sorted_id': 498}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[498] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3208', 'op': 'aten::size', 'in': [495, 10], 'output_id': 0, 'shape': [], 'out': [501, 504], 'sorted_id': 499}
        {
            SizeOp* op = new SizeOp();
            forward_result[499] = op;
            
            op->set_inputs( forward_result[495] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3209', 'op': 'aten::size', 'in': [495, 12], 'output_id': 0, 'shape': [], 'out': [501, 504], 'sorted_id': 500}
        {
            SizeOp* op = new SizeOp();
            forward_result[500] = op;
            
            op->set_inputs( forward_result[495] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3210', 'op': 'prim::ListConstruct', 'in': [499, 500], 'output_id': 0, 'shape': [], 'out': [502], 'sorted_id': 501}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[501] = op;
            
            op->set_inputs( forward_result[499] );
            op->set_inputs( forward_result[500] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3211', 'op': 'aten::expand', 'in': [498, 501, 15], 'output_id': 0, 'shape': [400, 400], 'out': [506], 'sorted_id': 502}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[502] = op;
            
            op->set_inputs( forward_result[498] );
            op->set_inputs( forward_result[501] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/2946', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 4, 'shape': [], 'out': [505, 852], 'sorted_id': 503}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[503] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3212', 'op': 'prim::ListConstruct', 'in': [499, 500], 'output_id': 0, 'shape': [], 'out': [505], 'sorted_id': 504}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[504] = op;
            
            op->set_inputs( forward_result[499] );
            op->set_inputs( forward_result[500] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3213', 'op': 'aten::expand', 'in': [503, 504, 15], 'output_id': 0, 'shape': [400, 400], 'out': [506], 'sorted_id': 505}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[505] = op;
            
            op->set_inputs( forward_result[503] );
            op->set_inputs( forward_result[504] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3214', 'op': 'aten::normal', 'in': [502, 505, 20], 'output_id': 0, 'shape': [400, 400], 'out': [507], 'sorted_id': 506}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[506] = op;
            
            op->set_inputs( forward_result[502] );
            op->set_inputs( forward_result[505] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.17', 'op': 'aten::to', 'in': [506, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 400], 'out': [508], 'sorted_id': 507}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[507] = op;
            
            op->set_inputs( forward_result[506] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3218', 'op': 'aten::mul', 'in': [497, 507], 'output_id': 0, 'shape': [400, 400], 'out': [509], 'sorted_id': 508}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[508] = op;
            
            op->set_inputs( forward_result[497] );
            op->set_inputs( forward_result[507] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.17', 'op': 'aten::add', 'in': [494, 508, 12], 'output_id': 0, 'shape': [400, 400], 'out': [525, 587, 540, 527], 'sorted_id': 509}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[509] = op;
            
            op->set_inputs( forward_result[494] );
            op->set_inputs( forward_result[508] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_mu/bias_mu.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [524, 600], 'sorted_id': 510}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[510] = new VariableTensor( l2_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_rho/bias_rho.9', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [515, 512, 596, 602], 'sorted_id': 511}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[511] = new VariableTensor( l2_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3227', 'op': 'aten::exp', 'in': [511], 'output_id': 0, 'shape': [400], 'out': [513], 'sorted_id': 512}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[512] = op;
            
            op->set_inputs( forward_result[511] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3228', 'op': 'aten::log1p', 'in': [512], 'output_id': 0, 'shape': [400], 'out': [523], 'sorted_id': 513}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[513] = op;
            
            op->set_inputs( forward_result[512] );
        }
        
        // {'name': 'Model/2947', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 5, 'shape': [], 'out': [863, 517], 'sorted_id': 514}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[514] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3220', 'op': 'aten::size', 'in': [511, 10], 'output_id': 0, 'shape': [], 'out': [519, 516], 'sorted_id': 515}
        {
            SizeOp* op = new SizeOp();
            forward_result[515] = op;
            
            op->set_inputs( forward_result[511] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3221', 'op': 'prim::ListConstruct', 'in': [515], 'output_id': 0, 'shape': [], 'out': [517], 'sorted_id': 516}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[516] = op;
            
            op->set_inputs( forward_result[515] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3222', 'op': 'aten::expand', 'in': [514, 516, 15], 'output_id': 0, 'shape': [400], 'out': [521], 'sorted_id': 517}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[517] = op;
            
            op->set_inputs( forward_result[514] );
            op->set_inputs( forward_result[516] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/2948', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 6, 'shape': [], 'out': [865, 520], 'sorted_id': 518}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 6 );
            forward_result[518] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3223', 'op': 'prim::ListConstruct', 'in': [515], 'output_id': 0, 'shape': [], 'out': [520], 'sorted_id': 519}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[519] = op;
            
            op->set_inputs( forward_result[515] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3224', 'op': 'aten::expand', 'in': [518, 519, 15], 'output_id': 0, 'shape': [400], 'out': [521], 'sorted_id': 520}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[520] = op;
            
            op->set_inputs( forward_result[518] );
            op->set_inputs( forward_result[519] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3225', 'op': 'aten::normal', 'in': [517, 520, 20], 'output_id': 0, 'shape': [400], 'out': [522], 'sorted_id': 521}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[521] = op;
            
            op->set_inputs( forward_result[517] );
            op->set_inputs( forward_result[520] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.19', 'op': 'aten::to', 'in': [521, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [523], 'sorted_id': 522}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[522] = op;
            
            op->set_inputs( forward_result[521] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3229', 'op': 'aten::mul', 'in': [513, 522], 'output_id': 0, 'shape': [400], 'out': [524], 'sorted_id': 523}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[523] = op;
            
            op->set_inputs( forward_result[513] );
            op->set_inputs( forward_result[522] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.19', 'op': 'aten::add', 'in': [510, 523, 12], 'output_id': 0, 'shape': [400], 'out': [525, 568, 556, 600], 'sorted_id': 524}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[524] = op;
            
            op->set_inputs( forward_result[510] );
            op->set_inputs( forward_result[523] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/input.19', 'op': 'aten::linear', 'in': [493, 509, 524], 'output_id': 0, 'shape': [4, 400], 'out': [610], 'sorted_id': 525}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[525] = op;
            
            op->set_inputs( forward_result[493] );
            op->set_inputs( forward_result[509] );
            op->set_inputs( forward_result[524] );
        }
        
        // {'name': 'Model/2950', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 8, 'shape': [1], 'out': [871, 527], 'sorted_id': 526}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 8 );
            forward_result[526] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3233', 'op': 'aten::sub', 'in': [509, 526, 12], 'output_id': 0, 'shape': [400, 400], 'out': [528], 'sorted_id': 527}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[527] = op;
            
            op->set_inputs( forward_result[509] );
            op->set_inputs( forward_result[526] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3234', 'op': 'aten::pow', 'in': [527, 45], 'output_id': 0, 'shape': [400, 400], 'out': [529], 'sorted_id': 528}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[528] = op;
            
            op->set_inputs( forward_result[527] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3235', 'op': 'aten::neg', 'in': [528], 'output_id': 0, 'shape': [400, 400], 'out': [533], 'sorted_id': 529}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[529] = op;
            
            op->set_inputs( forward_result[528] );
        }
        
        // {'name': 'Model/2949', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 7, 'shape': [1], 'out': [899, 877, 559, 902, 562, 534, 531, 874], 'sorted_id': 530}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 7 );
            forward_result[530] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.33', 'op': 'aten::pow', 'in': [530, 45], 'output_id': 0, 'shape': [1], 'out': [532], 'sorted_id': 531}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[531] = op;
            
            op->set_inputs( forward_result[530] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3236', 'op': 'aten::mul', 'in': [531, 50], 'output_id': 0, 'shape': [1], 'out': [533], 'sorted_id': 532}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[532] = op;
            
            op->set_inputs( forward_result[531] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3237', 'op': 'aten::div', 'in': [529, 532], 'output_id': 0, 'shape': [400, 400], 'out': [535], 'sorted_id': 533}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[533] = op;
            
            op->set_inputs( forward_result[529] );
            op->set_inputs( forward_result[532] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.33', 'op': 'aten::log', 'in': [530], 'output_id': 0, 'shape': [1], 'out': [535], 'sorted_id': 534}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[534] = op;
            
            op->set_inputs( forward_result[530] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3238', 'op': 'aten::sub', 'in': [533, 534, 12], 'output_id': 0, 'shape': [400, 400], 'out': [536], 'sorted_id': 535}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[535] = op;
            
            op->set_inputs( forward_result[533] );
            op->set_inputs( forward_result[534] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3239', 'op': 'aten::sub', 'in': [535, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [537], 'sorted_id': 536}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[536] = op;
            
            op->set_inputs( forward_result[535] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.17', 'op': 'aten::exp', 'in': [536], 'output_id': 0, 'shape': [400, 400], 'out': [538], 'sorted_id': 537}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[537] = op;
            
            op->set_inputs( forward_result[536] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3251', 'op': 'aten::mul', 'in': [537, 58], 'output_id': 0, 'shape': [400, 400], 'out': [552], 'sorted_id': 538}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[538] = op;
            
            op->set_inputs( forward_result[537] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/2952', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 10, 'shape': [1], 'out': [540, 882], 'sorted_id': 539}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 10 );
            forward_result[539] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3243', 'op': 'aten::sub', 'in': [509, 539, 12], 'output_id': 0, 'shape': [400, 400], 'out': [541], 'sorted_id': 540}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[540] = op;
            
            op->set_inputs( forward_result[509] );
            op->set_inputs( forward_result[539] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3244', 'op': 'aten::pow', 'in': [540, 45], 'output_id': 0, 'shape': [400, 400], 'out': [542], 'sorted_id': 541}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[541] = op;
            
            op->set_inputs( forward_result[540] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3245', 'op': 'aten::neg', 'in': [541], 'output_id': 0, 'shape': [400, 400], 'out': [546], 'sorted_id': 542}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[542] = op;
            
            op->set_inputs( forward_result[541] );
        }
        
        // {'name': 'Model/2951', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 9, 'shape': [1], 'out': [913, 544, 574, 885, 888, 910, 571, 547], 'sorted_id': 543}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 9 );
            forward_result[543] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.35', 'op': 'aten::pow', 'in': [543, 45], 'output_id': 0, 'shape': [1], 'out': [545], 'sorted_id': 544}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[544] = op;
            
            op->set_inputs( forward_result[543] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3246', 'op': 'aten::mul', 'in': [544, 50], 'output_id': 0, 'shape': [1], 'out': [546], 'sorted_id': 545}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[545] = op;
            
            op->set_inputs( forward_result[544] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3247', 'op': 'aten::div', 'in': [542, 545], 'output_id': 0, 'shape': [400, 400], 'out': [548], 'sorted_id': 546}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[546] = op;
            
            op->set_inputs( forward_result[542] );
            op->set_inputs( forward_result[545] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.35', 'op': 'aten::log', 'in': [543], 'output_id': 0, 'shape': [1], 'out': [548], 'sorted_id': 547}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[547] = op;
            
            op->set_inputs( forward_result[543] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3248', 'op': 'aten::sub', 'in': [546, 547, 12], 'output_id': 0, 'shape': [400, 400], 'out': [549], 'sorted_id': 548}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[548] = op;
            
            op->set_inputs( forward_result[546] );
            op->set_inputs( forward_result[547] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3249', 'op': 'aten::sub', 'in': [548, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [550], 'sorted_id': 549}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[549] = op;
            
            op->set_inputs( forward_result[548] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.17', 'op': 'aten::exp', 'in': [549], 'output_id': 0, 'shape': [400, 400], 'out': [551], 'sorted_id': 550}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[550] = op;
            
            op->set_inputs( forward_result[549] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3252', 'op': 'aten::mul', 'in': [550, 58], 'output_id': 0, 'shape': [400, 400], 'out': [552], 'sorted_id': 551}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[551] = op;
            
            op->set_inputs( forward_result[550] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3253', 'op': 'aten::add', 'in': [538, 551, 12], 'output_id': 0, 'shape': [400, 400], 'out': [553], 'sorted_id': 552}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[552] = op;
            
            op->set_inputs( forward_result[538] );
            op->set_inputs( forward_result[551] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3254', 'op': 'aten::log', 'in': [552], 'output_id': 0, 'shape': [400, 400], 'out': [554], 'sorted_id': 553}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[553] = op;
            
            op->set_inputs( forward_result[552] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3255', 'op': 'aten::sum', 'in': [553, 20], 'output_id': 0, 'shape': [], 'out': [582], 'sorted_id': 554}
        {
            SumOp* op = new SumOp();
            forward_result[554] = op;
            
            op->set_inputs( forward_result[553] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/2953', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 11, 'shape': [1], 'out': [556, 896], 'sorted_id': 555}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 11 );
            forward_result[555] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3258', 'op': 'aten::sub', 'in': [524, 555, 12], 'output_id': 0, 'shape': [400], 'out': [557], 'sorted_id': 556}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[556] = op;
            
            op->set_inputs( forward_result[524] );
            op->set_inputs( forward_result[555] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3259', 'op': 'aten::pow', 'in': [556, 45], 'output_id': 0, 'shape': [400], 'out': [558], 'sorted_id': 557}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[557] = op;
            
            op->set_inputs( forward_result[556] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3260', 'op': 'aten::neg', 'in': [557], 'output_id': 0, 'shape': [400], 'out': [561], 'sorted_id': 558}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[558] = op;
            
            op->set_inputs( forward_result[557] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.37', 'op': 'aten::pow', 'in': [530, 45], 'output_id': 0, 'shape': [1], 'out': [560], 'sorted_id': 559}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[559] = op;
            
            op->set_inputs( forward_result[530] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3261', 'op': 'aten::mul', 'in': [559, 50], 'output_id': 0, 'shape': [1], 'out': [561], 'sorted_id': 560}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[560] = op;
            
            op->set_inputs( forward_result[559] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3262', 'op': 'aten::div', 'in': [558, 560], 'output_id': 0, 'shape': [400], 'out': [563], 'sorted_id': 561}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[561] = op;
            
            op->set_inputs( forward_result[558] );
            op->set_inputs( forward_result[560] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.37', 'op': 'aten::log', 'in': [530], 'output_id': 0, 'shape': [1], 'out': [563], 'sorted_id': 562}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[562] = op;
            
            op->set_inputs( forward_result[530] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3263', 'op': 'aten::sub', 'in': [561, 562, 12], 'output_id': 0, 'shape': [400], 'out': [564], 'sorted_id': 563}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[563] = op;
            
            op->set_inputs( forward_result[561] );
            op->set_inputs( forward_result[562] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3264', 'op': 'aten::sub', 'in': [563, 55, 12], 'output_id': 0, 'shape': [400], 'out': [565], 'sorted_id': 564}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[564] = op;
            
            op->set_inputs( forward_result[563] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.19', 'op': 'aten::exp', 'in': [564], 'output_id': 0, 'shape': [400], 'out': [566], 'sorted_id': 565}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[565] = op;
            
            op->set_inputs( forward_result[564] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3276', 'op': 'aten::mul', 'in': [565, 58], 'output_id': 0, 'shape': [400], 'out': [579], 'sorted_id': 566}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[566] = op;
            
            op->set_inputs( forward_result[565] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/2954', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 12, 'shape': [1], 'out': [907, 568], 'sorted_id': 567}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 12 );
            forward_result[567] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3268', 'op': 'aten::sub', 'in': [524, 567, 12], 'output_id': 0, 'shape': [400], 'out': [569], 'sorted_id': 568}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[568] = op;
            
            op->set_inputs( forward_result[524] );
            op->set_inputs( forward_result[567] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3269', 'op': 'aten::pow', 'in': [568, 45], 'output_id': 0, 'shape': [400], 'out': [570], 'sorted_id': 569}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[569] = op;
            
            op->set_inputs( forward_result[568] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3270', 'op': 'aten::neg', 'in': [569], 'output_id': 0, 'shape': [400], 'out': [573], 'sorted_id': 570}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[570] = op;
            
            op->set_inputs( forward_result[569] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.39', 'op': 'aten::pow', 'in': [543, 45], 'output_id': 0, 'shape': [1], 'out': [572], 'sorted_id': 571}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[571] = op;
            
            op->set_inputs( forward_result[543] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3271', 'op': 'aten::mul', 'in': [571, 50], 'output_id': 0, 'shape': [1], 'out': [573], 'sorted_id': 572}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[572] = op;
            
            op->set_inputs( forward_result[571] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3272', 'op': 'aten::div', 'in': [570, 572], 'output_id': 0, 'shape': [400], 'out': [575], 'sorted_id': 573}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[573] = op;
            
            op->set_inputs( forward_result[570] );
            op->set_inputs( forward_result[572] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.39', 'op': 'aten::log', 'in': [543], 'output_id': 0, 'shape': [1], 'out': [575], 'sorted_id': 574}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[574] = op;
            
            op->set_inputs( forward_result[543] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3273', 'op': 'aten::sub', 'in': [573, 574, 12], 'output_id': 0, 'shape': [400], 'out': [576], 'sorted_id': 575}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[575] = op;
            
            op->set_inputs( forward_result[573] );
            op->set_inputs( forward_result[574] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3274', 'op': 'aten::sub', 'in': [575, 55, 12], 'output_id': 0, 'shape': [400], 'out': [577], 'sorted_id': 576}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[576] = op;
            
            op->set_inputs( forward_result[575] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.19', 'op': 'aten::exp', 'in': [576], 'output_id': 0, 'shape': [400], 'out': [578], 'sorted_id': 577}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[577] = op;
            
            op->set_inputs( forward_result[576] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3277', 'op': 'aten::mul', 'in': [577, 58], 'output_id': 0, 'shape': [400], 'out': [579], 'sorted_id': 578}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[578] = op;
            
            op->set_inputs( forward_result[577] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3278', 'op': 'aten::add', 'in': [566, 578, 12], 'output_id': 0, 'shape': [400], 'out': [580], 'sorted_id': 579}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[579] = op;
            
            op->set_inputs( forward_result[566] );
            op->set_inputs( forward_result[578] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3279', 'op': 'aten::log', 'in': [579], 'output_id': 0, 'shape': [400], 'out': [581], 'sorted_id': 580}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[580] = op;
            
            op->set_inputs( forward_result[579] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3280', 'op': 'aten::sum', 'in': [580, 20], 'output_id': 0, 'shape': [], 'out': [582], 'sorted_id': 581}
        {
            SumOp* op = new SumOp();
            forward_result[581] = op;
            
            op->set_inputs( forward_result[580] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3281', 'op': 'aten::add', 'in': [554, 581, 12], 'output_id': 0, 'shape': [], 'out': [610], 'sorted_id': 582}
        {
            AddOp* op = new AddOp();
            forward_result[582] = op;
            
            op->set_inputs( forward_result[554] );
            op->set_inputs( forward_result[581] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3282', 'op': 'aten::exp', 'in': [495], 'output_id': 0, 'shape': [400, 400], 'out': [584], 'sorted_id': 583}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[583] = op;
            
            op->set_inputs( forward_result[495] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3283', 'op': 'aten::log1p', 'in': [583], 'output_id': 0, 'shape': [400, 400], 'out': [585], 'sorted_id': 584}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[584] = op;
            
            op->set_inputs( forward_result[583] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3284', 'op': 'aten::log', 'in': [584], 'output_id': 0, 'shape': [400, 400], 'out': [586], 'sorted_id': 585}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[585] = op;
            
            op->set_inputs( forward_result[584] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3285', 'op': 'aten::rsub', 'in': [585, 107, 12], 'output_id': 0, 'shape': [400, 400], 'out': [594], 'sorted_id': 586}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[586] = op;
            
            op->set_inputs( forward_result[585] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3286', 'op': 'aten::sub', 'in': [509, 494, 12], 'output_id': 0, 'shape': [400, 400], 'out': [588], 'sorted_id': 587}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[587] = op;
            
            op->set_inputs( forward_result[509] );
            op->set_inputs( forward_result[494] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3287', 'op': 'aten::pow', 'in': [587, 45], 'output_id': 0, 'shape': [400, 400], 'out': [593], 'sorted_id': 588}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[588] = op;
            
            op->set_inputs( forward_result[587] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3288', 'op': 'aten::exp', 'in': [495], 'output_id': 0, 'shape': [400, 400], 'out': [590], 'sorted_id': 589}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[589] = op;
            
            op->set_inputs( forward_result[495] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3289', 'op': 'aten::log1p', 'in': [589], 'output_id': 0, 'shape': [400, 400], 'out': [591], 'sorted_id': 590}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[590] = op;
            
            op->set_inputs( forward_result[589] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3290', 'op': 'aten::pow', 'in': [590, 45], 'output_id': 0, 'shape': [400, 400], 'out': [592], 'sorted_id': 591}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[591] = op;
            
            op->set_inputs( forward_result[590] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3291', 'op': 'aten::mul', 'in': [591, 50], 'output_id': 0, 'shape': [400, 400], 'out': [593], 'sorted_id': 592}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[592] = op;
            
            op->set_inputs( forward_result[591] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3292', 'op': 'aten::div', 'in': [588, 592], 'output_id': 0, 'shape': [400, 400], 'out': [594], 'sorted_id': 593}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[593] = op;
            
            op->set_inputs( forward_result[588] );
            op->set_inputs( forward_result[592] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3293', 'op': 'aten::sub', 'in': [586, 593, 12], 'output_id': 0, 'shape': [400, 400], 'out': [595], 'sorted_id': 594}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[594] = op;
            
            op->set_inputs( forward_result[586] );
            op->set_inputs( forward_result[593] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3294', 'op': 'aten::sum', 'in': [594, 20], 'output_id': 0, 'shape': [], 'out': [609], 'sorted_id': 595}
        {
            SumOp* op = new SumOp();
            forward_result[595] = op;
            
            op->set_inputs( forward_result[594] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3295', 'op': 'aten::exp', 'in': [511], 'output_id': 0, 'shape': [400], 'out': [597], 'sorted_id': 596}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[596] = op;
            
            op->set_inputs( forward_result[511] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3296', 'op': 'aten::log1p', 'in': [596], 'output_id': 0, 'shape': [400], 'out': [598], 'sorted_id': 597}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[597] = op;
            
            op->set_inputs( forward_result[596] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3297', 'op': 'aten::log', 'in': [597], 'output_id': 0, 'shape': [400], 'out': [599], 'sorted_id': 598}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[598] = op;
            
            op->set_inputs( forward_result[597] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3298', 'op': 'aten::rsub', 'in': [598, 107, 12], 'output_id': 0, 'shape': [400], 'out': [607], 'sorted_id': 599}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[599] = op;
            
            op->set_inputs( forward_result[598] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3299', 'op': 'aten::sub', 'in': [524, 510, 12], 'output_id': 0, 'shape': [400], 'out': [601], 'sorted_id': 600}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[600] = op;
            
            op->set_inputs( forward_result[524] );
            op->set_inputs( forward_result[510] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3300', 'op': 'aten::pow', 'in': [600, 45], 'output_id': 0, 'shape': [400], 'out': [606], 'sorted_id': 601}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[601] = op;
            
            op->set_inputs( forward_result[600] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3301', 'op': 'aten::exp', 'in': [511], 'output_id': 0, 'shape': [400], 'out': [603], 'sorted_id': 602}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[602] = op;
            
            op->set_inputs( forward_result[511] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3302', 'op': 'aten::log1p', 'in': [602], 'output_id': 0, 'shape': [400], 'out': [604], 'sorted_id': 603}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[603] = op;
            
            op->set_inputs( forward_result[602] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3303', 'op': 'aten::pow', 'in': [603, 45], 'output_id': 0, 'shape': [400], 'out': [605], 'sorted_id': 604}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[604] = op;
            
            op->set_inputs( forward_result[603] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3304', 'op': 'aten::mul', 'in': [604, 50], 'output_id': 0, 'shape': [400], 'out': [606], 'sorted_id': 605}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[605] = op;
            
            op->set_inputs( forward_result[604] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3305', 'op': 'aten::div', 'in': [601, 605], 'output_id': 0, 'shape': [400], 'out': [607], 'sorted_id': 606}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[606] = op;
            
            op->set_inputs( forward_result[601] );
            op->set_inputs( forward_result[605] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3306', 'op': 'aten::sub', 'in': [599, 606, 12], 'output_id': 0, 'shape': [400], 'out': [608], 'sorted_id': 607}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[607] = op;
            
            op->set_inputs( forward_result[599] );
            op->set_inputs( forward_result[606] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3307', 'op': 'aten::sum', 'in': [607, 20], 'output_id': 0, 'shape': [], 'out': [609], 'sorted_id': 608}
        {
            SumOp* op = new SumOp();
            forward_result[608] = op;
            
            op->set_inputs( forward_result[607] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3308', 'op': 'aten::add', 'in': [595, 608, 12], 'output_id': 0, 'shape': [], 'out': [610], 'sorted_id': 609}
        {
            AddOp* op = new AddOp();
            forward_result[609] = op;
            
            op->set_inputs( forward_result[595] );
            op->set_inputs( forward_result[608] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3310', 'op': 'prim::TupleConstruct', 'in': [525, 582, 609], 'output_id': 0, 'shape': [], 'out': [1071, 1089, 611], 'sorted_id': 610}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[610] = op;
            
            op->set_inputs( forward_result[525] );
            op->set_inputs( forward_result[582] );
            op->set_inputs( forward_result[609] );
        }
        
        // {'name': 'Model/3311', 'op': 'prim::TupleUnpack', 'in': [610], 'output_id': 0, 'shape': [4, 400], 'out': [612], 'sorted_id': 611}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[611] = op;
            
            op->set_inputs( forward_result[610] );
        }
        
        // {'name': 'Model/Net[net]/input.21', 'op': 'aten::relu', 'in': [611], 'output_id': 0, 'shape': [4, 400], 'out': [644], 'sorted_id': 612}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[612] = op;
            
            op->set_inputs( forward_result[611] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_mu/weight_mu.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [706, 628], 'sorted_id': 613}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[613] = new VariableTensor( l3_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_rho/weight_rho.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [708, 702, 618, 619, 615], 'sorted_id': 614}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[614] = new VariableTensor( l3_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3327', 'op': 'aten::exp', 'in': [614], 'output_id': 0, 'shape': [10, 400], 'out': [616], 'sorted_id': 615}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[615] = op;
            
            op->set_inputs( forward_result[614] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3328', 'op': 'aten::log1p', 'in': [615], 'output_id': 0, 'shape': [10, 400], 'out': [627], 'sorted_id': 616}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[616] = op;
            
            op->set_inputs( forward_result[615] );
        }
        
        // {'name': 'Model/3076', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 3, 'shape': [], 'out': [959, 621], 'sorted_id': 617}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[617] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3319', 'op': 'aten::size', 'in': [614, 10], 'output_id': 0, 'shape': [], 'out': [620, 623], 'sorted_id': 618}
        {
            SizeOp* op = new SizeOp();
            forward_result[618] = op;
            
            op->set_inputs( forward_result[614] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3320', 'op': 'aten::size', 'in': [614, 12], 'output_id': 0, 'shape': [], 'out': [620, 623], 'sorted_id': 619}
        {
            SizeOp* op = new SizeOp();
            forward_result[619] = op;
            
            op->set_inputs( forward_result[614] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3321', 'op': 'prim::ListConstruct', 'in': [618, 619], 'output_id': 0, 'shape': [], 'out': [621], 'sorted_id': 620}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[620] = op;
            
            op->set_inputs( forward_result[618] );
            op->set_inputs( forward_result[619] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3322', 'op': 'aten::expand', 'in': [617, 620, 15], 'output_id': 0, 'shape': [10, 400], 'out': [625], 'sorted_id': 621}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[621] = op;
            
            op->set_inputs( forward_result[617] );
            op->set_inputs( forward_result[620] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/3077', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 4, 'shape': [], 'out': [624, 961], 'sorted_id': 622}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[622] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3323', 'op': 'prim::ListConstruct', 'in': [618, 619], 'output_id': 0, 'shape': [], 'out': [624], 'sorted_id': 623}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[623] = op;
            
            op->set_inputs( forward_result[618] );
            op->set_inputs( forward_result[619] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3324', 'op': 'aten::expand', 'in': [622, 623, 15], 'output_id': 0, 'shape': [10, 400], 'out': [625], 'sorted_id': 624}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[624] = op;
            
            op->set_inputs( forward_result[622] );
            op->set_inputs( forward_result[623] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3325', 'op': 'aten::normal', 'in': [621, 624, 20], 'output_id': 0, 'shape': [10, 400], 'out': [626], 'sorted_id': 625}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[625] = op;
            
            op->set_inputs( forward_result[621] );
            op->set_inputs( forward_result[624] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.21', 'op': 'aten::to', 'in': [625, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10, 400], 'out': [627], 'sorted_id': 626}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[626] = op;
            
            op->set_inputs( forward_result[625] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3329', 'op': 'aten::mul', 'in': [616, 626], 'output_id': 0, 'shape': [10, 400], 'out': [628], 'sorted_id': 627}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[627] = op;
            
            op->set_inputs( forward_result[616] );
            op->set_inputs( forward_result[626] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.21', 'op': 'aten::add', 'in': [613, 627, 12], 'output_id': 0, 'shape': [10, 400], 'out': [706, 659, 646, 644], 'sorted_id': 628}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[628] = op;
            
            op->set_inputs( forward_result[613] );
            op->set_inputs( forward_result[627] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_mu/bias_mu.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [719, 643], 'sorted_id': 629}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[629] = new VariableTensor( l3_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_rho/bias_rho.11', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [634, 715, 631, 721], 'sorted_id': 630}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[630] = new VariableTensor( l3_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3338', 'op': 'aten::exp', 'in': [630], 'output_id': 0, 'shape': [10], 'out': [632], 'sorted_id': 631}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[631] = op;
            
            op->set_inputs( forward_result[630] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3339', 'op': 'aten::log1p', 'in': [631], 'output_id': 0, 'shape': [10], 'out': [642], 'sorted_id': 632}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[632] = op;
            
            op->set_inputs( forward_result[631] );
        }
        
        // {'name': 'Model/3078', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 5, 'shape': [], 'out': [636, 972], 'sorted_id': 633}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[633] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3331', 'op': 'aten::size', 'in': [630, 10], 'output_id': 0, 'shape': [], 'out': [638, 635], 'sorted_id': 634}
        {
            SizeOp* op = new SizeOp();
            forward_result[634] = op;
            
            op->set_inputs( forward_result[630] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3332', 'op': 'prim::ListConstruct', 'in': [634], 'output_id': 0, 'shape': [], 'out': [636], 'sorted_id': 635}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[635] = op;
            
            op->set_inputs( forward_result[634] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3333', 'op': 'aten::expand', 'in': [633, 635, 15], 'output_id': 0, 'shape': [10], 'out': [640], 'sorted_id': 636}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[636] = op;
            
            op->set_inputs( forward_result[633] );
            op->set_inputs( forward_result[635] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/3079', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 6, 'shape': [], 'out': [974, 639], 'sorted_id': 637}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 6 );
            forward_result[637] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3334', 'op': 'prim::ListConstruct', 'in': [634], 'output_id': 0, 'shape': [], 'out': [639], 'sorted_id': 638}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[638] = op;
            
            op->set_inputs( forward_result[634] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3335', 'op': 'aten::expand', 'in': [637, 638, 15], 'output_id': 0, 'shape': [10], 'out': [640], 'sorted_id': 639}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[639] = op;
            
            op->set_inputs( forward_result[637] );
            op->set_inputs( forward_result[638] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3336', 'op': 'aten::normal', 'in': [636, 639, 20], 'output_id': 0, 'shape': [10], 'out': [641], 'sorted_id': 640}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[640] = op;
            
            op->set_inputs( forward_result[636] );
            op->set_inputs( forward_result[639] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.23', 'op': 'aten::to', 'in': [640, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10], 'out': [642], 'sorted_id': 641}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[641] = op;
            
            op->set_inputs( forward_result[640] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3340', 'op': 'aten::mul', 'in': [632, 641], 'output_id': 0, 'shape': [10], 'out': [643], 'sorted_id': 642}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[642] = op;
            
            op->set_inputs( forward_result[632] );
            op->set_inputs( forward_result[641] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.23', 'op': 'aten::add', 'in': [629, 642, 12], 'output_id': 0, 'shape': [10], 'out': [644, 719, 687, 675], 'sorted_id': 643}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[643] = op;
            
            op->set_inputs( forward_result[629] );
            op->set_inputs( forward_result[642] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/input.23', 'op': 'aten::linear', 'in': [612, 628, 643], 'output_id': 0, 'shape': [4, 10], 'out': [729], 'sorted_id': 644}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[644] = op;
            
            op->set_inputs( forward_result[612] );
            op->set_inputs( forward_result[628] );
            op->set_inputs( forward_result[643] );
        }
        
        // {'name': 'Model/3081', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 8, 'shape': [1], 'out': [980, 646], 'sorted_id': 645}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 8 );
            forward_result[645] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3344', 'op': 'aten::sub', 'in': [628, 645, 12], 'output_id': 0, 'shape': [10, 400], 'out': [647], 'sorted_id': 646}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[646] = op;
            
            op->set_inputs( forward_result[628] );
            op->set_inputs( forward_result[645] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3345', 'op': 'aten::pow', 'in': [646, 45], 'output_id': 0, 'shape': [10, 400], 'out': [648], 'sorted_id': 647}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[647] = op;
            
            op->set_inputs( forward_result[646] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3346', 'op': 'aten::neg', 'in': [647], 'output_id': 0, 'shape': [10, 400], 'out': [652], 'sorted_id': 648}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[648] = op;
            
            op->set_inputs( forward_result[647] );
        }
        
        // {'name': 'Model/3080', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 7, 'shape': [1], 'out': [983, 678, 986, 653, 650, 681, 1008, 1011], 'sorted_id': 649}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 7 );
            forward_result[649] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.41', 'op': 'aten::pow', 'in': [649, 45], 'output_id': 0, 'shape': [1], 'out': [651], 'sorted_id': 650}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[650] = op;
            
            op->set_inputs( forward_result[649] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3347', 'op': 'aten::mul', 'in': [650, 50], 'output_id': 0, 'shape': [1], 'out': [652], 'sorted_id': 651}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[651] = op;
            
            op->set_inputs( forward_result[650] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3348', 'op': 'aten::div', 'in': [648, 651], 'output_id': 0, 'shape': [10, 400], 'out': [654], 'sorted_id': 652}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[652] = op;
            
            op->set_inputs( forward_result[648] );
            op->set_inputs( forward_result[651] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.41', 'op': 'aten::log', 'in': [649], 'output_id': 0, 'shape': [1], 'out': [654], 'sorted_id': 653}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[653] = op;
            
            op->set_inputs( forward_result[649] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3349', 'op': 'aten::sub', 'in': [652, 653, 12], 'output_id': 0, 'shape': [10, 400], 'out': [655], 'sorted_id': 654}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[654] = op;
            
            op->set_inputs( forward_result[652] );
            op->set_inputs( forward_result[653] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3350', 'op': 'aten::sub', 'in': [654, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [656], 'sorted_id': 655}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[655] = op;
            
            op->set_inputs( forward_result[654] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.21', 'op': 'aten::exp', 'in': [655], 'output_id': 0, 'shape': [10, 400], 'out': [657], 'sorted_id': 656}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[656] = op;
            
            op->set_inputs( forward_result[655] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3362', 'op': 'aten::mul', 'in': [656, 58], 'output_id': 0, 'shape': [10, 400], 'out': [671], 'sorted_id': 657}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[657] = op;
            
            op->set_inputs( forward_result[656] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/3083', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 10, 'shape': [1], 'out': [991, 659], 'sorted_id': 658}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 10 );
            forward_result[658] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3354', 'op': 'aten::sub', 'in': [628, 658, 12], 'output_id': 0, 'shape': [10, 400], 'out': [660], 'sorted_id': 659}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[659] = op;
            
            op->set_inputs( forward_result[628] );
            op->set_inputs( forward_result[658] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3355', 'op': 'aten::pow', 'in': [659, 45], 'output_id': 0, 'shape': [10, 400], 'out': [661], 'sorted_id': 660}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[660] = op;
            
            op->set_inputs( forward_result[659] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3356', 'op': 'aten::neg', 'in': [660], 'output_id': 0, 'shape': [10, 400], 'out': [665], 'sorted_id': 661}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[661] = op;
            
            op->set_inputs( forward_result[660] );
        }
        
        // {'name': 'Model/3082', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 9, 'shape': [1], 'out': [666, 693, 1019, 663, 1022, 994, 997, 690], 'sorted_id': 662}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 9 );
            forward_result[662] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.43', 'op': 'aten::pow', 'in': [662, 45], 'output_id': 0, 'shape': [1], 'out': [664], 'sorted_id': 663}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[663] = op;
            
            op->set_inputs( forward_result[662] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3357', 'op': 'aten::mul', 'in': [663, 50], 'output_id': 0, 'shape': [1], 'out': [665], 'sorted_id': 664}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[664] = op;
            
            op->set_inputs( forward_result[663] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3358', 'op': 'aten::div', 'in': [661, 664], 'output_id': 0, 'shape': [10, 400], 'out': [667], 'sorted_id': 665}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[665] = op;
            
            op->set_inputs( forward_result[661] );
            op->set_inputs( forward_result[664] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.43', 'op': 'aten::log', 'in': [662], 'output_id': 0, 'shape': [1], 'out': [667], 'sorted_id': 666}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[666] = op;
            
            op->set_inputs( forward_result[662] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3359', 'op': 'aten::sub', 'in': [665, 666, 12], 'output_id': 0, 'shape': [10, 400], 'out': [668], 'sorted_id': 667}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[667] = op;
            
            op->set_inputs( forward_result[665] );
            op->set_inputs( forward_result[666] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3360', 'op': 'aten::sub', 'in': [667, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [669], 'sorted_id': 668}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[668] = op;
            
            op->set_inputs( forward_result[667] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.21', 'op': 'aten::exp', 'in': [668], 'output_id': 0, 'shape': [10, 400], 'out': [670], 'sorted_id': 669}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[669] = op;
            
            op->set_inputs( forward_result[668] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3363', 'op': 'aten::mul', 'in': [669, 58], 'output_id': 0, 'shape': [10, 400], 'out': [671], 'sorted_id': 670}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[670] = op;
            
            op->set_inputs( forward_result[669] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3364', 'op': 'aten::add', 'in': [657, 670, 12], 'output_id': 0, 'shape': [10, 400], 'out': [672], 'sorted_id': 671}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[671] = op;
            
            op->set_inputs( forward_result[657] );
            op->set_inputs( forward_result[670] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3365', 'op': 'aten::log', 'in': [671], 'output_id': 0, 'shape': [10, 400], 'out': [673], 'sorted_id': 672}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[672] = op;
            
            op->set_inputs( forward_result[671] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3366', 'op': 'aten::sum', 'in': [672, 20], 'output_id': 0, 'shape': [], 'out': [701], 'sorted_id': 673}
        {
            SumOp* op = new SumOp();
            forward_result[673] = op;
            
            op->set_inputs( forward_result[672] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/3084', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 11, 'shape': [1], 'out': [1005, 675], 'sorted_id': 674}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 11 );
            forward_result[674] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3369', 'op': 'aten::sub', 'in': [643, 674, 12], 'output_id': 0, 'shape': [10], 'out': [676], 'sorted_id': 675}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[675] = op;
            
            op->set_inputs( forward_result[643] );
            op->set_inputs( forward_result[674] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3370', 'op': 'aten::pow', 'in': [675, 45], 'output_id': 0, 'shape': [10], 'out': [677], 'sorted_id': 676}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[676] = op;
            
            op->set_inputs( forward_result[675] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3371', 'op': 'aten::neg', 'in': [676], 'output_id': 0, 'shape': [10], 'out': [680], 'sorted_id': 677}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[677] = op;
            
            op->set_inputs( forward_result[676] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.45', 'op': 'aten::pow', 'in': [649, 45], 'output_id': 0, 'shape': [1], 'out': [679], 'sorted_id': 678}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[678] = op;
            
            op->set_inputs( forward_result[649] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3372', 'op': 'aten::mul', 'in': [678, 50], 'output_id': 0, 'shape': [1], 'out': [680], 'sorted_id': 679}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[679] = op;
            
            op->set_inputs( forward_result[678] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3373', 'op': 'aten::div', 'in': [677, 679], 'output_id': 0, 'shape': [10], 'out': [682], 'sorted_id': 680}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[680] = op;
            
            op->set_inputs( forward_result[677] );
            op->set_inputs( forward_result[679] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.45', 'op': 'aten::log', 'in': [649], 'output_id': 0, 'shape': [1], 'out': [682], 'sorted_id': 681}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[681] = op;
            
            op->set_inputs( forward_result[649] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3374', 'op': 'aten::sub', 'in': [680, 681, 12], 'output_id': 0, 'shape': [10], 'out': [683], 'sorted_id': 682}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[682] = op;
            
            op->set_inputs( forward_result[680] );
            op->set_inputs( forward_result[681] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3375', 'op': 'aten::sub', 'in': [682, 55, 12], 'output_id': 0, 'shape': [10], 'out': [684], 'sorted_id': 683}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[683] = op;
            
            op->set_inputs( forward_result[682] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.23', 'op': 'aten::exp', 'in': [683], 'output_id': 0, 'shape': [10], 'out': [685], 'sorted_id': 684}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[684] = op;
            
            op->set_inputs( forward_result[683] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3387', 'op': 'aten::mul', 'in': [684, 58], 'output_id': 0, 'shape': [10], 'out': [698], 'sorted_id': 685}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[685] = op;
            
            op->set_inputs( forward_result[684] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/3085', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 12, 'shape': [1], 'out': [1016, 687], 'sorted_id': 686}
        {
            Tensor::shape_type shape = {1};
            TupleUnpackOp* op = new TupleUnpackOp( 12 );
            forward_result[686] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3379', 'op': 'aten::sub', 'in': [643, 686, 12], 'output_id': 0, 'shape': [10], 'out': [688], 'sorted_id': 687}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[687] = op;
            
            op->set_inputs( forward_result[643] );
            op->set_inputs( forward_result[686] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3380', 'op': 'aten::pow', 'in': [687, 45], 'output_id': 0, 'shape': [10], 'out': [689], 'sorted_id': 688}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[688] = op;
            
            op->set_inputs( forward_result[687] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3381', 'op': 'aten::neg', 'in': [688], 'output_id': 0, 'shape': [10], 'out': [692], 'sorted_id': 689}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[689] = op;
            
            op->set_inputs( forward_result[688] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.47', 'op': 'aten::pow', 'in': [662, 45], 'output_id': 0, 'shape': [1], 'out': [691], 'sorted_id': 690}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[690] = op;
            
            op->set_inputs( forward_result[662] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3382', 'op': 'aten::mul', 'in': [690, 50], 'output_id': 0, 'shape': [1], 'out': [692], 'sorted_id': 691}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[691] = op;
            
            op->set_inputs( forward_result[690] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3383', 'op': 'aten::div', 'in': [689, 691], 'output_id': 0, 'shape': [10], 'out': [694], 'sorted_id': 692}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[692] = op;
            
            op->set_inputs( forward_result[689] );
            op->set_inputs( forward_result[691] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.47', 'op': 'aten::log', 'in': [662], 'output_id': 0, 'shape': [1], 'out': [694], 'sorted_id': 693}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[693] = op;
            
            op->set_inputs( forward_result[662] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3384', 'op': 'aten::sub', 'in': [692, 693, 12], 'output_id': 0, 'shape': [10], 'out': [695], 'sorted_id': 694}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[694] = op;
            
            op->set_inputs( forward_result[692] );
            op->set_inputs( forward_result[693] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3385', 'op': 'aten::sub', 'in': [694, 55, 12], 'output_id': 0, 'shape': [10], 'out': [696], 'sorted_id': 695}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[695] = op;
            
            op->set_inputs( forward_result[694] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.23', 'op': 'aten::exp', 'in': [695], 'output_id': 0, 'shape': [10], 'out': [697], 'sorted_id': 696}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[696] = op;
            
            op->set_inputs( forward_result[695] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3388', 'op': 'aten::mul', 'in': [696, 58], 'output_id': 0, 'shape': [10], 'out': [698], 'sorted_id': 697}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[697] = op;
            
            op->set_inputs( forward_result[696] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3389', 'op': 'aten::add', 'in': [685, 697, 12], 'output_id': 0, 'shape': [10], 'out': [699], 'sorted_id': 698}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[698] = op;
            
            op->set_inputs( forward_result[685] );
            op->set_inputs( forward_result[697] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3390', 'op': 'aten::log', 'in': [698], 'output_id': 0, 'shape': [10], 'out': [700], 'sorted_id': 699}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[699] = op;
            
            op->set_inputs( forward_result[698] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3391', 'op': 'aten::sum', 'in': [699, 20], 'output_id': 0, 'shape': [], 'out': [701], 'sorted_id': 700}
        {
            SumOp* op = new SumOp();
            forward_result[700] = op;
            
            op->set_inputs( forward_result[699] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3392', 'op': 'aten::add', 'in': [673, 700, 12], 'output_id': 0, 'shape': [], 'out': [729], 'sorted_id': 701}
        {
            AddOp* op = new AddOp();
            forward_result[701] = op;
            
            op->set_inputs( forward_result[673] );
            op->set_inputs( forward_result[700] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3393', 'op': 'aten::exp', 'in': [614], 'output_id': 0, 'shape': [10, 400], 'out': [703], 'sorted_id': 702}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[702] = op;
            
            op->set_inputs( forward_result[614] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3394', 'op': 'aten::log1p', 'in': [702], 'output_id': 0, 'shape': [10, 400], 'out': [704], 'sorted_id': 703}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[703] = op;
            
            op->set_inputs( forward_result[702] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3395', 'op': 'aten::log', 'in': [703], 'output_id': 0, 'shape': [10, 400], 'out': [705], 'sorted_id': 704}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[704] = op;
            
            op->set_inputs( forward_result[703] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3396', 'op': 'aten::rsub', 'in': [704, 107, 12], 'output_id': 0, 'shape': [10, 400], 'out': [713], 'sorted_id': 705}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[705] = op;
            
            op->set_inputs( forward_result[704] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3397', 'op': 'aten::sub', 'in': [628, 613, 12], 'output_id': 0, 'shape': [10, 400], 'out': [707], 'sorted_id': 706}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[706] = op;
            
            op->set_inputs( forward_result[628] );
            op->set_inputs( forward_result[613] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3398', 'op': 'aten::pow', 'in': [706, 45], 'output_id': 0, 'shape': [10, 400], 'out': [712], 'sorted_id': 707}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[707] = op;
            
            op->set_inputs( forward_result[706] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3399', 'op': 'aten::exp', 'in': [614], 'output_id': 0, 'shape': [10, 400], 'out': [709], 'sorted_id': 708}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[708] = op;
            
            op->set_inputs( forward_result[614] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3400', 'op': 'aten::log1p', 'in': [708], 'output_id': 0, 'shape': [10, 400], 'out': [710], 'sorted_id': 709}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[709] = op;
            
            op->set_inputs( forward_result[708] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3401', 'op': 'aten::pow', 'in': [709, 45], 'output_id': 0, 'shape': [10, 400], 'out': [711], 'sorted_id': 710}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[710] = op;
            
            op->set_inputs( forward_result[709] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3402', 'op': 'aten::mul', 'in': [710, 50], 'output_id': 0, 'shape': [10, 400], 'out': [712], 'sorted_id': 711}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[711] = op;
            
            op->set_inputs( forward_result[710] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3403', 'op': 'aten::div', 'in': [707, 711], 'output_id': 0, 'shape': [10, 400], 'out': [713], 'sorted_id': 712}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[712] = op;
            
            op->set_inputs( forward_result[707] );
            op->set_inputs( forward_result[711] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3404', 'op': 'aten::sub', 'in': [705, 712, 12], 'output_id': 0, 'shape': [10, 400], 'out': [714], 'sorted_id': 713}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[713] = op;
            
            op->set_inputs( forward_result[705] );
            op->set_inputs( forward_result[712] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3405', 'op': 'aten::sum', 'in': [713, 20], 'output_id': 0, 'shape': [], 'out': [728], 'sorted_id': 714}
        {
            SumOp* op = new SumOp();
            forward_result[714] = op;
            
            op->set_inputs( forward_result[713] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3406', 'op': 'aten::exp', 'in': [630], 'output_id': 0, 'shape': [10], 'out': [716], 'sorted_id': 715}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[715] = op;
            
            op->set_inputs( forward_result[630] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3407', 'op': 'aten::log1p', 'in': [715], 'output_id': 0, 'shape': [10], 'out': [717], 'sorted_id': 716}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[716] = op;
            
            op->set_inputs( forward_result[715] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3408', 'op': 'aten::log', 'in': [716], 'output_id': 0, 'shape': [10], 'out': [718], 'sorted_id': 717}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[717] = op;
            
            op->set_inputs( forward_result[716] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3409', 'op': 'aten::rsub', 'in': [717, 107, 12], 'output_id': 0, 'shape': [10], 'out': [726], 'sorted_id': 718}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[718] = op;
            
            op->set_inputs( forward_result[717] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3410', 'op': 'aten::sub', 'in': [643, 629, 12], 'output_id': 0, 'shape': [10], 'out': [720], 'sorted_id': 719}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[719] = op;
            
            op->set_inputs( forward_result[643] );
            op->set_inputs( forward_result[629] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3411', 'op': 'aten::pow', 'in': [719, 45], 'output_id': 0, 'shape': [10], 'out': [725], 'sorted_id': 720}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[720] = op;
            
            op->set_inputs( forward_result[719] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3412', 'op': 'aten::exp', 'in': [630], 'output_id': 0, 'shape': [10], 'out': [722], 'sorted_id': 721}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[721] = op;
            
            op->set_inputs( forward_result[630] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3413', 'op': 'aten::log1p', 'in': [721], 'output_id': 0, 'shape': [10], 'out': [723], 'sorted_id': 722}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[722] = op;
            
            op->set_inputs( forward_result[721] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3414', 'op': 'aten::pow', 'in': [722, 45], 'output_id': 0, 'shape': [10], 'out': [724], 'sorted_id': 723}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[723] = op;
            
            op->set_inputs( forward_result[722] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3415', 'op': 'aten::mul', 'in': [723, 50], 'output_id': 0, 'shape': [10], 'out': [725], 'sorted_id': 724}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[724] = op;
            
            op->set_inputs( forward_result[723] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3416', 'op': 'aten::div', 'in': [720, 724], 'output_id': 0, 'shape': [10], 'out': [726], 'sorted_id': 725}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[725] = op;
            
            op->set_inputs( forward_result[720] );
            op->set_inputs( forward_result[724] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3417', 'op': 'aten::sub', 'in': [718, 725, 12], 'output_id': 0, 'shape': [10], 'out': [727], 'sorted_id': 726}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[726] = op;
            
            op->set_inputs( forward_result[718] );
            op->set_inputs( forward_result[725] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3418', 'op': 'aten::sum', 'in': [726, 20], 'output_id': 0, 'shape': [], 'out': [728], 'sorted_id': 727}
        {
            SumOp* op = new SumOp();
            forward_result[727] = op;
            
            op->set_inputs( forward_result[726] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3419', 'op': 'aten::add', 'in': [714, 727, 12], 'output_id': 0, 'shape': [], 'out': [729], 'sorted_id': 728}
        {
            AddOp* op = new AddOp();
            forward_result[728] = op;
            
            op->set_inputs( forward_result[714] );
            op->set_inputs( forward_result[727] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3421', 'op': 'prim::TupleConstruct', 'in': [644, 701, 728], 'output_id': 0, 'shape': [], 'out': [1073, 1091, 730], 'sorted_id': 729}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[729] = op;
            
            op->set_inputs( forward_result[644] );
            op->set_inputs( forward_result[701] );
            op->set_inputs( forward_result[728] );
        }
        
        // {'name': 'Model/3422', 'op': 'prim::TupleUnpack', 'in': [729], 'output_id': 0, 'shape': [4, 10], 'out': [731], 'sorted_id': 730}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[730] = op;
            
            op->set_inputs( forward_result[729] );
        }
        
        // {'name': 'Model/Net[net]/3425', 'op': 'aten::log_softmax', 'in': [730, 12, 20], 'output_id': 0, 'shape': [4, 10], 'out': [1061], 'sorted_id': 731}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[731] = op;
            
            op->set_inputs( forward_result[730] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/3430', 'op': 'prim::ListConstruct', 'in': [1, 2], 'output_id': 0, 'shape': [], 'out': [733], 'sorted_id': 732}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[732] = op;
            
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Model/Net[net]/input.25', 'op': 'aten::view', 'in': [0, 732], 'output_id': 0, 'shape': [4, 784], 'out': [761], 'sorted_id': 733}
        {
            Tensor::shape_type shape = {4,784};
            ViewOp* op = new ViewOp();
            forward_result[733] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[732] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_mu/weight_mu.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [817, 747], 'sorted_id': 734}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_mu.reshape( shape );
            forward_result[734] = new VariableTensor( l1_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/weight_rho/weight_rho.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [819, 736, 739, 738, 813], 'sorted_id': 735}
        {
            Tensor::shape_type shape = {400,784};
            l1_weight_rho.reshape( shape );
            forward_result[735] = new VariableTensor( l1_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3444', 'op': 'aten::exp', 'in': [735], 'output_id': 0, 'shape': [400, 784], 'out': [737], 'sorted_id': 736}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[736] = op;
            
            op->set_inputs( forward_result[735] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3445', 'op': 'aten::log1p', 'in': [736], 'output_id': 0, 'shape': [400, 784], 'out': [746], 'sorted_id': 737}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[737] = op;
            
            op->set_inputs( forward_result[736] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3436', 'op': 'aten::size', 'in': [735, 10], 'output_id': 0, 'shape': [], 'out': [742, 740], 'sorted_id': 738}
        {
            SizeOp* op = new SizeOp();
            forward_result[738] = op;
            
            op->set_inputs( forward_result[735] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3437', 'op': 'aten::size', 'in': [735, 12], 'output_id': 0, 'shape': [], 'out': [742, 740], 'sorted_id': 739}
        {
            SizeOp* op = new SizeOp();
            forward_result[739] = op;
            
            op->set_inputs( forward_result[735] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3438', 'op': 'prim::ListConstruct', 'in': [738, 739], 'output_id': 0, 'shape': [], 'out': [741], 'sorted_id': 740}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[740] = op;
            
            op->set_inputs( forward_result[738] );
            op->set_inputs( forward_result[739] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3439', 'op': 'aten::expand', 'in': [379, 740, 15], 'output_id': 0, 'shape': [400, 784], 'out': [744], 'sorted_id': 741}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[741] = op;
            
            op->set_inputs( forward_result[379] );
            op->set_inputs( forward_result[740] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3440', 'op': 'prim::ListConstruct', 'in': [738, 739], 'output_id': 0, 'shape': [], 'out': [743], 'sorted_id': 742}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[742] = op;
            
            op->set_inputs( forward_result[738] );
            op->set_inputs( forward_result[739] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3441', 'op': 'aten::expand', 'in': [384, 742, 15], 'output_id': 0, 'shape': [400, 784], 'out': [744], 'sorted_id': 743}
        {
            Tensor::shape_type shape = {400,784};
            ExpandOp* op = new ExpandOp();
            forward_result[743] = op;
            
            op->set_inputs( forward_result[384] );
            op->set_inputs( forward_result[742] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3442', 'op': 'aten::normal', 'in': [741, 743, 20], 'output_id': 0, 'shape': [400, 784], 'out': [745], 'sorted_id': 744}
        {
            Tensor::shape_type shape = {400,784};
            NormalOp* op = new NormalOp();
            forward_result[744] = op;
            
            op->set_inputs( forward_result[741] );
            op->set_inputs( forward_result[743] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.25', 'op': 'aten::to', 'in': [744, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 784], 'out': [746], 'sorted_id': 745}
        {
            Tensor::shape_type shape = {400,784};
            ToOp* op = new ToOp();
            forward_result[745] = op;
            
            op->set_inputs( forward_result[744] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3446', 'op': 'aten::mul', 'in': [737, 745], 'output_id': 0, 'shape': [400, 784], 'out': [747], 'sorted_id': 746}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[746] = op;
            
            op->set_inputs( forward_result[737] );
            op->set_inputs( forward_result[745] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.25', 'op': 'aten::add', 'in': [734, 746, 12], 'output_id': 0, 'shape': [400, 784], 'out': [762, 773, 761, 817], 'sorted_id': 747}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[747] = op;
            
            op->set_inputs( forward_result[734] );
            op->set_inputs( forward_result[746] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_mu/bias_mu.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [760, 830], 'sorted_id': 748}
        {
            Tensor::shape_type shape = {400};
            l1_bias_mu.reshape( shape );
            forward_result[748] = new VariableTensor( l1_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/bias_rho/bias_rho.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [826, 750, 752, 832], 'sorted_id': 749}
        {
            Tensor::shape_type shape = {400};
            l1_bias_rho.reshape( shape );
            forward_result[749] = new VariableTensor( l1_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3455', 'op': 'aten::exp', 'in': [749], 'output_id': 0, 'shape': [400], 'out': [751], 'sorted_id': 750}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[750] = op;
            
            op->set_inputs( forward_result[749] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3456', 'op': 'aten::log1p', 'in': [750], 'output_id': 0, 'shape': [400], 'out': [759], 'sorted_id': 751}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[751] = op;
            
            op->set_inputs( forward_result[750] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3448', 'op': 'aten::size', 'in': [749, 10], 'output_id': 0, 'shape': [], 'out': [755, 753], 'sorted_id': 752}
        {
            SizeOp* op = new SizeOp();
            forward_result[752] = op;
            
            op->set_inputs( forward_result[749] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3449', 'op': 'prim::ListConstruct', 'in': [752], 'output_id': 0, 'shape': [], 'out': [754], 'sorted_id': 753}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[753] = op;
            
            op->set_inputs( forward_result[752] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3450', 'op': 'aten::expand', 'in': [395, 753, 15], 'output_id': 0, 'shape': [400], 'out': [757], 'sorted_id': 754}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[754] = op;
            
            op->set_inputs( forward_result[395] );
            op->set_inputs( forward_result[753] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3451', 'op': 'prim::ListConstruct', 'in': [752], 'output_id': 0, 'shape': [], 'out': [756], 'sorted_id': 755}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[755] = op;
            
            op->set_inputs( forward_result[752] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3452', 'op': 'aten::expand', 'in': [399, 755, 15], 'output_id': 0, 'shape': [400], 'out': [757], 'sorted_id': 756}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[756] = op;
            
            op->set_inputs( forward_result[399] );
            op->set_inputs( forward_result[755] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3453', 'op': 'aten::normal', 'in': [754, 756, 20], 'output_id': 0, 'shape': [400], 'out': [758], 'sorted_id': 757}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[757] = op;
            
            op->set_inputs( forward_result[754] );
            op->set_inputs( forward_result[756] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/epsilon.27', 'op': 'aten::to', 'in': [757, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [759], 'sorted_id': 758}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[758] = op;
            
            op->set_inputs( forward_result[757] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3457', 'op': 'aten::mul', 'in': [751, 758], 'output_id': 0, 'shape': [400], 'out': [760], 'sorted_id': 759}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[759] = op;
            
            op->set_inputs( forward_result[751] );
            op->set_inputs( forward_result[758] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/value.27', 'op': 'aten::add', 'in': [748, 759, 12], 'output_id': 0, 'shape': [400], 'out': [798, 761, 830, 787], 'sorted_id': 760}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[760] = op;
            
            op->set_inputs( forward_result[748] );
            op->set_inputs( forward_result[759] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/input.27', 'op': 'aten::linear', 'in': [733, 747, 760], 'output_id': 0, 'shape': [4, 400], 'out': [840], 'sorted_id': 761}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[761] = op;
            
            op->set_inputs( forward_result[733] );
            op->set_inputs( forward_result[747] );
            op->set_inputs( forward_result[760] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3461', 'op': 'aten::sub', 'in': [747, 407, 12], 'output_id': 0, 'shape': [400, 784], 'out': [763], 'sorted_id': 762}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[762] = op;
            
            op->set_inputs( forward_result[747] );
            op->set_inputs( forward_result[407] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3462', 'op': 'aten::pow', 'in': [762, 45], 'output_id': 0, 'shape': [400, 784], 'out': [764], 'sorted_id': 763}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[763] = op;
            
            op->set_inputs( forward_result[762] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3463', 'op': 'aten::neg', 'in': [763], 'output_id': 0, 'shape': [400, 784], 'out': [767], 'sorted_id': 764}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[764] = op;
            
            op->set_inputs( forward_result[763] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.49', 'op': 'aten::pow', 'in': [411, 45], 'output_id': 0, 'shape': [1], 'out': [766], 'sorted_id': 765}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[765] = op;
            
            op->set_inputs( forward_result[411] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3464', 'op': 'aten::mul', 'in': [765, 50], 'output_id': 0, 'shape': [1], 'out': [767], 'sorted_id': 766}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[766] = op;
            
            op->set_inputs( forward_result[765] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3465', 'op': 'aten::div', 'in': [764, 766], 'output_id': 0, 'shape': [400, 784], 'out': [769], 'sorted_id': 767}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[767] = op;
            
            op->set_inputs( forward_result[764] );
            op->set_inputs( forward_result[766] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.49', 'op': 'aten::log', 'in': [411], 'output_id': 0, 'shape': [1], 'out': [769], 'sorted_id': 768}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[768] = op;
            
            op->set_inputs( forward_result[411] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3466', 'op': 'aten::sub', 'in': [767, 768, 12], 'output_id': 0, 'shape': [400, 784], 'out': [770], 'sorted_id': 769}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[769] = op;
            
            op->set_inputs( forward_result[767] );
            op->set_inputs( forward_result[768] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3467', 'op': 'aten::sub', 'in': [769, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [771], 'sorted_id': 770}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[770] = op;
            
            op->set_inputs( forward_result[769] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.25', 'op': 'aten::exp', 'in': [770], 'output_id': 0, 'shape': [400, 784], 'out': [772], 'sorted_id': 771}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[771] = op;
            
            op->set_inputs( forward_result[770] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3479', 'op': 'aten::mul', 'in': [771, 58], 'output_id': 0, 'shape': [400, 784], 'out': [784], 'sorted_id': 772}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[772] = op;
            
            op->set_inputs( forward_result[771] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3471', 'op': 'aten::sub', 'in': [747, 420, 12], 'output_id': 0, 'shape': [400, 784], 'out': [774], 'sorted_id': 773}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[773] = op;
            
            op->set_inputs( forward_result[747] );
            op->set_inputs( forward_result[420] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3472', 'op': 'aten::pow', 'in': [773, 45], 'output_id': 0, 'shape': [400, 784], 'out': [775], 'sorted_id': 774}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[774] = op;
            
            op->set_inputs( forward_result[773] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3473', 'op': 'aten::neg', 'in': [774], 'output_id': 0, 'shape': [400, 784], 'out': [778], 'sorted_id': 775}
        {
            Tensor::shape_type shape = {400,784};
            NegOp* op = new NegOp();
            forward_result[775] = op;
            
            op->set_inputs( forward_result[774] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.51', 'op': 'aten::pow', 'in': [424, 45], 'output_id': 0, 'shape': [1], 'out': [777], 'sorted_id': 776}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[776] = op;
            
            op->set_inputs( forward_result[424] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3474', 'op': 'aten::mul', 'in': [776, 50], 'output_id': 0, 'shape': [1], 'out': [778], 'sorted_id': 777}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[777] = op;
            
            op->set_inputs( forward_result[776] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3475', 'op': 'aten::div', 'in': [775, 777], 'output_id': 0, 'shape': [400, 784], 'out': [780], 'sorted_id': 778}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[778] = op;
            
            op->set_inputs( forward_result[775] );
            op->set_inputs( forward_result[777] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.51', 'op': 'aten::log', 'in': [424], 'output_id': 0, 'shape': [1], 'out': [780], 'sorted_id': 779}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[779] = op;
            
            op->set_inputs( forward_result[424] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3476', 'op': 'aten::sub', 'in': [778, 779, 12], 'output_id': 0, 'shape': [400, 784], 'out': [781], 'sorted_id': 780}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[780] = op;
            
            op->set_inputs( forward_result[778] );
            op->set_inputs( forward_result[779] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3477', 'op': 'aten::sub', 'in': [780, 55, 12], 'output_id': 0, 'shape': [400, 784], 'out': [782], 'sorted_id': 781}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[781] = op;
            
            op->set_inputs( forward_result[780] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.25', 'op': 'aten::exp', 'in': [781], 'output_id': 0, 'shape': [400, 784], 'out': [783], 'sorted_id': 782}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[782] = op;
            
            op->set_inputs( forward_result[781] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3480', 'op': 'aten::mul', 'in': [782, 58], 'output_id': 0, 'shape': [400, 784], 'out': [784], 'sorted_id': 783}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[783] = op;
            
            op->set_inputs( forward_result[782] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3481', 'op': 'aten::add', 'in': [772, 783, 12], 'output_id': 0, 'shape': [400, 784], 'out': [785], 'sorted_id': 784}
        {
            Tensor::shape_type shape = {400,784};
            AddOp* op = new AddOp();
            forward_result[784] = op;
            
            op->set_inputs( forward_result[772] );
            op->set_inputs( forward_result[783] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3482', 'op': 'aten::log', 'in': [784], 'output_id': 0, 'shape': [400, 784], 'out': [786], 'sorted_id': 785}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[785] = op;
            
            op->set_inputs( forward_result[784] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3483', 'op': 'aten::sum', 'in': [785, 20], 'output_id': 0, 'shape': [], 'out': [812], 'sorted_id': 786}
        {
            SumOp* op = new SumOp();
            forward_result[786] = op;
            
            op->set_inputs( forward_result[785] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3486', 'op': 'aten::sub', 'in': [760, 436, 12], 'output_id': 0, 'shape': [400], 'out': [788], 'sorted_id': 787}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[787] = op;
            
            op->set_inputs( forward_result[760] );
            op->set_inputs( forward_result[436] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3487', 'op': 'aten::pow', 'in': [787, 45], 'output_id': 0, 'shape': [400], 'out': [789], 'sorted_id': 788}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[788] = op;
            
            op->set_inputs( forward_result[787] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3488', 'op': 'aten::neg', 'in': [788], 'output_id': 0, 'shape': [400], 'out': [792], 'sorted_id': 789}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[789] = op;
            
            op->set_inputs( forward_result[788] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.53', 'op': 'aten::pow', 'in': [411, 45], 'output_id': 0, 'shape': [1], 'out': [791], 'sorted_id': 790}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[790] = op;
            
            op->set_inputs( forward_result[411] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3489', 'op': 'aten::mul', 'in': [790, 50], 'output_id': 0, 'shape': [1], 'out': [792], 'sorted_id': 791}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[791] = op;
            
            op->set_inputs( forward_result[790] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3490', 'op': 'aten::div', 'in': [789, 791], 'output_id': 0, 'shape': [400], 'out': [794], 'sorted_id': 792}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[792] = op;
            
            op->set_inputs( forward_result[789] );
            op->set_inputs( forward_result[791] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.53', 'op': 'aten::log', 'in': [411], 'output_id': 0, 'shape': [1], 'out': [794], 'sorted_id': 793}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[793] = op;
            
            op->set_inputs( forward_result[411] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3491', 'op': 'aten::sub', 'in': [792, 793, 12], 'output_id': 0, 'shape': [400], 'out': [795], 'sorted_id': 794}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[794] = op;
            
            op->set_inputs( forward_result[792] );
            op->set_inputs( forward_result[793] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3492', 'op': 'aten::sub', 'in': [794, 55, 12], 'output_id': 0, 'shape': [400], 'out': [796], 'sorted_id': 795}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[795] = op;
            
            op->set_inputs( forward_result[794] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob1.27', 'op': 'aten::exp', 'in': [795], 'output_id': 0, 'shape': [400], 'out': [797], 'sorted_id': 796}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[796] = op;
            
            op->set_inputs( forward_result[795] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3504', 'op': 'aten::mul', 'in': [796, 58], 'output_id': 0, 'shape': [400], 'out': [809], 'sorted_id': 797}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[797] = op;
            
            op->set_inputs( forward_result[796] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3496', 'op': 'aten::sub', 'in': [760, 448, 12], 'output_id': 0, 'shape': [400], 'out': [799], 'sorted_id': 798}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[798] = op;
            
            op->set_inputs( forward_result[760] );
            op->set_inputs( forward_result[448] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3497', 'op': 'aten::pow', 'in': [798, 45], 'output_id': 0, 'shape': [400], 'out': [800], 'sorted_id': 799}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[799] = op;
            
            op->set_inputs( forward_result[798] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3498', 'op': 'aten::neg', 'in': [799], 'output_id': 0, 'shape': [400], 'out': [803], 'sorted_id': 800}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[800] = op;
            
            op->set_inputs( forward_result[799] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/var.55', 'op': 'aten::pow', 'in': [424, 45], 'output_id': 0, 'shape': [1], 'out': [802], 'sorted_id': 801}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[801] = op;
            
            op->set_inputs( forward_result[424] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3499', 'op': 'aten::mul', 'in': [801, 50], 'output_id': 0, 'shape': [1], 'out': [803], 'sorted_id': 802}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[802] = op;
            
            op->set_inputs( forward_result[801] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3500', 'op': 'aten::div', 'in': [800, 802], 'output_id': 0, 'shape': [400], 'out': [805], 'sorted_id': 803}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[803] = op;
            
            op->set_inputs( forward_result[800] );
            op->set_inputs( forward_result[802] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/log_scale.55', 'op': 'aten::log', 'in': [424], 'output_id': 0, 'shape': [1], 'out': [805], 'sorted_id': 804}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[804] = op;
            
            op->set_inputs( forward_result[424] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3501', 'op': 'aten::sub', 'in': [803, 804, 12], 'output_id': 0, 'shape': [400], 'out': [806], 'sorted_id': 805}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[805] = op;
            
            op->set_inputs( forward_result[803] );
            op->set_inputs( forward_result[804] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3502', 'op': 'aten::sub', 'in': [805, 55, 12], 'output_id': 0, 'shape': [400], 'out': [807], 'sorted_id': 806}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[806] = op;
            
            op->set_inputs( forward_result[805] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/prob2.27', 'op': 'aten::exp', 'in': [806], 'output_id': 0, 'shape': [400], 'out': [808], 'sorted_id': 807}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[807] = op;
            
            op->set_inputs( forward_result[806] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3505', 'op': 'aten::mul', 'in': [807, 58], 'output_id': 0, 'shape': [400], 'out': [809], 'sorted_id': 808}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[808] = op;
            
            op->set_inputs( forward_result[807] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3506', 'op': 'aten::add', 'in': [797, 808, 12], 'output_id': 0, 'shape': [400], 'out': [810], 'sorted_id': 809}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[809] = op;
            
            op->set_inputs( forward_result[797] );
            op->set_inputs( forward_result[808] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3507', 'op': 'aten::log', 'in': [809], 'output_id': 0, 'shape': [400], 'out': [811], 'sorted_id': 810}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[810] = op;
            
            op->set_inputs( forward_result[809] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3508', 'op': 'aten::sum', 'in': [810, 20], 'output_id': 0, 'shape': [], 'out': [812], 'sorted_id': 811}
        {
            SumOp* op = new SumOp();
            forward_result[811] = op;
            
            op->set_inputs( forward_result[810] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3509', 'op': 'aten::add', 'in': [786, 811, 12], 'output_id': 0, 'shape': [], 'out': [840], 'sorted_id': 812}
        {
            AddOp* op = new AddOp();
            forward_result[812] = op;
            
            op->set_inputs( forward_result[786] );
            op->set_inputs( forward_result[811] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3510', 'op': 'aten::exp', 'in': [735], 'output_id': 0, 'shape': [400, 784], 'out': [814], 'sorted_id': 813}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[813] = op;
            
            op->set_inputs( forward_result[735] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3511', 'op': 'aten::log1p', 'in': [813], 'output_id': 0, 'shape': [400, 784], 'out': [815], 'sorted_id': 814}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[814] = op;
            
            op->set_inputs( forward_result[813] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3512', 'op': 'aten::log', 'in': [814], 'output_id': 0, 'shape': [400, 784], 'out': [816], 'sorted_id': 815}
        {
            Tensor::shape_type shape = {400,784};
            LogOp* op = new LogOp();
            forward_result[815] = op;
            
            op->set_inputs( forward_result[814] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3513', 'op': 'aten::rsub', 'in': [815, 107, 12], 'output_id': 0, 'shape': [400, 784], 'out': [824], 'sorted_id': 816}
        {
            Tensor::shape_type shape = {400,784};
            RsubOp* op = new RsubOp();
            forward_result[816] = op;
            
            op->set_inputs( forward_result[815] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3514', 'op': 'aten::sub', 'in': [747, 734, 12], 'output_id': 0, 'shape': [400, 784], 'out': [818], 'sorted_id': 817}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[817] = op;
            
            op->set_inputs( forward_result[747] );
            op->set_inputs( forward_result[734] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3515', 'op': 'aten::pow', 'in': [817, 45], 'output_id': 0, 'shape': [400, 784], 'out': [823], 'sorted_id': 818}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[818] = op;
            
            op->set_inputs( forward_result[817] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3516', 'op': 'aten::exp', 'in': [735], 'output_id': 0, 'shape': [400, 784], 'out': [820], 'sorted_id': 819}
        {
            Tensor::shape_type shape = {400,784};
            ExpOp* op = new ExpOp();
            forward_result[819] = op;
            
            op->set_inputs( forward_result[735] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3517', 'op': 'aten::log1p', 'in': [819], 'output_id': 0, 'shape': [400, 784], 'out': [821], 'sorted_id': 820}
        {
            Tensor::shape_type shape = {400,784};
            Log1pOp* op = new Log1pOp();
            forward_result[820] = op;
            
            op->set_inputs( forward_result[819] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3518', 'op': 'aten::pow', 'in': [820, 45], 'output_id': 0, 'shape': [400, 784], 'out': [822], 'sorted_id': 821}
        {
            Tensor::shape_type shape = {400,784};
            PowOp* op = new PowOp();
            forward_result[821] = op;
            
            op->set_inputs( forward_result[820] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3519', 'op': 'aten::mul', 'in': [821, 50], 'output_id': 0, 'shape': [400, 784], 'out': [823], 'sorted_id': 822}
        {
            Tensor::shape_type shape = {400,784};
            MulOp* op = new MulOp();
            forward_result[822] = op;
            
            op->set_inputs( forward_result[821] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3520', 'op': 'aten::div', 'in': [818, 822], 'output_id': 0, 'shape': [400, 784], 'out': [824], 'sorted_id': 823}
        {
            Tensor::shape_type shape = {400,784};
            DivOp* op = new DivOp();
            forward_result[823] = op;
            
            op->set_inputs( forward_result[818] );
            op->set_inputs( forward_result[822] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3521', 'op': 'aten::sub', 'in': [816, 823, 12], 'output_id': 0, 'shape': [400, 784], 'out': [825], 'sorted_id': 824}
        {
            Tensor::shape_type shape = {400,784};
            SubOp* op = new SubOp();
            forward_result[824] = op;
            
            op->set_inputs( forward_result[816] );
            op->set_inputs( forward_result[823] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3522', 'op': 'aten::sum', 'in': [824, 20], 'output_id': 0, 'shape': [], 'out': [839], 'sorted_id': 825}
        {
            SumOp* op = new SumOp();
            forward_result[825] = op;
            
            op->set_inputs( forward_result[824] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3523', 'op': 'aten::exp', 'in': [749], 'output_id': 0, 'shape': [400], 'out': [827], 'sorted_id': 826}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[826] = op;
            
            op->set_inputs( forward_result[749] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3524', 'op': 'aten::log1p', 'in': [826], 'output_id': 0, 'shape': [400], 'out': [828], 'sorted_id': 827}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[827] = op;
            
            op->set_inputs( forward_result[826] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3525', 'op': 'aten::log', 'in': [827], 'output_id': 0, 'shape': [400], 'out': [829], 'sorted_id': 828}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[828] = op;
            
            op->set_inputs( forward_result[827] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3526', 'op': 'aten::rsub', 'in': [828, 107, 12], 'output_id': 0, 'shape': [400], 'out': [837], 'sorted_id': 829}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[829] = op;
            
            op->set_inputs( forward_result[828] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3527', 'op': 'aten::sub', 'in': [760, 748, 12], 'output_id': 0, 'shape': [400], 'out': [831], 'sorted_id': 830}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[830] = op;
            
            op->set_inputs( forward_result[760] );
            op->set_inputs( forward_result[748] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3528', 'op': 'aten::pow', 'in': [830, 45], 'output_id': 0, 'shape': [400], 'out': [836], 'sorted_id': 831}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[831] = op;
            
            op->set_inputs( forward_result[830] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3529', 'op': 'aten::exp', 'in': [749], 'output_id': 0, 'shape': [400], 'out': [833], 'sorted_id': 832}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[832] = op;
            
            op->set_inputs( forward_result[749] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3530', 'op': 'aten::log1p', 'in': [832], 'output_id': 0, 'shape': [400], 'out': [834], 'sorted_id': 833}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[833] = op;
            
            op->set_inputs( forward_result[832] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3531', 'op': 'aten::pow', 'in': [833, 45], 'output_id': 0, 'shape': [400], 'out': [835], 'sorted_id': 834}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[834] = op;
            
            op->set_inputs( forward_result[833] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3532', 'op': 'aten::mul', 'in': [834, 50], 'output_id': 0, 'shape': [400], 'out': [836], 'sorted_id': 835}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[835] = op;
            
            op->set_inputs( forward_result[834] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3533', 'op': 'aten::div', 'in': [831, 835], 'output_id': 0, 'shape': [400], 'out': [837], 'sorted_id': 836}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[836] = op;
            
            op->set_inputs( forward_result[831] );
            op->set_inputs( forward_result[835] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3534', 'op': 'aten::sub', 'in': [829, 836, 12], 'output_id': 0, 'shape': [400], 'out': [838], 'sorted_id': 837}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[837] = op;
            
            op->set_inputs( forward_result[829] );
            op->set_inputs( forward_result[836] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3535', 'op': 'aten::sum', 'in': [837, 20], 'output_id': 0, 'shape': [], 'out': [839], 'sorted_id': 838}
        {
            SumOp* op = new SumOp();
            forward_result[838] = op;
            
            op->set_inputs( forward_result[837] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l1]/3536', 'op': 'aten::add', 'in': [825, 838, 12], 'output_id': 0, 'shape': [], 'out': [840], 'sorted_id': 839}
        {
            AddOp* op = new AddOp();
            forward_result[839] = op;
            
            op->set_inputs( forward_result[825] );
            op->set_inputs( forward_result[838] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3538', 'op': 'prim::TupleConstruct', 'in': [761, 812, 839], 'output_id': 0, 'shape': [], 'out': [841, 1075, 1093], 'sorted_id': 840}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[840] = op;
            
            op->set_inputs( forward_result[761] );
            op->set_inputs( forward_result[812] );
            op->set_inputs( forward_result[839] );
        }
        
        // {'name': 'Model/3539', 'op': 'prim::TupleUnpack', 'in': [840], 'output_id': 0, 'shape': [4, 400], 'out': [842], 'sorted_id': 841}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[841] = op;
            
            op->set_inputs( forward_result[840] );
        }
        
        // {'name': 'Model/Net[net]/input.29', 'op': 'aten::relu', 'in': [841], 'output_id': 0, 'shape': [4, 400], 'out': [870], 'sorted_id': 842}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[842] = op;
            
            op->set_inputs( forward_result[841] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_mu/weight_mu.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [926, 856], 'sorted_id': 843}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_mu.reshape( shape );
            forward_result[843] = new VariableTensor( l2_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/weight_rho/weight_rho.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [928, 848, 845, 847, 922], 'sorted_id': 844}
        {
            Tensor::shape_type shape = {400,400};
            l2_weight_rho.reshape( shape );
            forward_result[844] = new VariableTensor( l2_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3555', 'op': 'aten::exp', 'in': [844], 'output_id': 0, 'shape': [400, 400], 'out': [846], 'sorted_id': 845}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[845] = op;
            
            op->set_inputs( forward_result[844] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3556', 'op': 'aten::log1p', 'in': [845], 'output_id': 0, 'shape': [400, 400], 'out': [855], 'sorted_id': 846}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[846] = op;
            
            op->set_inputs( forward_result[845] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3547', 'op': 'aten::size', 'in': [844, 10], 'output_id': 0, 'shape': [], 'out': [851, 849], 'sorted_id': 847}
        {
            SizeOp* op = new SizeOp();
            forward_result[847] = op;
            
            op->set_inputs( forward_result[844] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3548', 'op': 'aten::size', 'in': [844, 12], 'output_id': 0, 'shape': [], 'out': [851, 849], 'sorted_id': 848}
        {
            SizeOp* op = new SizeOp();
            forward_result[848] = op;
            
            op->set_inputs( forward_result[844] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3549', 'op': 'prim::ListConstruct', 'in': [847, 848], 'output_id': 0, 'shape': [], 'out': [850], 'sorted_id': 849}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[849] = op;
            
            op->set_inputs( forward_result[847] );
            op->set_inputs( forward_result[848] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3550', 'op': 'aten::expand', 'in': [498, 849, 15], 'output_id': 0, 'shape': [400, 400], 'out': [853], 'sorted_id': 850}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[850] = op;
            
            op->set_inputs( forward_result[498] );
            op->set_inputs( forward_result[849] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3551', 'op': 'prim::ListConstruct', 'in': [847, 848], 'output_id': 0, 'shape': [], 'out': [852], 'sorted_id': 851}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[851] = op;
            
            op->set_inputs( forward_result[847] );
            op->set_inputs( forward_result[848] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3552', 'op': 'aten::expand', 'in': [503, 851, 15], 'output_id': 0, 'shape': [400, 400], 'out': [853], 'sorted_id': 852}
        {
            Tensor::shape_type shape = {400,400};
            ExpandOp* op = new ExpandOp();
            forward_result[852] = op;
            
            op->set_inputs( forward_result[503] );
            op->set_inputs( forward_result[851] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3553', 'op': 'aten::normal', 'in': [850, 852, 20], 'output_id': 0, 'shape': [400, 400], 'out': [854], 'sorted_id': 853}
        {
            Tensor::shape_type shape = {400,400};
            NormalOp* op = new NormalOp();
            forward_result[853] = op;
            
            op->set_inputs( forward_result[850] );
            op->set_inputs( forward_result[852] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.29', 'op': 'aten::to', 'in': [853, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400, 400], 'out': [855], 'sorted_id': 854}
        {
            Tensor::shape_type shape = {400,400};
            ToOp* op = new ToOp();
            forward_result[854] = op;
            
            op->set_inputs( forward_result[853] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3557', 'op': 'aten::mul', 'in': [846, 854], 'output_id': 0, 'shape': [400, 400], 'out': [856], 'sorted_id': 855}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[855] = op;
            
            op->set_inputs( forward_result[846] );
            op->set_inputs( forward_result[854] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.29', 'op': 'aten::add', 'in': [843, 855, 12], 'output_id': 0, 'shape': [400, 400], 'out': [926, 870, 871, 882], 'sorted_id': 856}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[856] = op;
            
            op->set_inputs( forward_result[843] );
            op->set_inputs( forward_result[855] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_mu/bias_mu.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [939, 869], 'sorted_id': 857}
        {
            Tensor::shape_type shape = {400};
            l2_bias_mu.reshape( shape );
            forward_result[857] = new VariableTensor( l2_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/bias_rho/bias_rho.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [859, 941, 861, 935], 'sorted_id': 858}
        {
            Tensor::shape_type shape = {400};
            l2_bias_rho.reshape( shape );
            forward_result[858] = new VariableTensor( l2_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3566', 'op': 'aten::exp', 'in': [858], 'output_id': 0, 'shape': [400], 'out': [860], 'sorted_id': 859}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[859] = op;
            
            op->set_inputs( forward_result[858] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3567', 'op': 'aten::log1p', 'in': [859], 'output_id': 0, 'shape': [400], 'out': [868], 'sorted_id': 860}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[860] = op;
            
            op->set_inputs( forward_result[859] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3559', 'op': 'aten::size', 'in': [858, 10], 'output_id': 0, 'shape': [], 'out': [864, 862], 'sorted_id': 861}
        {
            SizeOp* op = new SizeOp();
            forward_result[861] = op;
            
            op->set_inputs( forward_result[858] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3560', 'op': 'prim::ListConstruct', 'in': [861], 'output_id': 0, 'shape': [], 'out': [863], 'sorted_id': 862}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[862] = op;
            
            op->set_inputs( forward_result[861] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3561', 'op': 'aten::expand', 'in': [514, 862, 15], 'output_id': 0, 'shape': [400], 'out': [866], 'sorted_id': 863}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[863] = op;
            
            op->set_inputs( forward_result[514] );
            op->set_inputs( forward_result[862] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3562', 'op': 'prim::ListConstruct', 'in': [861], 'output_id': 0, 'shape': [], 'out': [865], 'sorted_id': 864}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[864] = op;
            
            op->set_inputs( forward_result[861] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3563', 'op': 'aten::expand', 'in': [518, 864, 15], 'output_id': 0, 'shape': [400], 'out': [866], 'sorted_id': 865}
        {
            Tensor::shape_type shape = {400};
            ExpandOp* op = new ExpandOp();
            forward_result[865] = op;
            
            op->set_inputs( forward_result[518] );
            op->set_inputs( forward_result[864] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3564', 'op': 'aten::normal', 'in': [863, 865, 20], 'output_id': 0, 'shape': [400], 'out': [867], 'sorted_id': 866}
        {
            Tensor::shape_type shape = {400};
            NormalOp* op = new NormalOp();
            forward_result[866] = op;
            
            op->set_inputs( forward_result[863] );
            op->set_inputs( forward_result[865] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/epsilon.31', 'op': 'aten::to', 'in': [866, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [400], 'out': [868], 'sorted_id': 867}
        {
            Tensor::shape_type shape = {400};
            ToOp* op = new ToOp();
            forward_result[867] = op;
            
            op->set_inputs( forward_result[866] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3568', 'op': 'aten::mul', 'in': [860, 867], 'output_id': 0, 'shape': [400], 'out': [869], 'sorted_id': 868}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[868] = op;
            
            op->set_inputs( forward_result[860] );
            op->set_inputs( forward_result[867] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/value.31', 'op': 'aten::add', 'in': [857, 868, 12], 'output_id': 0, 'shape': [400], 'out': [907, 939, 870, 896], 'sorted_id': 869}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[869] = op;
            
            op->set_inputs( forward_result[857] );
            op->set_inputs( forward_result[868] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/input.31', 'op': 'aten::linear', 'in': [842, 856, 869], 'output_id': 0, 'shape': [4, 400], 'out': [949], 'sorted_id': 870}
        {
            Tensor::shape_type shape = {4,400};
            LinearOp* op = new LinearOp();
            forward_result[870] = op;
            
            op->set_inputs( forward_result[842] );
            op->set_inputs( forward_result[856] );
            op->set_inputs( forward_result[869] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3572', 'op': 'aten::sub', 'in': [856, 526, 12], 'output_id': 0, 'shape': [400, 400], 'out': [872], 'sorted_id': 871}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[871] = op;
            
            op->set_inputs( forward_result[856] );
            op->set_inputs( forward_result[526] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3573', 'op': 'aten::pow', 'in': [871, 45], 'output_id': 0, 'shape': [400, 400], 'out': [873], 'sorted_id': 872}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[872] = op;
            
            op->set_inputs( forward_result[871] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3574', 'op': 'aten::neg', 'in': [872], 'output_id': 0, 'shape': [400, 400], 'out': [876], 'sorted_id': 873}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[873] = op;
            
            op->set_inputs( forward_result[872] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.57', 'op': 'aten::pow', 'in': [530, 45], 'output_id': 0, 'shape': [1], 'out': [875], 'sorted_id': 874}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[874] = op;
            
            op->set_inputs( forward_result[530] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3575', 'op': 'aten::mul', 'in': [874, 50], 'output_id': 0, 'shape': [1], 'out': [876], 'sorted_id': 875}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[875] = op;
            
            op->set_inputs( forward_result[874] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3576', 'op': 'aten::div', 'in': [873, 875], 'output_id': 0, 'shape': [400, 400], 'out': [878], 'sorted_id': 876}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[876] = op;
            
            op->set_inputs( forward_result[873] );
            op->set_inputs( forward_result[875] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.57', 'op': 'aten::log', 'in': [530], 'output_id': 0, 'shape': [1], 'out': [878], 'sorted_id': 877}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[877] = op;
            
            op->set_inputs( forward_result[530] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3577', 'op': 'aten::sub', 'in': [876, 877, 12], 'output_id': 0, 'shape': [400, 400], 'out': [879], 'sorted_id': 878}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[878] = op;
            
            op->set_inputs( forward_result[876] );
            op->set_inputs( forward_result[877] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3578', 'op': 'aten::sub', 'in': [878, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [880], 'sorted_id': 879}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[879] = op;
            
            op->set_inputs( forward_result[878] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.29', 'op': 'aten::exp', 'in': [879], 'output_id': 0, 'shape': [400, 400], 'out': [881], 'sorted_id': 880}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[880] = op;
            
            op->set_inputs( forward_result[879] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3590', 'op': 'aten::mul', 'in': [880, 58], 'output_id': 0, 'shape': [400, 400], 'out': [893], 'sorted_id': 881}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[881] = op;
            
            op->set_inputs( forward_result[880] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3582', 'op': 'aten::sub', 'in': [856, 539, 12], 'output_id': 0, 'shape': [400, 400], 'out': [883], 'sorted_id': 882}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[882] = op;
            
            op->set_inputs( forward_result[856] );
            op->set_inputs( forward_result[539] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3583', 'op': 'aten::pow', 'in': [882, 45], 'output_id': 0, 'shape': [400, 400], 'out': [884], 'sorted_id': 883}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[883] = op;
            
            op->set_inputs( forward_result[882] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3584', 'op': 'aten::neg', 'in': [883], 'output_id': 0, 'shape': [400, 400], 'out': [887], 'sorted_id': 884}
        {
            Tensor::shape_type shape = {400,400};
            NegOp* op = new NegOp();
            forward_result[884] = op;
            
            op->set_inputs( forward_result[883] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.59', 'op': 'aten::pow', 'in': [543, 45], 'output_id': 0, 'shape': [1], 'out': [886], 'sorted_id': 885}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[885] = op;
            
            op->set_inputs( forward_result[543] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3585', 'op': 'aten::mul', 'in': [885, 50], 'output_id': 0, 'shape': [1], 'out': [887], 'sorted_id': 886}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[886] = op;
            
            op->set_inputs( forward_result[885] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3586', 'op': 'aten::div', 'in': [884, 886], 'output_id': 0, 'shape': [400, 400], 'out': [889], 'sorted_id': 887}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[887] = op;
            
            op->set_inputs( forward_result[884] );
            op->set_inputs( forward_result[886] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.59', 'op': 'aten::log', 'in': [543], 'output_id': 0, 'shape': [1], 'out': [889], 'sorted_id': 888}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[888] = op;
            
            op->set_inputs( forward_result[543] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3587', 'op': 'aten::sub', 'in': [887, 888, 12], 'output_id': 0, 'shape': [400, 400], 'out': [890], 'sorted_id': 889}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[889] = op;
            
            op->set_inputs( forward_result[887] );
            op->set_inputs( forward_result[888] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3588', 'op': 'aten::sub', 'in': [889, 55, 12], 'output_id': 0, 'shape': [400, 400], 'out': [891], 'sorted_id': 890}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[890] = op;
            
            op->set_inputs( forward_result[889] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.29', 'op': 'aten::exp', 'in': [890], 'output_id': 0, 'shape': [400, 400], 'out': [892], 'sorted_id': 891}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[891] = op;
            
            op->set_inputs( forward_result[890] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3591', 'op': 'aten::mul', 'in': [891, 58], 'output_id': 0, 'shape': [400, 400], 'out': [893], 'sorted_id': 892}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[892] = op;
            
            op->set_inputs( forward_result[891] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3592', 'op': 'aten::add', 'in': [881, 892, 12], 'output_id': 0, 'shape': [400, 400], 'out': [894], 'sorted_id': 893}
        {
            Tensor::shape_type shape = {400,400};
            AddOp* op = new AddOp();
            forward_result[893] = op;
            
            op->set_inputs( forward_result[881] );
            op->set_inputs( forward_result[892] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3593', 'op': 'aten::log', 'in': [893], 'output_id': 0, 'shape': [400, 400], 'out': [895], 'sorted_id': 894}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[894] = op;
            
            op->set_inputs( forward_result[893] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3594', 'op': 'aten::sum', 'in': [894, 20], 'output_id': 0, 'shape': [], 'out': [921], 'sorted_id': 895}
        {
            SumOp* op = new SumOp();
            forward_result[895] = op;
            
            op->set_inputs( forward_result[894] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3597', 'op': 'aten::sub', 'in': [869, 555, 12], 'output_id': 0, 'shape': [400], 'out': [897], 'sorted_id': 896}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[896] = op;
            
            op->set_inputs( forward_result[869] );
            op->set_inputs( forward_result[555] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3598', 'op': 'aten::pow', 'in': [896, 45], 'output_id': 0, 'shape': [400], 'out': [898], 'sorted_id': 897}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[897] = op;
            
            op->set_inputs( forward_result[896] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3599', 'op': 'aten::neg', 'in': [897], 'output_id': 0, 'shape': [400], 'out': [901], 'sorted_id': 898}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[898] = op;
            
            op->set_inputs( forward_result[897] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.61', 'op': 'aten::pow', 'in': [530, 45], 'output_id': 0, 'shape': [1], 'out': [900], 'sorted_id': 899}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[899] = op;
            
            op->set_inputs( forward_result[530] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3600', 'op': 'aten::mul', 'in': [899, 50], 'output_id': 0, 'shape': [1], 'out': [901], 'sorted_id': 900}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[900] = op;
            
            op->set_inputs( forward_result[899] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3601', 'op': 'aten::div', 'in': [898, 900], 'output_id': 0, 'shape': [400], 'out': [903], 'sorted_id': 901}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[901] = op;
            
            op->set_inputs( forward_result[898] );
            op->set_inputs( forward_result[900] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.61', 'op': 'aten::log', 'in': [530], 'output_id': 0, 'shape': [1], 'out': [903], 'sorted_id': 902}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[902] = op;
            
            op->set_inputs( forward_result[530] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3602', 'op': 'aten::sub', 'in': [901, 902, 12], 'output_id': 0, 'shape': [400], 'out': [904], 'sorted_id': 903}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[903] = op;
            
            op->set_inputs( forward_result[901] );
            op->set_inputs( forward_result[902] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3603', 'op': 'aten::sub', 'in': [903, 55, 12], 'output_id': 0, 'shape': [400], 'out': [905], 'sorted_id': 904}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[904] = op;
            
            op->set_inputs( forward_result[903] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob1.31', 'op': 'aten::exp', 'in': [904], 'output_id': 0, 'shape': [400], 'out': [906], 'sorted_id': 905}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[905] = op;
            
            op->set_inputs( forward_result[904] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3615', 'op': 'aten::mul', 'in': [905, 58], 'output_id': 0, 'shape': [400], 'out': [918], 'sorted_id': 906}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[906] = op;
            
            op->set_inputs( forward_result[905] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3607', 'op': 'aten::sub', 'in': [869, 567, 12], 'output_id': 0, 'shape': [400], 'out': [908], 'sorted_id': 907}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[907] = op;
            
            op->set_inputs( forward_result[869] );
            op->set_inputs( forward_result[567] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3608', 'op': 'aten::pow', 'in': [907, 45], 'output_id': 0, 'shape': [400], 'out': [909], 'sorted_id': 908}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[908] = op;
            
            op->set_inputs( forward_result[907] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3609', 'op': 'aten::neg', 'in': [908], 'output_id': 0, 'shape': [400], 'out': [912], 'sorted_id': 909}
        {
            Tensor::shape_type shape = {400};
            NegOp* op = new NegOp();
            forward_result[909] = op;
            
            op->set_inputs( forward_result[908] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/var.63', 'op': 'aten::pow', 'in': [543, 45], 'output_id': 0, 'shape': [1], 'out': [911], 'sorted_id': 910}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[910] = op;
            
            op->set_inputs( forward_result[543] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3610', 'op': 'aten::mul', 'in': [910, 50], 'output_id': 0, 'shape': [1], 'out': [912], 'sorted_id': 911}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[911] = op;
            
            op->set_inputs( forward_result[910] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3611', 'op': 'aten::div', 'in': [909, 911], 'output_id': 0, 'shape': [400], 'out': [914], 'sorted_id': 912}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[912] = op;
            
            op->set_inputs( forward_result[909] );
            op->set_inputs( forward_result[911] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/log_scale.63', 'op': 'aten::log', 'in': [543], 'output_id': 0, 'shape': [1], 'out': [914], 'sorted_id': 913}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[913] = op;
            
            op->set_inputs( forward_result[543] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3612', 'op': 'aten::sub', 'in': [912, 913, 12], 'output_id': 0, 'shape': [400], 'out': [915], 'sorted_id': 914}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[914] = op;
            
            op->set_inputs( forward_result[912] );
            op->set_inputs( forward_result[913] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3613', 'op': 'aten::sub', 'in': [914, 55, 12], 'output_id': 0, 'shape': [400], 'out': [916], 'sorted_id': 915}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[915] = op;
            
            op->set_inputs( forward_result[914] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/prob2.31', 'op': 'aten::exp', 'in': [915], 'output_id': 0, 'shape': [400], 'out': [917], 'sorted_id': 916}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[916] = op;
            
            op->set_inputs( forward_result[915] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3616', 'op': 'aten::mul', 'in': [916, 58], 'output_id': 0, 'shape': [400], 'out': [918], 'sorted_id': 917}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[917] = op;
            
            op->set_inputs( forward_result[916] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3617', 'op': 'aten::add', 'in': [906, 917, 12], 'output_id': 0, 'shape': [400], 'out': [919], 'sorted_id': 918}
        {
            Tensor::shape_type shape = {400};
            AddOp* op = new AddOp();
            forward_result[918] = op;
            
            op->set_inputs( forward_result[906] );
            op->set_inputs( forward_result[917] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3618', 'op': 'aten::log', 'in': [918], 'output_id': 0, 'shape': [400], 'out': [920], 'sorted_id': 919}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[919] = op;
            
            op->set_inputs( forward_result[918] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3619', 'op': 'aten::sum', 'in': [919, 20], 'output_id': 0, 'shape': [], 'out': [921], 'sorted_id': 920}
        {
            SumOp* op = new SumOp();
            forward_result[920] = op;
            
            op->set_inputs( forward_result[919] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3620', 'op': 'aten::add', 'in': [895, 920, 12], 'output_id': 0, 'shape': [], 'out': [949], 'sorted_id': 921}
        {
            AddOp* op = new AddOp();
            forward_result[921] = op;
            
            op->set_inputs( forward_result[895] );
            op->set_inputs( forward_result[920] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3621', 'op': 'aten::exp', 'in': [844], 'output_id': 0, 'shape': [400, 400], 'out': [923], 'sorted_id': 922}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[922] = op;
            
            op->set_inputs( forward_result[844] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3622', 'op': 'aten::log1p', 'in': [922], 'output_id': 0, 'shape': [400, 400], 'out': [924], 'sorted_id': 923}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[923] = op;
            
            op->set_inputs( forward_result[922] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3623', 'op': 'aten::log', 'in': [923], 'output_id': 0, 'shape': [400, 400], 'out': [925], 'sorted_id': 924}
        {
            Tensor::shape_type shape = {400,400};
            LogOp* op = new LogOp();
            forward_result[924] = op;
            
            op->set_inputs( forward_result[923] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3624', 'op': 'aten::rsub', 'in': [924, 107, 12], 'output_id': 0, 'shape': [400, 400], 'out': [933], 'sorted_id': 925}
        {
            Tensor::shape_type shape = {400,400};
            RsubOp* op = new RsubOp();
            forward_result[925] = op;
            
            op->set_inputs( forward_result[924] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3625', 'op': 'aten::sub', 'in': [856, 843, 12], 'output_id': 0, 'shape': [400, 400], 'out': [927], 'sorted_id': 926}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[926] = op;
            
            op->set_inputs( forward_result[856] );
            op->set_inputs( forward_result[843] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3626', 'op': 'aten::pow', 'in': [926, 45], 'output_id': 0, 'shape': [400, 400], 'out': [932], 'sorted_id': 927}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[927] = op;
            
            op->set_inputs( forward_result[926] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3627', 'op': 'aten::exp', 'in': [844], 'output_id': 0, 'shape': [400, 400], 'out': [929], 'sorted_id': 928}
        {
            Tensor::shape_type shape = {400,400};
            ExpOp* op = new ExpOp();
            forward_result[928] = op;
            
            op->set_inputs( forward_result[844] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3628', 'op': 'aten::log1p', 'in': [928], 'output_id': 0, 'shape': [400, 400], 'out': [930], 'sorted_id': 929}
        {
            Tensor::shape_type shape = {400,400};
            Log1pOp* op = new Log1pOp();
            forward_result[929] = op;
            
            op->set_inputs( forward_result[928] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3629', 'op': 'aten::pow', 'in': [929, 45], 'output_id': 0, 'shape': [400, 400], 'out': [931], 'sorted_id': 930}
        {
            Tensor::shape_type shape = {400,400};
            PowOp* op = new PowOp();
            forward_result[930] = op;
            
            op->set_inputs( forward_result[929] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3630', 'op': 'aten::mul', 'in': [930, 50], 'output_id': 0, 'shape': [400, 400], 'out': [932], 'sorted_id': 931}
        {
            Tensor::shape_type shape = {400,400};
            MulOp* op = new MulOp();
            forward_result[931] = op;
            
            op->set_inputs( forward_result[930] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3631', 'op': 'aten::div', 'in': [927, 931], 'output_id': 0, 'shape': [400, 400], 'out': [933], 'sorted_id': 932}
        {
            Tensor::shape_type shape = {400,400};
            DivOp* op = new DivOp();
            forward_result[932] = op;
            
            op->set_inputs( forward_result[927] );
            op->set_inputs( forward_result[931] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3632', 'op': 'aten::sub', 'in': [925, 932, 12], 'output_id': 0, 'shape': [400, 400], 'out': [934], 'sorted_id': 933}
        {
            Tensor::shape_type shape = {400,400};
            SubOp* op = new SubOp();
            forward_result[933] = op;
            
            op->set_inputs( forward_result[925] );
            op->set_inputs( forward_result[932] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3633', 'op': 'aten::sum', 'in': [933, 20], 'output_id': 0, 'shape': [], 'out': [948], 'sorted_id': 934}
        {
            SumOp* op = new SumOp();
            forward_result[934] = op;
            
            op->set_inputs( forward_result[933] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3634', 'op': 'aten::exp', 'in': [858], 'output_id': 0, 'shape': [400], 'out': [936], 'sorted_id': 935}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[935] = op;
            
            op->set_inputs( forward_result[858] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3635', 'op': 'aten::log1p', 'in': [935], 'output_id': 0, 'shape': [400], 'out': [937], 'sorted_id': 936}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[936] = op;
            
            op->set_inputs( forward_result[935] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3636', 'op': 'aten::log', 'in': [936], 'output_id': 0, 'shape': [400], 'out': [938], 'sorted_id': 937}
        {
            Tensor::shape_type shape = {400};
            LogOp* op = new LogOp();
            forward_result[937] = op;
            
            op->set_inputs( forward_result[936] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3637', 'op': 'aten::rsub', 'in': [937, 107, 12], 'output_id': 0, 'shape': [400], 'out': [946], 'sorted_id': 938}
        {
            Tensor::shape_type shape = {400};
            RsubOp* op = new RsubOp();
            forward_result[938] = op;
            
            op->set_inputs( forward_result[937] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3638', 'op': 'aten::sub', 'in': [869, 857, 12], 'output_id': 0, 'shape': [400], 'out': [940], 'sorted_id': 939}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[939] = op;
            
            op->set_inputs( forward_result[869] );
            op->set_inputs( forward_result[857] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3639', 'op': 'aten::pow', 'in': [939, 45], 'output_id': 0, 'shape': [400], 'out': [945], 'sorted_id': 940}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[940] = op;
            
            op->set_inputs( forward_result[939] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3640', 'op': 'aten::exp', 'in': [858], 'output_id': 0, 'shape': [400], 'out': [942], 'sorted_id': 941}
        {
            Tensor::shape_type shape = {400};
            ExpOp* op = new ExpOp();
            forward_result[941] = op;
            
            op->set_inputs( forward_result[858] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3641', 'op': 'aten::log1p', 'in': [941], 'output_id': 0, 'shape': [400], 'out': [943], 'sorted_id': 942}
        {
            Tensor::shape_type shape = {400};
            Log1pOp* op = new Log1pOp();
            forward_result[942] = op;
            
            op->set_inputs( forward_result[941] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3642', 'op': 'aten::pow', 'in': [942, 45], 'output_id': 0, 'shape': [400], 'out': [944], 'sorted_id': 943}
        {
            Tensor::shape_type shape = {400};
            PowOp* op = new PowOp();
            forward_result[943] = op;
            
            op->set_inputs( forward_result[942] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3643', 'op': 'aten::mul', 'in': [943, 50], 'output_id': 0, 'shape': [400], 'out': [945], 'sorted_id': 944}
        {
            Tensor::shape_type shape = {400};
            MulOp* op = new MulOp();
            forward_result[944] = op;
            
            op->set_inputs( forward_result[943] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3644', 'op': 'aten::div', 'in': [940, 944], 'output_id': 0, 'shape': [400], 'out': [946], 'sorted_id': 945}
        {
            Tensor::shape_type shape = {400};
            DivOp* op = new DivOp();
            forward_result[945] = op;
            
            op->set_inputs( forward_result[940] );
            op->set_inputs( forward_result[944] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3645', 'op': 'aten::sub', 'in': [938, 945, 12], 'output_id': 0, 'shape': [400], 'out': [947], 'sorted_id': 946}
        {
            Tensor::shape_type shape = {400};
            SubOp* op = new SubOp();
            forward_result[946] = op;
            
            op->set_inputs( forward_result[938] );
            op->set_inputs( forward_result[945] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3646', 'op': 'aten::sum', 'in': [946, 20], 'output_id': 0, 'shape': [], 'out': [948], 'sorted_id': 947}
        {
            SumOp* op = new SumOp();
            forward_result[947] = op;
            
            op->set_inputs( forward_result[946] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l2]/3647', 'op': 'aten::add', 'in': [934, 947, 12], 'output_id': 0, 'shape': [], 'out': [949], 'sorted_id': 948}
        {
            AddOp* op = new AddOp();
            forward_result[948] = op;
            
            op->set_inputs( forward_result[934] );
            op->set_inputs( forward_result[947] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3649', 'op': 'prim::TupleConstruct', 'in': [870, 921, 948], 'output_id': 0, 'shape': [], 'out': [950, 1076, 1094], 'sorted_id': 949}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[949] = op;
            
            op->set_inputs( forward_result[870] );
            op->set_inputs( forward_result[921] );
            op->set_inputs( forward_result[948] );
        }
        
        // {'name': 'Model/3650', 'op': 'prim::TupleUnpack', 'in': [949], 'output_id': 0, 'shape': [4, 400], 'out': [951], 'sorted_id': 950}
        {
            Tensor::shape_type shape = {4,400};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[950] = op;
            
            op->set_inputs( forward_result[949] );
        }
        
        // {'name': 'Model/Net[net]/input.33', 'op': 'aten::relu', 'in': [950], 'output_id': 0, 'shape': [4, 400], 'out': [979], 'sorted_id': 951}
        {
            Tensor::shape_type shape = {4,400};
            ReluOp* op = new ReluOp();
            forward_result[951] = op;
            
            op->set_inputs( forward_result[950] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_mu/weight_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [965, 1035], 'sorted_id': 952}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_mu.reshape( shape );
            forward_result[952] = new VariableTensor( l3_weight_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/weight_rho/weight_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [957, 954, 1031, 1037, 956], 'sorted_id': 953}
        {
            Tensor::shape_type shape = {10,400};
            l3_weight_rho.reshape( shape );
            forward_result[953] = new VariableTensor( l3_weight_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3666', 'op': 'aten::exp', 'in': [953], 'output_id': 0, 'shape': [10, 400], 'out': [955], 'sorted_id': 954}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[954] = op;
            
            op->set_inputs( forward_result[953] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3667', 'op': 'aten::log1p', 'in': [954], 'output_id': 0, 'shape': [10, 400], 'out': [964], 'sorted_id': 955}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[955] = op;
            
            op->set_inputs( forward_result[954] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3658', 'op': 'aten::size', 'in': [953, 10], 'output_id': 0, 'shape': [], 'out': [960, 958], 'sorted_id': 956}
        {
            SizeOp* op = new SizeOp();
            forward_result[956] = op;
            
            op->set_inputs( forward_result[953] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3659', 'op': 'aten::size', 'in': [953, 12], 'output_id': 0, 'shape': [], 'out': [960, 958], 'sorted_id': 957}
        {
            SizeOp* op = new SizeOp();
            forward_result[957] = op;
            
            op->set_inputs( forward_result[953] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3660', 'op': 'prim::ListConstruct', 'in': [956, 957], 'output_id': 0, 'shape': [], 'out': [959], 'sorted_id': 958}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[958] = op;
            
            op->set_inputs( forward_result[956] );
            op->set_inputs( forward_result[957] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3661', 'op': 'aten::expand', 'in': [617, 958, 15], 'output_id': 0, 'shape': [10, 400], 'out': [962], 'sorted_id': 959}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[959] = op;
            
            op->set_inputs( forward_result[617] );
            op->set_inputs( forward_result[958] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3662', 'op': 'prim::ListConstruct', 'in': [956, 957], 'output_id': 0, 'shape': [], 'out': [961], 'sorted_id': 960}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[960] = op;
            
            op->set_inputs( forward_result[956] );
            op->set_inputs( forward_result[957] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3663', 'op': 'aten::expand', 'in': [622, 960, 15], 'output_id': 0, 'shape': [10, 400], 'out': [962], 'sorted_id': 961}
        {
            Tensor::shape_type shape = {10,400};
            ExpandOp* op = new ExpandOp();
            forward_result[961] = op;
            
            op->set_inputs( forward_result[622] );
            op->set_inputs( forward_result[960] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3664', 'op': 'aten::normal', 'in': [959, 961, 20], 'output_id': 0, 'shape': [10, 400], 'out': [963], 'sorted_id': 962}
        {
            Tensor::shape_type shape = {10,400};
            NormalOp* op = new NormalOp();
            forward_result[962] = op;
            
            op->set_inputs( forward_result[959] );
            op->set_inputs( forward_result[961] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon.33', 'op': 'aten::to', 'in': [962, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10, 400], 'out': [964], 'sorted_id': 963}
        {
            Tensor::shape_type shape = {10,400};
            ToOp* op = new ToOp();
            forward_result[963] = op;
            
            op->set_inputs( forward_result[962] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3668', 'op': 'aten::mul', 'in': [955, 963], 'output_id': 0, 'shape': [10, 400], 'out': [965], 'sorted_id': 964}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[964] = op;
            
            op->set_inputs( forward_result[955] );
            op->set_inputs( forward_result[963] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value.33', 'op': 'aten::add', 'in': [952, 964, 12], 'output_id': 0, 'shape': [10, 400], 'out': [991, 980, 1035, 979], 'sorted_id': 965}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[965] = op;
            
            op->set_inputs( forward_result[952] );
            op->set_inputs( forward_result[964] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_mu/bias_mu', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [978, 1048], 'sorted_id': 966}
        {
            Tensor::shape_type shape = {10};
            l3_bias_mu.reshape( shape );
            forward_result[966] = new VariableTensor( l3_bias_mu, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/bias_rho/bias_rho', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [970, 1050, 1044, 968], 'sorted_id': 967}
        {
            Tensor::shape_type shape = {10};
            l3_bias_rho.reshape( shape );
            forward_result[967] = new VariableTensor( l3_bias_rho, 2 );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3677', 'op': 'aten::exp', 'in': [967], 'output_id': 0, 'shape': [10], 'out': [969], 'sorted_id': 968}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[968] = op;
            
            op->set_inputs( forward_result[967] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3678', 'op': 'aten::log1p', 'in': [968], 'output_id': 0, 'shape': [10], 'out': [977], 'sorted_id': 969}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[969] = op;
            
            op->set_inputs( forward_result[968] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3670', 'op': 'aten::size', 'in': [967, 10], 'output_id': 0, 'shape': [], 'out': [973, 971], 'sorted_id': 970}
        {
            SizeOp* op = new SizeOp();
            forward_result[970] = op;
            
            op->set_inputs( forward_result[967] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3671', 'op': 'prim::ListConstruct', 'in': [970], 'output_id': 0, 'shape': [], 'out': [972], 'sorted_id': 971}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[971] = op;
            
            op->set_inputs( forward_result[970] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3672', 'op': 'aten::expand', 'in': [633, 971, 15], 'output_id': 0, 'shape': [10], 'out': [975], 'sorted_id': 972}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[972] = op;
            
            op->set_inputs( forward_result[633] );
            op->set_inputs( forward_result[971] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3673', 'op': 'prim::ListConstruct', 'in': [970], 'output_id': 0, 'shape': [], 'out': [974], 'sorted_id': 973}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[973] = op;
            
            op->set_inputs( forward_result[970] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3674', 'op': 'aten::expand', 'in': [637, 973, 15], 'output_id': 0, 'shape': [10], 'out': [975], 'sorted_id': 974}
        {
            Tensor::shape_type shape = {10};
            ExpandOp* op = new ExpandOp();
            forward_result[974] = op;
            
            op->set_inputs( forward_result[637] );
            op->set_inputs( forward_result[973] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3675', 'op': 'aten::normal', 'in': [972, 974, 20], 'output_id': 0, 'shape': [10], 'out': [976], 'sorted_id': 975}
        {
            Tensor::shape_type shape = {10};
            NormalOp* op = new NormalOp();
            forward_result[975] = op;
            
            op->set_inputs( forward_result[972] );
            op->set_inputs( forward_result[974] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/epsilon', 'op': 'aten::to', 'in': [975, 22, 10, 23, 20, 15, 15, 20], 'output_id': 0, 'shape': [10], 'out': [977], 'sorted_id': 976}
        {
            Tensor::shape_type shape = {10};
            ToOp* op = new ToOp();
            forward_result[976] = op;
            
            op->set_inputs( forward_result[975] );
            op->set_inputs( forward_result[22] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[23] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3679', 'op': 'aten::mul', 'in': [969, 976], 'output_id': 0, 'shape': [10], 'out': [978], 'sorted_id': 977}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[977] = op;
            
            op->set_inputs( forward_result[969] );
            op->set_inputs( forward_result[976] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/value', 'op': 'aten::add', 'in': [966, 977, 12], 'output_id': 0, 'shape': [10], 'out': [1016, 1048, 1005, 979], 'sorted_id': 978}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[978] = op;
            
            op->set_inputs( forward_result[966] );
            op->set_inputs( forward_result[977] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/input.35', 'op': 'aten::linear', 'in': [951, 965, 978], 'output_id': 0, 'shape': [4, 10], 'out': [1058], 'sorted_id': 979}
        {
            Tensor::shape_type shape = {4,10};
            LinearOp* op = new LinearOp();
            forward_result[979] = op;
            
            op->set_inputs( forward_result[951] );
            op->set_inputs( forward_result[965] );
            op->set_inputs( forward_result[978] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3683', 'op': 'aten::sub', 'in': [965, 645, 12], 'output_id': 0, 'shape': [10, 400], 'out': [981], 'sorted_id': 980}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[980] = op;
            
            op->set_inputs( forward_result[965] );
            op->set_inputs( forward_result[645] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3684', 'op': 'aten::pow', 'in': [980, 45], 'output_id': 0, 'shape': [10, 400], 'out': [982], 'sorted_id': 981}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[981] = op;
            
            op->set_inputs( forward_result[980] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3685', 'op': 'aten::neg', 'in': [981], 'output_id': 0, 'shape': [10, 400], 'out': [985], 'sorted_id': 982}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[982] = op;
            
            op->set_inputs( forward_result[981] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.65', 'op': 'aten::pow', 'in': [649, 45], 'output_id': 0, 'shape': [1], 'out': [984], 'sorted_id': 983}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[983] = op;
            
            op->set_inputs( forward_result[649] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3686', 'op': 'aten::mul', 'in': [983, 50], 'output_id': 0, 'shape': [1], 'out': [985], 'sorted_id': 984}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[984] = op;
            
            op->set_inputs( forward_result[983] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3687', 'op': 'aten::div', 'in': [982, 984], 'output_id': 0, 'shape': [10, 400], 'out': [987], 'sorted_id': 985}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[985] = op;
            
            op->set_inputs( forward_result[982] );
            op->set_inputs( forward_result[984] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.65', 'op': 'aten::log', 'in': [649], 'output_id': 0, 'shape': [1], 'out': [987], 'sorted_id': 986}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[986] = op;
            
            op->set_inputs( forward_result[649] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3688', 'op': 'aten::sub', 'in': [985, 986, 12], 'output_id': 0, 'shape': [10, 400], 'out': [988], 'sorted_id': 987}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[987] = op;
            
            op->set_inputs( forward_result[985] );
            op->set_inputs( forward_result[986] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3689', 'op': 'aten::sub', 'in': [987, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [989], 'sorted_id': 988}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[988] = op;
            
            op->set_inputs( forward_result[987] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1.33', 'op': 'aten::exp', 'in': [988], 'output_id': 0, 'shape': [10, 400], 'out': [990], 'sorted_id': 989}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[989] = op;
            
            op->set_inputs( forward_result[988] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3701', 'op': 'aten::mul', 'in': [989, 58], 'output_id': 0, 'shape': [10, 400], 'out': [1002], 'sorted_id': 990}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[990] = op;
            
            op->set_inputs( forward_result[989] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3693', 'op': 'aten::sub', 'in': [965, 658, 12], 'output_id': 0, 'shape': [10, 400], 'out': [992], 'sorted_id': 991}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[991] = op;
            
            op->set_inputs( forward_result[965] );
            op->set_inputs( forward_result[658] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3694', 'op': 'aten::pow', 'in': [991, 45], 'output_id': 0, 'shape': [10, 400], 'out': [993], 'sorted_id': 992}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[992] = op;
            
            op->set_inputs( forward_result[991] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3695', 'op': 'aten::neg', 'in': [992], 'output_id': 0, 'shape': [10, 400], 'out': [996], 'sorted_id': 993}
        {
            Tensor::shape_type shape = {10,400};
            NegOp* op = new NegOp();
            forward_result[993] = op;
            
            op->set_inputs( forward_result[992] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.67', 'op': 'aten::pow', 'in': [662, 45], 'output_id': 0, 'shape': [1], 'out': [995], 'sorted_id': 994}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[994] = op;
            
            op->set_inputs( forward_result[662] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3696', 'op': 'aten::mul', 'in': [994, 50], 'output_id': 0, 'shape': [1], 'out': [996], 'sorted_id': 995}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[995] = op;
            
            op->set_inputs( forward_result[994] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3697', 'op': 'aten::div', 'in': [993, 995], 'output_id': 0, 'shape': [10, 400], 'out': [998], 'sorted_id': 996}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[996] = op;
            
            op->set_inputs( forward_result[993] );
            op->set_inputs( forward_result[995] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.67', 'op': 'aten::log', 'in': [662], 'output_id': 0, 'shape': [1], 'out': [998], 'sorted_id': 997}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[997] = op;
            
            op->set_inputs( forward_result[662] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3698', 'op': 'aten::sub', 'in': [996, 997, 12], 'output_id': 0, 'shape': [10, 400], 'out': [999], 'sorted_id': 998}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[998] = op;
            
            op->set_inputs( forward_result[996] );
            op->set_inputs( forward_result[997] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3699', 'op': 'aten::sub', 'in': [998, 55, 12], 'output_id': 0, 'shape': [10, 400], 'out': [1000], 'sorted_id': 999}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[999] = op;
            
            op->set_inputs( forward_result[998] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2.33', 'op': 'aten::exp', 'in': [999], 'output_id': 0, 'shape': [10, 400], 'out': [1001], 'sorted_id': 1000}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[1000] = op;
            
            op->set_inputs( forward_result[999] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3702', 'op': 'aten::mul', 'in': [1000, 58], 'output_id': 0, 'shape': [10, 400], 'out': [1002], 'sorted_id': 1001}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[1001] = op;
            
            op->set_inputs( forward_result[1000] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3703', 'op': 'aten::add', 'in': [990, 1001, 12], 'output_id': 0, 'shape': [10, 400], 'out': [1003], 'sorted_id': 1002}
        {
            Tensor::shape_type shape = {10,400};
            AddOp* op = new AddOp();
            forward_result[1002] = op;
            
            op->set_inputs( forward_result[990] );
            op->set_inputs( forward_result[1001] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3704', 'op': 'aten::log', 'in': [1002], 'output_id': 0, 'shape': [10, 400], 'out': [1004], 'sorted_id': 1003}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[1003] = op;
            
            op->set_inputs( forward_result[1002] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3705', 'op': 'aten::sum', 'in': [1003, 20], 'output_id': 0, 'shape': [], 'out': [1030], 'sorted_id': 1004}
        {
            SumOp* op = new SumOp();
            forward_result[1004] = op;
            
            op->set_inputs( forward_result[1003] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3708', 'op': 'aten::sub', 'in': [978, 674, 12], 'output_id': 0, 'shape': [10], 'out': [1006], 'sorted_id': 1005}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1005] = op;
            
            op->set_inputs( forward_result[978] );
            op->set_inputs( forward_result[674] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3709', 'op': 'aten::pow', 'in': [1005, 45], 'output_id': 0, 'shape': [10], 'out': [1007], 'sorted_id': 1006}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[1006] = op;
            
            op->set_inputs( forward_result[1005] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3710', 'op': 'aten::neg', 'in': [1006], 'output_id': 0, 'shape': [10], 'out': [1010], 'sorted_id': 1007}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[1007] = op;
            
            op->set_inputs( forward_result[1006] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var.69', 'op': 'aten::pow', 'in': [649, 45], 'output_id': 0, 'shape': [1], 'out': [1009], 'sorted_id': 1008}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[1008] = op;
            
            op->set_inputs( forward_result[649] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3711', 'op': 'aten::mul', 'in': [1008, 50], 'output_id': 0, 'shape': [1], 'out': [1010], 'sorted_id': 1009}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[1009] = op;
            
            op->set_inputs( forward_result[1008] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3712', 'op': 'aten::div', 'in': [1007, 1009], 'output_id': 0, 'shape': [10], 'out': [1012], 'sorted_id': 1010}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[1010] = op;
            
            op->set_inputs( forward_result[1007] );
            op->set_inputs( forward_result[1009] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale.69', 'op': 'aten::log', 'in': [649], 'output_id': 0, 'shape': [1], 'out': [1012], 'sorted_id': 1011}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[1011] = op;
            
            op->set_inputs( forward_result[649] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3713', 'op': 'aten::sub', 'in': [1010, 1011, 12], 'output_id': 0, 'shape': [10], 'out': [1013], 'sorted_id': 1012}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1012] = op;
            
            op->set_inputs( forward_result[1010] );
            op->set_inputs( forward_result[1011] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3714', 'op': 'aten::sub', 'in': [1012, 55, 12], 'output_id': 0, 'shape': [10], 'out': [1014], 'sorted_id': 1013}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1013] = op;
            
            op->set_inputs( forward_result[1012] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob1', 'op': 'aten::exp', 'in': [1013], 'output_id': 0, 'shape': [10], 'out': [1015], 'sorted_id': 1014}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[1014] = op;
            
            op->set_inputs( forward_result[1013] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3726', 'op': 'aten::mul', 'in': [1014, 58], 'output_id': 0, 'shape': [10], 'out': [1027], 'sorted_id': 1015}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[1015] = op;
            
            op->set_inputs( forward_result[1014] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3718', 'op': 'aten::sub', 'in': [978, 686, 12], 'output_id': 0, 'shape': [10], 'out': [1017], 'sorted_id': 1016}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1016] = op;
            
            op->set_inputs( forward_result[978] );
            op->set_inputs( forward_result[686] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3719', 'op': 'aten::pow', 'in': [1016, 45], 'output_id': 0, 'shape': [10], 'out': [1018], 'sorted_id': 1017}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[1017] = op;
            
            op->set_inputs( forward_result[1016] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3720', 'op': 'aten::neg', 'in': [1017], 'output_id': 0, 'shape': [10], 'out': [1021], 'sorted_id': 1018}
        {
            Tensor::shape_type shape = {10};
            NegOp* op = new NegOp();
            forward_result[1018] = op;
            
            op->set_inputs( forward_result[1017] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/var', 'op': 'aten::pow', 'in': [662, 45], 'output_id': 0, 'shape': [1], 'out': [1020], 'sorted_id': 1019}
        {
            Tensor::shape_type shape = {1};
            PowOp* op = new PowOp();
            forward_result[1019] = op;
            
            op->set_inputs( forward_result[662] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3721', 'op': 'aten::mul', 'in': [1019, 50], 'output_id': 0, 'shape': [1], 'out': [1021], 'sorted_id': 1020}
        {
            Tensor::shape_type shape = {1};
            MulOp* op = new MulOp();
            forward_result[1020] = op;
            
            op->set_inputs( forward_result[1019] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3722', 'op': 'aten::div', 'in': [1018, 1020], 'output_id': 0, 'shape': [10], 'out': [1023], 'sorted_id': 1021}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[1021] = op;
            
            op->set_inputs( forward_result[1018] );
            op->set_inputs( forward_result[1020] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/log_scale', 'op': 'aten::log', 'in': [662], 'output_id': 0, 'shape': [1], 'out': [1023], 'sorted_id': 1022}
        {
            Tensor::shape_type shape = {1};
            LogOp* op = new LogOp();
            forward_result[1022] = op;
            
            op->set_inputs( forward_result[662] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3723', 'op': 'aten::sub', 'in': [1021, 1022, 12], 'output_id': 0, 'shape': [10], 'out': [1024], 'sorted_id': 1023}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1023] = op;
            
            op->set_inputs( forward_result[1021] );
            op->set_inputs( forward_result[1022] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3724', 'op': 'aten::sub', 'in': [1023, 55, 12], 'output_id': 0, 'shape': [10], 'out': [1025], 'sorted_id': 1024}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1024] = op;
            
            op->set_inputs( forward_result[1023] );
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/prob2', 'op': 'aten::exp', 'in': [1024], 'output_id': 0, 'shape': [10], 'out': [1026], 'sorted_id': 1025}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[1025] = op;
            
            op->set_inputs( forward_result[1024] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3727', 'op': 'aten::mul', 'in': [1025, 58], 'output_id': 0, 'shape': [10], 'out': [1027], 'sorted_id': 1026}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[1026] = op;
            
            op->set_inputs( forward_result[1025] );
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3728', 'op': 'aten::add', 'in': [1015, 1026, 12], 'output_id': 0, 'shape': [10], 'out': [1028], 'sorted_id': 1027}
        {
            Tensor::shape_type shape = {10};
            AddOp* op = new AddOp();
            forward_result[1027] = op;
            
            op->set_inputs( forward_result[1015] );
            op->set_inputs( forward_result[1026] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3729', 'op': 'aten::log', 'in': [1027], 'output_id': 0, 'shape': [10], 'out': [1029], 'sorted_id': 1028}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[1028] = op;
            
            op->set_inputs( forward_result[1027] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3730', 'op': 'aten::sum', 'in': [1028, 20], 'output_id': 0, 'shape': [], 'out': [1030], 'sorted_id': 1029}
        {
            SumOp* op = new SumOp();
            forward_result[1029] = op;
            
            op->set_inputs( forward_result[1028] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3731', 'op': 'aten::add', 'in': [1004, 1029, 12], 'output_id': 0, 'shape': [], 'out': [1058], 'sorted_id': 1030}
        {
            AddOp* op = new AddOp();
            forward_result[1030] = op;
            
            op->set_inputs( forward_result[1004] );
            op->set_inputs( forward_result[1029] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3732', 'op': 'aten::exp', 'in': [953], 'output_id': 0, 'shape': [10, 400], 'out': [1032], 'sorted_id': 1031}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[1031] = op;
            
            op->set_inputs( forward_result[953] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3733', 'op': 'aten::log1p', 'in': [1031], 'output_id': 0, 'shape': [10, 400], 'out': [1033], 'sorted_id': 1032}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[1032] = op;
            
            op->set_inputs( forward_result[1031] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3734', 'op': 'aten::log', 'in': [1032], 'output_id': 0, 'shape': [10, 400], 'out': [1034], 'sorted_id': 1033}
        {
            Tensor::shape_type shape = {10,400};
            LogOp* op = new LogOp();
            forward_result[1033] = op;
            
            op->set_inputs( forward_result[1032] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3735', 'op': 'aten::rsub', 'in': [1033, 107, 12], 'output_id': 0, 'shape': [10, 400], 'out': [1042], 'sorted_id': 1034}
        {
            Tensor::shape_type shape = {10,400};
            RsubOp* op = new RsubOp();
            forward_result[1034] = op;
            
            op->set_inputs( forward_result[1033] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3736', 'op': 'aten::sub', 'in': [965, 952, 12], 'output_id': 0, 'shape': [10, 400], 'out': [1036], 'sorted_id': 1035}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[1035] = op;
            
            op->set_inputs( forward_result[965] );
            op->set_inputs( forward_result[952] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3737', 'op': 'aten::pow', 'in': [1035, 45], 'output_id': 0, 'shape': [10, 400], 'out': [1041], 'sorted_id': 1036}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[1036] = op;
            
            op->set_inputs( forward_result[1035] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3738', 'op': 'aten::exp', 'in': [953], 'output_id': 0, 'shape': [10, 400], 'out': [1038], 'sorted_id': 1037}
        {
            Tensor::shape_type shape = {10,400};
            ExpOp* op = new ExpOp();
            forward_result[1037] = op;
            
            op->set_inputs( forward_result[953] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3739', 'op': 'aten::log1p', 'in': [1037], 'output_id': 0, 'shape': [10, 400], 'out': [1039], 'sorted_id': 1038}
        {
            Tensor::shape_type shape = {10,400};
            Log1pOp* op = new Log1pOp();
            forward_result[1038] = op;
            
            op->set_inputs( forward_result[1037] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3740', 'op': 'aten::pow', 'in': [1038, 45], 'output_id': 0, 'shape': [10, 400], 'out': [1040], 'sorted_id': 1039}
        {
            Tensor::shape_type shape = {10,400};
            PowOp* op = new PowOp();
            forward_result[1039] = op;
            
            op->set_inputs( forward_result[1038] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3741', 'op': 'aten::mul', 'in': [1039, 50], 'output_id': 0, 'shape': [10, 400], 'out': [1041], 'sorted_id': 1040}
        {
            Tensor::shape_type shape = {10,400};
            MulOp* op = new MulOp();
            forward_result[1040] = op;
            
            op->set_inputs( forward_result[1039] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3742', 'op': 'aten::div', 'in': [1036, 1040], 'output_id': 0, 'shape': [10, 400], 'out': [1042], 'sorted_id': 1041}
        {
            Tensor::shape_type shape = {10,400};
            DivOp* op = new DivOp();
            forward_result[1041] = op;
            
            op->set_inputs( forward_result[1036] );
            op->set_inputs( forward_result[1040] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3743', 'op': 'aten::sub', 'in': [1034, 1041, 12], 'output_id': 0, 'shape': [10, 400], 'out': [1043], 'sorted_id': 1042}
        {
            Tensor::shape_type shape = {10,400};
            SubOp* op = new SubOp();
            forward_result[1042] = op;
            
            op->set_inputs( forward_result[1034] );
            op->set_inputs( forward_result[1041] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3744', 'op': 'aten::sum', 'in': [1042, 20], 'output_id': 0, 'shape': [], 'out': [1057], 'sorted_id': 1043}
        {
            SumOp* op = new SumOp();
            forward_result[1043] = op;
            
            op->set_inputs( forward_result[1042] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3745', 'op': 'aten::exp', 'in': [967], 'output_id': 0, 'shape': [10], 'out': [1045], 'sorted_id': 1044}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[1044] = op;
            
            op->set_inputs( forward_result[967] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3746', 'op': 'aten::log1p', 'in': [1044], 'output_id': 0, 'shape': [10], 'out': [1046], 'sorted_id': 1045}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[1045] = op;
            
            op->set_inputs( forward_result[1044] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3747', 'op': 'aten::log', 'in': [1045], 'output_id': 0, 'shape': [10], 'out': [1047], 'sorted_id': 1046}
        {
            Tensor::shape_type shape = {10};
            LogOp* op = new LogOp();
            forward_result[1046] = op;
            
            op->set_inputs( forward_result[1045] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3748', 'op': 'aten::rsub', 'in': [1046, 107, 12], 'output_id': 0, 'shape': [10], 'out': [1055], 'sorted_id': 1047}
        {
            Tensor::shape_type shape = {10};
            RsubOp* op = new RsubOp();
            forward_result[1047] = op;
            
            op->set_inputs( forward_result[1046] );
            op->set_inputs( forward_result[107] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3749', 'op': 'aten::sub', 'in': [978, 966, 12], 'output_id': 0, 'shape': [10], 'out': [1049], 'sorted_id': 1048}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1048] = op;
            
            op->set_inputs( forward_result[978] );
            op->set_inputs( forward_result[966] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3750', 'op': 'aten::pow', 'in': [1048, 45], 'output_id': 0, 'shape': [10], 'out': [1054], 'sorted_id': 1049}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[1049] = op;
            
            op->set_inputs( forward_result[1048] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3751', 'op': 'aten::exp', 'in': [967], 'output_id': 0, 'shape': [10], 'out': [1051], 'sorted_id': 1050}
        {
            Tensor::shape_type shape = {10};
            ExpOp* op = new ExpOp();
            forward_result[1050] = op;
            
            op->set_inputs( forward_result[967] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3752', 'op': 'aten::log1p', 'in': [1050], 'output_id': 0, 'shape': [10], 'out': [1052], 'sorted_id': 1051}
        {
            Tensor::shape_type shape = {10};
            Log1pOp* op = new Log1pOp();
            forward_result[1051] = op;
            
            op->set_inputs( forward_result[1050] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3753', 'op': 'aten::pow', 'in': [1051, 45], 'output_id': 0, 'shape': [10], 'out': [1053], 'sorted_id': 1052}
        {
            Tensor::shape_type shape = {10};
            PowOp* op = new PowOp();
            forward_result[1052] = op;
            
            op->set_inputs( forward_result[1051] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3754', 'op': 'aten::mul', 'in': [1052, 50], 'output_id': 0, 'shape': [10], 'out': [1054], 'sorted_id': 1053}
        {
            Tensor::shape_type shape = {10};
            MulOp* op = new MulOp();
            forward_result[1053] = op;
            
            op->set_inputs( forward_result[1052] );
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3755', 'op': 'aten::div', 'in': [1049, 1053], 'output_id': 0, 'shape': [10], 'out': [1055], 'sorted_id': 1054}
        {
            Tensor::shape_type shape = {10};
            DivOp* op = new DivOp();
            forward_result[1054] = op;
            
            op->set_inputs( forward_result[1049] );
            op->set_inputs( forward_result[1053] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3756', 'op': 'aten::sub', 'in': [1047, 1054, 12], 'output_id': 0, 'shape': [10], 'out': [1056], 'sorted_id': 1055}
        {
            Tensor::shape_type shape = {10};
            SubOp* op = new SubOp();
            forward_result[1055] = op;
            
            op->set_inputs( forward_result[1047] );
            op->set_inputs( forward_result[1054] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3757', 'op': 'aten::sum', 'in': [1055, 20], 'output_id': 0, 'shape': [], 'out': [1057], 'sorted_id': 1056}
        {
            SumOp* op = new SumOp();
            forward_result[1056] = op;
            
            op->set_inputs( forward_result[1055] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/BayesianLinear[l3]/3758', 'op': 'aten::add', 'in': [1043, 1056, 12], 'output_id': 0, 'shape': [], 'out': [1058], 'sorted_id': 1057}
        {
            AddOp* op = new AddOp();
            forward_result[1057] = op;
            
            op->set_inputs( forward_result[1043] );
            op->set_inputs( forward_result[1056] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3760', 'op': 'prim::TupleConstruct', 'in': [979, 1030, 1057], 'output_id': 0, 'shape': [], 'out': [1078, 1059, 1096], 'sorted_id': 1058}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[1058] = op;
            
            op->set_inputs( forward_result[979] );
            op->set_inputs( forward_result[1030] );
            op->set_inputs( forward_result[1057] );
        }
        
        // {'name': 'Model/3761', 'op': 'prim::TupleUnpack', 'in': [1058], 'output_id': 0, 'shape': [4, 10], 'out': [1060], 'sorted_id': 1059}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[1059] = op;
            
            op->set_inputs( forward_result[1058] );
        }
        
        // {'name': 'Model/Net[net]/3764', 'op': 'aten::log_softmax', 'in': [1059, 12, 20], 'output_id': 0, 'shape': [4, 10], 'out': [1061], 'sorted_id': 1060}
        {
            Tensor::shape_type shape = {4,10};
            LogSoftmaxOp* op = new LogSoftmaxOp();
            forward_result[1060] = op;
            
            op->set_inputs( forward_result[1059] );
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/Net[net]/3769', 'op': 'prim::ListConstruct', 'in': [372, 731, 1060], 'output_id': 0, 'shape': [], 'out': [1062], 'sorted_id': 1061}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[1061] = op;
            
            op->set_inputs( forward_result[372] );
            op->set_inputs( forward_result[731] );
            op->set_inputs( forward_result[1060] );
        }
        
        // {'name': 'Model/Net[net]/outputs', 'op': 'aten::stack', 'in': [1061, 10], 'output_id': 0, 'shape': [3, 4, 10], 'out': [1064], 'sorted_id': 1062}
        {
            Tensor::shape_type shape = {3,4,10};
            StackOp* op = new StackOp();
            forward_result[1062] = op;
            
            op->set_inputs( forward_result[1061] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/3775', 'op': 'prim::ListConstruct', 'in': [10], 'output_id': 0, 'shape': [], 'out': [1064], 'sorted_id': 1063}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[1063] = op;
            
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/input', 'op': 'aten::mean', 'in': [1062, 1063, 15, 20], 'output_id': 0, 'shape': [4, 10], 'out': [1101], 'sorted_id': 1064}
        {
            Tensor::shape_type shape = {4,10};
            MeanOp* op = new MeanOp();
            forward_result[1064] = op;
            
            op->set_inputs( forward_result[1062] );
            op->set_inputs( forward_result[1063] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/2813', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 2, 'shape': [], 'out': [1067], 'sorted_id': 1065}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1065] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/2944', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 2, 'shape': [], 'out': [1067], 'sorted_id': 1066}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1066] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/3089', 'op': 'aten::add', 'in': [1065, 1066, 12], 'output_id': 0, 'shape': [], 'out': [1069], 'sorted_id': 1067}
        {
            AddOp* op = new AddOp();
            forward_result[1067] = op;
            
            op->set_inputs( forward_result[1065] );
            op->set_inputs( forward_result[1066] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3075', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 2, 'shape': [], 'out': [1069], 'sorted_id': 1068}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1068] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/3090', 'op': 'aten::add', 'in': [1067, 1068, 12], 'output_id': 0, 'shape': [], 'out': [1080], 'sorted_id': 1069}
        {
            AddOp* op = new AddOp();
            forward_result[1069] = op;
            
            op->set_inputs( forward_result[1067] );
            op->set_inputs( forward_result[1068] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3202', 'op': 'prim::TupleUnpack', 'in': [491], 'output_id': 2, 'shape': [], 'out': [1072], 'sorted_id': 1070}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1070] = op;
            
            op->set_inputs( forward_result[491] );
        }
        
        // {'name': 'Model/3313', 'op': 'prim::TupleUnpack', 'in': [610], 'output_id': 2, 'shape': [], 'out': [1072], 'sorted_id': 1071}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1071] = op;
            
            op->set_inputs( forward_result[610] );
        }
        
        // {'name': 'Model/Net[net]/3428', 'op': 'aten::add', 'in': [1070, 1071, 12], 'output_id': 0, 'shape': [], 'out': [1074], 'sorted_id': 1072}
        {
            AddOp* op = new AddOp();
            forward_result[1072] = op;
            
            op->set_inputs( forward_result[1070] );
            op->set_inputs( forward_result[1071] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3424', 'op': 'prim::TupleUnpack', 'in': [729], 'output_id': 2, 'shape': [], 'out': [1074], 'sorted_id': 1073}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1073] = op;
            
            op->set_inputs( forward_result[729] );
        }
        
        // {'name': 'Model/Net[net]/3429', 'op': 'aten::add', 'in': [1072, 1073, 12], 'output_id': 0, 'shape': [], 'out': [1080], 'sorted_id': 1074}
        {
            AddOp* op = new AddOp();
            forward_result[1074] = op;
            
            op->set_inputs( forward_result[1072] );
            op->set_inputs( forward_result[1073] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3541', 'op': 'prim::TupleUnpack', 'in': [840], 'output_id': 2, 'shape': [], 'out': [1077], 'sorted_id': 1075}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1075] = op;
            
            op->set_inputs( forward_result[840] );
        }
        
        // {'name': 'Model/3652', 'op': 'prim::TupleUnpack', 'in': [949], 'output_id': 2, 'shape': [], 'out': [1077], 'sorted_id': 1076}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1076] = op;
            
            op->set_inputs( forward_result[949] );
        }
        
        // {'name': 'Model/Net[net]/3767', 'op': 'aten::add', 'in': [1075, 1076, 12], 'output_id': 0, 'shape': [], 'out': [1079], 'sorted_id': 1077}
        {
            AddOp* op = new AddOp();
            forward_result[1077] = op;
            
            op->set_inputs( forward_result[1075] );
            op->set_inputs( forward_result[1076] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3763', 'op': 'prim::TupleUnpack', 'in': [1058], 'output_id': 2, 'shape': [], 'out': [1079], 'sorted_id': 1078}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1078] = op;
            
            op->set_inputs( forward_result[1058] );
        }
        
        // {'name': 'Model/Net[net]/3768', 'op': 'aten::add', 'in': [1077, 1078, 12], 'output_id': 0, 'shape': [], 'out': [1080], 'sorted_id': 1079}
        {
            AddOp* op = new AddOp();
            forward_result[1079] = op;
            
            op->set_inputs( forward_result[1077] );
            op->set_inputs( forward_result[1078] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/3773', 'op': 'prim::ListConstruct', 'in': [1069, 1074, 1079], 'output_id': 0, 'shape': [], 'out': [1081], 'sorted_id': 1080}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[1080] = op;
            
            op->set_inputs( forward_result[1069] );
            op->set_inputs( forward_result[1074] );
            op->set_inputs( forward_result[1079] );
        }
        
        // {'name': 'Model/Net[net]/log_qs', 'op': 'aten::stack', 'in': [1080, 10], 'output_id': 0, 'shape': [3], 'out': [1082], 'sorted_id': 1081}
        {
            Tensor::shape_type shape = {3};
            StackOp* op = new StackOp();
            forward_result[1081] = op;
            
            op->set_inputs( forward_result[1080] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/log_q', 'op': 'aten::mean', 'in': [1081, 20], 'output_id': 0, 'shape': [], 'out': [1101], 'sorted_id': 1082}
        {
            MeanOp* op = new MeanOp();
            forward_result[1082] = op;
            
            op->set_inputs( forward_result[1081] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/2812', 'op': 'prim::TupleUnpack', 'in': [132], 'output_id': 1, 'shape': [], 'out': [1085], 'sorted_id': 1083}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1083] = op;
            
            op->set_inputs( forward_result[132] );
        }
        
        // {'name': 'Model/2943', 'op': 'prim::TupleUnpack', 'in': [251], 'output_id': 1, 'shape': [], 'out': [1085], 'sorted_id': 1084}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1084] = op;
            
            op->set_inputs( forward_result[251] );
        }
        
        // {'name': 'Model/Net[net]/3087', 'op': 'aten::add', 'in': [1083, 1084, 12], 'output_id': 0, 'shape': [], 'out': [1087], 'sorted_id': 1085}
        {
            AddOp* op = new AddOp();
            forward_result[1085] = op;
            
            op->set_inputs( forward_result[1083] );
            op->set_inputs( forward_result[1084] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3074', 'op': 'prim::TupleUnpack', 'in': [370], 'output_id': 1, 'shape': [], 'out': [1087], 'sorted_id': 1086}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1086] = op;
            
            op->set_inputs( forward_result[370] );
        }
        
        // {'name': 'Model/Net[net]/3088', 'op': 'aten::add', 'in': [1085, 1086, 12], 'output_id': 0, 'shape': [], 'out': [1098], 'sorted_id': 1087}
        {
            AddOp* op = new AddOp();
            forward_result[1087] = op;
            
            op->set_inputs( forward_result[1085] );
            op->set_inputs( forward_result[1086] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3201', 'op': 'prim::TupleUnpack', 'in': [491], 'output_id': 1, 'shape': [], 'out': [1090], 'sorted_id': 1088}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1088] = op;
            
            op->set_inputs( forward_result[491] );
        }
        
        // {'name': 'Model/3312', 'op': 'prim::TupleUnpack', 'in': [610], 'output_id': 1, 'shape': [], 'out': [1090], 'sorted_id': 1089}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1089] = op;
            
            op->set_inputs( forward_result[610] );
        }
        
        // {'name': 'Model/Net[net]/3426', 'op': 'aten::add', 'in': [1088, 1089, 12], 'output_id': 0, 'shape': [], 'out': [1092], 'sorted_id': 1090}
        {
            AddOp* op = new AddOp();
            forward_result[1090] = op;
            
            op->set_inputs( forward_result[1088] );
            op->set_inputs( forward_result[1089] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3423', 'op': 'prim::TupleUnpack', 'in': [729], 'output_id': 1, 'shape': [], 'out': [1092], 'sorted_id': 1091}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1091] = op;
            
            op->set_inputs( forward_result[729] );
        }
        
        // {'name': 'Model/Net[net]/3427', 'op': 'aten::add', 'in': [1090, 1091, 12], 'output_id': 0, 'shape': [], 'out': [1098], 'sorted_id': 1092}
        {
            AddOp* op = new AddOp();
            forward_result[1092] = op;
            
            op->set_inputs( forward_result[1090] );
            op->set_inputs( forward_result[1091] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3540', 'op': 'prim::TupleUnpack', 'in': [840], 'output_id': 1, 'shape': [], 'out': [1095], 'sorted_id': 1093}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1093] = op;
            
            op->set_inputs( forward_result[840] );
        }
        
        // {'name': 'Model/3651', 'op': 'prim::TupleUnpack', 'in': [949], 'output_id': 1, 'shape': [], 'out': [1095], 'sorted_id': 1094}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1094] = op;
            
            op->set_inputs( forward_result[949] );
        }
        
        // {'name': 'Model/Net[net]/3765', 'op': 'aten::add', 'in': [1093, 1094, 12], 'output_id': 0, 'shape': [], 'out': [1097], 'sorted_id': 1095}
        {
            AddOp* op = new AddOp();
            forward_result[1095] = op;
            
            op->set_inputs( forward_result[1093] );
            op->set_inputs( forward_result[1094] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/3762', 'op': 'prim::TupleUnpack', 'in': [1058], 'output_id': 1, 'shape': [], 'out': [1097], 'sorted_id': 1096}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1096] = op;
            
            op->set_inputs( forward_result[1058] );
        }
        
        // {'name': 'Model/Net[net]/3766', 'op': 'aten::add', 'in': [1095, 1096, 12], 'output_id': 0, 'shape': [], 'out': [1098], 'sorted_id': 1097}
        {
            AddOp* op = new AddOp();
            forward_result[1097] = op;
            
            op->set_inputs( forward_result[1095] );
            op->set_inputs( forward_result[1096] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Model/Net[net]/3771', 'op': 'prim::ListConstruct', 'in': [1087, 1092, 1097], 'output_id': 0, 'shape': [], 'out': [1099], 'sorted_id': 1098}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[1098] = op;
            
            op->set_inputs( forward_result[1087] );
            op->set_inputs( forward_result[1092] );
            op->set_inputs( forward_result[1097] );
        }
        
        // {'name': 'Model/Net[net]/log_ps', 'op': 'aten::stack', 'in': [1098, 10], 'output_id': 0, 'shape': [3], 'out': [1100], 'sorted_id': 1099}
        {
            Tensor::shape_type shape = {3};
            StackOp* op = new StackOp();
            forward_result[1099] = op;
            
            op->set_inputs( forward_result[1098] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'Model/Net[net]/log_p', 'op': 'aten::mean', 'in': [1099, 20], 'output_id': 0, 'shape': [], 'out': [1101], 'sorted_id': 1100}
        {
            MeanOp* op = new MeanOp();
            forward_result[1100] = op;
            
            op->set_inputs( forward_result[1099] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'Model/3779', 'op': 'prim::TupleConstruct', 'in': [1064, 1082, 1100], 'output_id': 0, 'shape': [], 'out': [1102, 1108, 1103], 'sorted_id': 1101}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[1101] = op;
            
            op->set_inputs( forward_result[1064] );
            op->set_inputs( forward_result[1082] );
            op->set_inputs( forward_result[1100] );
        }
        
        // {'name': 'Model/2672', 'op': 'prim::TupleUnpack', 'in': [1101], 'output_id': 1, 'shape': [], 'out': [1105], 'sorted_id': 1102}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[1102] = op;
            
            op->set_inputs( forward_result[1101] );
        }
        
        // {'name': 'Model/2673', 'op': 'prim::TupleUnpack', 'in': [1101], 'output_id': 2, 'shape': [], 'out': [1105], 'sorted_id': 1103}
        {
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[1103] = op;
            
            op->set_inputs( forward_result[1101] );
        }
        
        // {'name': 'Model/Loss[loss]/3781', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [1114, 1105], 'sorted_id': 1104}
        {
            Tensor c = (fprec)1.0;
            forward_result[1104] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/3787', 'op': 'aten::sub', 'in': [1102, 1103, 1104], 'output_id': 0, 'shape': [], 'out': [1107], 'sorted_id': 1105}
        {
            SubOp* op = new SubOp();
            forward_result[1105] = op;
            
            op->set_inputs( forward_result[1102] );
            op->set_inputs( forward_result[1103] );
            op->set_inputs( forward_result[1104] );
        }
        
        // {'name': 'Model/Loss[loss]/3780', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 15000.0, 'out': [1107], 'sorted_id': 1106}
        {
            Tensor c = (fprec)15000.0;
            forward_result[1106] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/3788', 'op': 'aten::div', 'in': [1105, 1106], 'output_id': 0, 'shape': [], 'out': [1114], 'sorted_id': 1107}
        {
            DivOp* op = new DivOp();
            forward_result[1107] = op;
            
            op->set_inputs( forward_result[1105] );
            op->set_inputs( forward_result[1106] );
        }
        
        // {'name': 'Model/2671', 'op': 'prim::TupleUnpack', 'in': [1101], 'output_id': 0, 'shape': [4, 10], 'out': [1113], 'sorted_id': 1108}
        {
            Tensor::shape_type shape = {4,10};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[1108] = op;
            
            op->set_inputs( forward_result[1101] );
        }
        
        // {'name': 'Model/Loss[loss]/3785', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [4], 'constant_value': [5.0, 4.0, 9.0, 4.0], 'out': [1113], 'sorted_id': 1109}
        {
            Tensor::shape_type shape = {4};
            Constant19.reshape( shape );
            forward_result[1109] = new VariableTensor( Constant19, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/3784', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [1113], 'sorted_id': 1110}
        {
            forward_result[1110] = NULL;
        }
        
        // {'name': 'Model/Loss[loss]/3783', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [1113], 'sorted_id': 1111}
        {
            Tensor c = (fprec)2.0;
            forward_result[1111] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/3782', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [1113], 'sorted_id': 1112}
        {
            Tensor c = (fprec)-100.0;
            forward_result[1112] = new VariableTensor( c, 1 );
        }
        
        // {'name': 'Model/Loss[loss]/nll', 'op': 'aten::nll_loss_nd', 'in': [1108, 1109, 1110, 1111, 1112], 'output_id': 0, 'shape': [], 'out': [1114], 'sorted_id': 1113}
        {
            NLLLossOp* op = new NLLLossOp();
            forward_result[1113] = op;
            
            op->set_inputs( forward_result[1108] );
            op->set_inputs( forward_result[1109] );
            op->set_inputs( forward_result[1110] );
            op->set_inputs( forward_result[1111] );
            op->set_inputs( forward_result[1112] );
        }
        
        // {'name': 'Model/Loss[loss]/3789', 'op': 'aten::add', 'in': [1107, 1113, 1104], 'output_id': 0, 'shape': [], 'out': [1115], 'sorted_id': 1114}
        {
            AddOp* op = new AddOp();
            forward_result[1114] = op;
            
            op->set_inputs( forward_result[1107] );
            op->set_inputs( forward_result[1113] );
            op->set_inputs( forward_result[1104] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [1114], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 1115}
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
        vector<MCTNode*> forward_result(1116);
    
        // input data
        Tensor::shape_type shape = {4,1,28,28};
        xin.reshape( shape );
        VariableTensor input_var( xin, 3 );
    
        xt::random::seed( 1 );
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 1114 );
    #else
        do_train1( forward_result, input_var, 1114 );
    #endif
        
        return 0;
    }
    