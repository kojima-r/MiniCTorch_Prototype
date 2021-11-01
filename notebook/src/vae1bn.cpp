
    //
    //  vae1bn
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
    
    bool train_mode = true;
    
    void defineOp( vector<MCTNode*>& forward_result, VariableTensor &input_var )
    {
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [55, 52, 3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/273', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'Net/Linear[fc1]/bias/272', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
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
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [32, 16], 'out': [13], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/weight/282', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {16};
            bn1_weight.reshape( shape );
            forward_result[5] = new VariableTensor( bn1_weight );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/bias/281', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {16};
            bn1_bias.reshape( shape );
            forward_result[6] = new VariableTensor( bn1_bias );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/running_mean/280', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {16};
            bn1_running_mean.reshape( shape );
            forward_result[7] = new VariableTensor( bn1_running_mean );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/running_var/279', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {16};
            bn1_running_var.reshape( shape );
            forward_result[8] = new VariableTensor( bn1_running_var );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/278', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [13], 'sorted_id': 9}
        {
            Tensor c = (fprec)0.0;
            forward_result[9] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/277', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.1, 'out': [13], 'sorted_id': 10}
        {
            Tensor c = (fprec)0.1;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/276', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-05, 'out': [13], 'sorted_id': 11}
        {
            Tensor c = (fprec)1e-05;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/275', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [13], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/BatchNorm1d[bn1]/input.5', 'op': 'aten::batch_norm', 'in': [4, 5, 6, 7, 8, 9, 10, 11, 12], 'output_id': 0, 'shape': [32, 16], 'out': [33, 16], 'sorted_id': 13}
        {
            Tensor::shape_type shape = {32,16};
            BatchNormOp* op = new BatchNormOp();
            forward_result[13] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
            op->set_inputs( forward_result[12] );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/weight/285', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[14] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/bias/284', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {2};
            fc2_mean_bias.reshape( shape );
            forward_result[15] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'Net/Linear[fc2_mean]/mean', 'op': 'aten::linear', 'in': [13, 14, 15], 'output_id': 0, 'shape': [32, 2], 'out': [22, 18, 74, 39], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'Net/153', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [18], 'sorted_id': 17}
        {
            Tensor c = (fprec)0.0;
            forward_result[17] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/154', 'op': 'aten::size', 'in': [16, 17], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            SizeOp* op = new SizeOp();
            forward_result[18] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[17] );
        }
        
        // {'name': 'Net/155', 'op': 'prim::NumToTensor', 'in': [18], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 19}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'Net/159', 'op': 'aten::Int', 'in': [19], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 20}
        {
            IntOp* op = new IntOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'Net/156', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [22], 'sorted_id': 21}
        {
            Tensor c = (fprec)1.0;
            forward_result[21] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/157', 'op': 'aten::size', 'in': [16, 21], 'output_id': 0, 'shape': [], 'out': [23], 'sorted_id': 22}
        {
            SizeOp* op = new SizeOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[21] );
        }
        
        // {'name': 'Net/158', 'op': 'prim::NumToTensor', 'in': [22], 'output_id': 0, 'shape': [], 'out': [24], 'sorted_id': 23}
        {
            NumToTensorOp* op = new NumToTensorOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'Net/160', 'op': 'aten::Int', 'in': [23], 'output_id': 0, 'shape': [], 'out': [25], 'sorted_id': 24}
        {
            IntOp* op = new IntOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'Net/161', 'op': 'prim::ListConstruct', 'in': [20, 24], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 25}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'Net/162', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [30], 'sorted_id': 26}
        {
            Tensor c = (fprec)6.0;
            forward_result[26] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/163', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 27}
        {
            forward_result[27] = NULL;
        }
        
        // {'name': 'Net/164', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [30], 'sorted_id': 28}
        {
            forward_result[28] = NULL;
        }
        
        // {'name': 'Net/165', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [30], 'sorted_id': 29}
        {
            Tensor c = (fprec)0.0;
            forward_result[29] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/eps', 'op': 'aten::randn', 'in': [25, 26, 27, 28, 29], 'output_id': 0, 'shape': [32, 2], 'out': [37], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {32,2};
            RandnOp* op = new RandnOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[28] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'Net/Linear[fc2_var]/weight/288', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[31] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'Net/Linear[fc2_var]/bias/287', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [33], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {2};
            fc2_var_bias.reshape( shape );
            forward_result[32] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'Net/Linear[fc2_var]/log_var', 'op': 'aten::linear', 'in': [13, 31, 32], 'output_id': 0, 'shape': [32, 2], 'out': [35, 72, 77], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'Net/167', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [35], 'sorted_id': 34}
        {
            Tensor c = (fprec)0.5;
            forward_result[34] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/168', 'op': 'aten::mul', 'in': [33, 34], 'output_id': 0, 'shape': [32, 2], 'out': [36], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'Net/169', 'op': 'aten::exp', 'in': [35], 'output_id': 0, 'shape': [32, 2], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[35] );
        }
        
        // {'name': 'Net/170', 'op': 'aten::mul', 'in': [30, 36], 'output_id': 0, 'shape': [32, 2], 'out': [39], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[36] );
        }
        
        // {'name': 'Net/171', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [39], 'sorted_id': 38}
        {
            Tensor c = (fprec)1.0;
            forward_result[38] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/input.7', 'op': 'aten::add', 'in': [16, 37, 38], 'output_id': 0, 'shape': [32, 2], 'out': [42], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'Net/Linear[fc3]/weight/291', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [42], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[40] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'Net/Linear[fc3]/bias/290', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [42], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {16};
            fc3_bias.reshape( shape );
            forward_result[41] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'Net/Linear[fc3]/input.9', 'op': 'aten::linear', 'in': [39, 40, 41], 'output_id': 0, 'shape': [32, 16], 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[41] );
        }
        
        // {'name': 'Net/input', 'op': 'aten::relu', 'in': [42], 'output_id': 0, 'shape': [32, 16], 'out': [46], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'Net/Linear[fc4]/weight/294', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [46], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[44] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'Net/Linear[fc4]/bias/293', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [46], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {64};
            fc4_bias.reshape( shape );
            forward_result[45] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'Net/Linear[fc4]/295', 'op': 'aten::linear', 'in': [43, 44, 45], 'output_id': 0, 'shape': [32, 64], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[43] );
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
        }
        
        // {'name': 'Net/y', 'op': 'aten::sigmoid', 'in': [46], 'output_id': 0, 'shape': [32, 64], 'out': [50, 58], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'Net/177', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-07, 'out': [50], 'sorted_id': 48}
        {
            Tensor c = (fprec)1e-07;
            forward_result[48] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/178', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [50], 'sorted_id': 49}
        {
            Tensor c = (fprec)1.0;
            forward_result[49] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/179', 'op': 'aten::add', 'in': [47, 48, 49], 'output_id': 0, 'shape': [32, 64], 'out': [51], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'Net/180', 'op': 'aten::log', 'in': [50], 'output_id': 0, 'shape': [32, 64], 'out': [52], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[51] = op;
            
            op->set_inputs( forward_result[50] );
        }
        
        // {'name': 'Net/181', 'op': 'aten::mul', 'in': [0, 51], 'output_id': 0, 'shape': [32, 64], 'out': [65], 'sorted_id': 52}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[52] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'Net/182', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [55], 'sorted_id': 53}
        {
            Tensor c = (fprec)1.0;
            forward_result[53] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/183', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [55], 'sorted_id': 54}
        {
            Tensor c = (fprec)1.0;
            forward_result[54] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/184', 'op': 'aten::rsub', 'in': [0, 53, 54], 'output_id': 0, 'shape': [32, 64], 'out': [63], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[55] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[54] );
        }
        
        // {'name': 'Net/185', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [58], 'sorted_id': 56}
        {
            Tensor c = (fprec)1.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/186', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [58], 'sorted_id': 57}
        {
            Tensor c = (fprec)1.0;
            forward_result[57] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/187', 'op': 'aten::rsub', 'in': [47, 56, 57], 'output_id': 0, 'shape': [32, 64], 'out': [61], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[58] = op;
            
            op->set_inputs( forward_result[47] );
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[57] );
        }
        
        // {'name': 'Net/188', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-07, 'out': [61], 'sorted_id': 59}
        {
            Tensor c = (fprec)1e-07;
            forward_result[59] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/189', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [61], 'sorted_id': 60}
        {
            Tensor c = (fprec)1.0;
            forward_result[60] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/190', 'op': 'aten::add', 'in': [58, 59, 60], 'output_id': 0, 'shape': [32, 64], 'out': [62], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[60] );
        }
        
        // {'name': 'Net/191', 'op': 'aten::log', 'in': [61], 'output_id': 0, 'shape': [32, 64], 'out': [63], 'sorted_id': 62}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'Net/192', 'op': 'aten::mul', 'in': [55, 62], 'output_id': 0, 'shape': [32, 64], 'out': [65], 'sorted_id': 63}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'Net/193', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [65], 'sorted_id': 64}
        {
            Tensor c = (fprec)1.0;
            forward_result[64] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e', 'op': 'aten::add', 'in': [52, 63, 64], 'output_id': 0, 'shape': [32, 64], 'out': [67], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'Net/195', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            forward_result[66] = NULL;
        }
        
        // {'name': 'Net/196', 'op': 'aten::sum', 'in': [65, 66], 'output_id': 0, 'shape': [], 'out': [69], 'sorted_id': 67}
        {
            SumOp*    op = new SumOp();
            forward_result[67] = op;
            
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[66] );
        }
        
        // {'name': 'Net/203', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 32.0, 'out': [69], 'sorted_id': 68}
        {
            Tensor c = (fprec)32.0;
            forward_result[68] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e1', 'op': 'aten::div', 'in': [67, 68], 'output_id': 0, 'shape': [], 'out': [87], 'sorted_id': 69}
        {
            DivOp* op = new DivOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[67] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'Net/205', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [72], 'sorted_id': 70}
        {
            Tensor c = (fprec)1.0;
            forward_result[70] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/206', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [72], 'sorted_id': 71}
        {
            Tensor c = (fprec)1.0;
            forward_result[71] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/207', 'op': 'aten::add', 'in': [33, 70, 71], 'output_id': 0, 'shape': [32, 2], 'out': [76], 'sorted_id': 72}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[71] );
        }
        
        // {'name': 'Net/208', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [74], 'sorted_id': 73}
        {
            Tensor c = (fprec)2.0;
            forward_result[73] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/209', 'op': 'aten::pow', 'in': [16, 73], 'output_id': 0, 'shape': [32, 2], 'out': [76], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[73] );
        }
        
        // {'name': 'Net/210', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [76], 'sorted_id': 75}
        {
            Tensor c = (fprec)1.0;
            forward_result[75] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/211', 'op': 'aten::sub', 'in': [72, 74, 75], 'output_id': 0, 'shape': [32, 2], 'out': [79], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
        }
        
        // {'name': 'Net/212', 'op': 'aten::exp', 'in': [33], 'output_id': 0, 'shape': [32, 2], 'out': [79], 'sorted_id': 77}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[77] = op;
            
            op->set_inputs( forward_result[33] );
        }
        
        // {'name': 'Net/213', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [79], 'sorted_id': 78}
        {
            Tensor c = (fprec)1.0;
            forward_result[78] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/g', 'op': 'aten::sub', 'in': [76, 77, 78], 'output_id': 0, 'shape': [32, 2], 'out': [81], 'sorted_id': 79}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[77] );
            op->set_inputs( forward_result[78] );
        }
        
        // {'name': 'Net/215', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [81], 'sorted_id': 80}
        {
            forward_result[80] = NULL;
        }
        
        // {'name': 'Net/216', 'op': 'aten::sum', 'in': [79, 80], 'output_id': 0, 'shape': [], 'out': [83], 'sorted_id': 81}
        {
            SumOp*    op = new SumOp();
            forward_result[81] = op;
            
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[80] );
        }
        
        // {'name': 'Net/217', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [83], 'sorted_id': 82}
        {
            Tensor c = (fprec)0.5;
            forward_result[82] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/218', 'op': 'aten::mul', 'in': [81, 82], 'output_id': 0, 'shape': [], 'out': [85], 'sorted_id': 83}
        {
            MulOp* op = new MulOp();
            forward_result[83] = op;
            
            op->set_inputs( forward_result[81] );
            op->set_inputs( forward_result[82] );
        }
        
        // {'name': 'Net/225', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 32.0, 'out': [85], 'sorted_id': 84}
        {
            Tensor c = (fprec)32.0;
            forward_result[84] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/e2', 'op': 'aten::div', 'in': [83, 84], 'output_id': 0, 'shape': [], 'out': [87], 'sorted_id': 85}
        {
            DivOp* op = new DivOp();
            forward_result[85] = op;
            
            op->set_inputs( forward_result[83] );
            op->set_inputs( forward_result[84] );
        }
        
        // {'name': 'Net/227', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [87], 'sorted_id': 86}
        {
            Tensor c = (fprec)1.0;
            forward_result[86] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/228', 'op': 'aten::add', 'in': [69, 85, 86], 'output_id': 0, 'shape': [], 'out': [88], 'sorted_id': 87}
        {
            AddOp* op = new AddOp();
            forward_result[87] = op;
            
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[85] );
            op->set_inputs( forward_result[86] );
        }
        
        // {'name': 'Net/229', 'op': 'aten::neg', 'in': [87], 'output_id': 0, 'shape': [], 'out': [89], 'sorted_id': 88}
        {
            NegOp* op = new NegOp();
            forward_result[88] = op;
            
            op->set_inputs( forward_result[87] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [88], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 89}
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
    
    //extern void do_train_loop( vector<MCTNode*>& forward_result, VariableTensor &input_var, int N );
    
    int main()
    {
        vector<MCTNode*> forward_result(90);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
        do_train1( forward_result, input_var, 88 );
        //do_train_loop( forward_result, input_var, 88 );
        
        return 0;
    }
    