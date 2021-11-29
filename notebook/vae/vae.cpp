
    //
    //  vae
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
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [53, 3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc1]/weight/weight.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc1]/bias/bias.13', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {16};
            fc1_bias.reshape( shape );
            forward_result[2] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'output_id': 0, 'shape': [32, 16], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'VAE/Net[net]/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [32, 16], 'out': [13], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/weight/weight.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {16};
            bn1_weight.reshape( shape );
            forward_result[5] = new VariableTensor( bn1_weight );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/bias/bias.15', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {16};
            bn1_bias.reshape( shape );
            forward_result[6] = new VariableTensor( bn1_bias );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/running_mean/running_mean', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {16};
            bn1_running_mean.reshape( shape );
            forward_result[7] = new VariableTensor( bn1_running_mean );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/running_var/running_var', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {16};
            bn1_running_var.reshape( shape );
            forward_result[8] = new VariableTensor( bn1_running_var );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/224', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [13, 36, 34], 'sorted_id': 9}
        {
            Tensor c = (fprec)0.0;
            forward_result[9] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/225', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.1, 'out': [13], 'sorted_id': 10}
        {
            Tensor c = (fprec)0.1;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/226', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-05, 'out': [13], 'sorted_id': 11}
        {
            Tensor c = (fprec)1e-05;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/227', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [13], 'sorted_id': 12}
        {
            Tensor c = (fprec)1.0;
            forward_result[12] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/BatchNorm1d[bn1]/input.5', 'op': 'aten::batch_norm', 'in': [4, 5, 6, 7, 8, 9, 10, 11, 12], 'output_id': 0, 'shape': [32, 16], 'out': [19, 16], 'sorted_id': 13}
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
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/weight/weight.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 14}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[14] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/bias/bias.17', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [16], 'sorted_id': 15}
        {
            Tensor::shape_type shape = {2};
            fc2_mean_bias.reshape( shape );
            forward_result[15] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/245', 'op': 'aten::linear', 'in': [13, 14, 15], 'output_id': 0, 'shape': [32, 2], 'out': [23], 'sorted_id': 16}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[16] = op;
            
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/weight/weight.19', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[17] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/bias/bias.19', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [19], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {2};
            fc2_var_bias.reshape( shape );
            forward_result[18] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/v1', 'op': 'aten::linear', 'in': [13, 17, 18], 'output_id': 0, 'shape': [32, 2], 'out': [21], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[19] = op;
            
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[18] );
        }
        
        // {'name': 'VAE/Net[net]/223', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [21], 'sorted_id': 20}
        {
            Tensor c = (fprec)0.5;
            forward_result[20] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/249', 'op': 'aten::mul', 'in': [19, 20], 'output_id': 0, 'shape': [32, 2], 'out': [22], 'sorted_id': 21}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[21] = op;
            
            op->set_inputs( forward_result[19] );
            op->set_inputs( forward_result[20] );
        }
        
        // {'name': 'VAE/Net[net]/250', 'op': 'aten::exp', 'in': [21], 'output_id': 0, 'shape': [32, 2], 'out': [23], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[21] );
        }
        
        // {'name': 'VAE/Net[net]/251', 'op': 'prim::ListConstruct', 'in': [16, 22], 'output_id': 0, 'shape': [], 'out': [24], 'sorted_id': 23}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[16] );
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'VAE/Net[net]/252', 'op': 'aten::broadcast_tensors', 'in': [23], 'output_id': 0, 'shape': [], 'out': [25, 38], 'sorted_id': 24}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'VAE/Net[net]/loc.1', 'op': 'prim::ListUnpack', 'in': [24], 'output_id': 0, 'shape': [32, 2], 'out': [27, 40, 49, 29], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[25] = op;
            
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'VAE/Net[net]/222', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [27], 'sorted_id': 26}
        {
            Tensor c = (fprec)0.0;
            forward_result[26] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/255', 'op': 'aten::size', 'in': [25, 26], 'output_id': 0, 'shape': [], 'out': [35, 30], 'sorted_id': 27}
        {
            SizeOp* op = new SizeOp();
            forward_result[27] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
        }
        
        // {'name': 'VAE/Net[net]/221', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [40, 29], 'sorted_id': 28}
        {
            Tensor c = (fprec)1.0;
            forward_result[28] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/256', 'op': 'aten::size', 'in': [25, 28], 'output_id': 0, 'shape': [], 'out': [35, 30], 'sorted_id': 29}
        {
            SizeOp* op = new SizeOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'VAE/Net[net]/257', 'op': 'prim::ListConstruct', 'in': [27, 29], 'output_id': 0, 'shape': [], 'out': [34], 'sorted_id': 30}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[30] = op;
            
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'VAE/Net[net]/220', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [36, 34], 'sorted_id': 31}
        {
            Tensor c = (fprec)6.0;
            forward_result[31] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/219', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [37, 36, 34], 'sorted_id': 32}
        {
            forward_result[32] = NULL;
        }
        
        // {'name': 'VAE/Net[net]/218', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [36, 34], 'sorted_id': 33}
        {
            forward_result[33] = NULL;
        }
        
        // {'name': 'VAE/Net[net]/258', 'op': 'aten::zeros', 'in': [30, 31, 32, 33, 9], 'output_id': 0, 'shape': [32, 2], 'out': [37], 'sorted_id': 34}
        {
            Tensor::shape_type shape = {32,2};
            ZerosOp* op = new ZerosOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'VAE/Net[net]/259', 'op': 'prim::ListConstruct', 'in': [27, 29], 'output_id': 0, 'shape': [], 'out': [36], 'sorted_id': 35}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[35] = op;
            
            op->set_inputs( forward_result[27] );
            op->set_inputs( forward_result[29] );
        }
        
        // {'name': 'VAE/Net[net]/260', 'op': 'aten::ones', 'in': [35, 31, 32, 33, 9], 'output_id': 0, 'shape': [32, 2], 'out': [37], 'sorted_id': 36}
        {
            Tensor::shape_type shape = {32,2};
            OnesOp* op = new OnesOp();
            forward_result[36] = op;
            
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[31] );
            op->set_inputs( forward_result[32] );
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[9] );
        }
        
        // {'name': 'VAE/Net[net]/eps', 'op': 'aten::normal', 'in': [34, 36, 32], 'output_id': 0, 'shape': [32, 2], 'out': [39], 'sorted_id': 37}
        {
            Tensor::shape_type shape = {32,2};
            NormalOp* op = new NormalOp();
            forward_result[37] = op;
            
            op->set_inputs( forward_result[34] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'VAE/Net[net]/value.1', 'op': 'prim::ListUnpack', 'in': [24], 'output_id': 1, 'shape': [32, 2], 'out': [49, 39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[38] = op;
            
            op->set_inputs( forward_result[24] );
        }
        
        // {'name': 'VAE/Net[net]/262', 'op': 'aten::mul', 'in': [37, 38], 'output_id': 0, 'shape': [32, 2], 'out': [40], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'VAE/Net[net]/input.7', 'op': 'aten::add', 'in': [25, 39, 28], 'output_id': 0, 'shape': [32, 2], 'out': [43], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[39] );
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/weight/weight.21', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [43], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[41] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/bias/bias.21', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {16};
            fc3_bias.reshape( shape );
            forward_result[42] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/input.9', 'op': 'aten::linear', 'in': [40, 41, 42], 'output_id': 0, 'shape': [32, 16], 'out': [44], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[42] );
        }
        
        // {'name': 'VAE/Net[net]/input.11', 'op': 'aten::relu', 'in': [43], 'output_id': 0, 'shape': [32, 16], 'out': [47], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/weight/weight', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[45] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/bias/bias', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [47], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {64};
            fc4_bias.reshape( shape );
            forward_result[46] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/270', 'op': 'aten::linear', 'in': [44, 45, 46], 'output_id': 0, 'shape': [32, 64], 'out': [48], 'sorted_id': 47}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[47] = op;
            
            op->set_inputs( forward_result[44] );
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[46] );
        }
        
        // {'name': 'VAE/Net[net]/input', 'op': 'aten::sigmoid', 'in': [47], 'output_id': 0, 'shape': [32, 64], 'out': [49], 'sorted_id': 48}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[48] = op;
            
            op->set_inputs( forward_result[47] );
        }
        
        // {'name': 'VAE/272', 'op': 'prim::TupleConstruct', 'in': [48, 25, 38], 'output_id': 0, 'shape': [], 'out': [50, 55, 54], 'sorted_id': 49}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[49] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'VAE/213', 'op': 'prim::TupleUnpack', 'in': [49], 'output_id': 0, 'shape': [32, 64], 'out': [53], 'sorted_id': 50}
        {
            Tensor::shape_type shape = {32,64};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[50] = op;
            
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'VAE/Loss[loss]/281', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [53, 60, 79, 61], 'sorted_id': 51}
        {
            forward_result[51] = NULL;
        }
        
        // {'name': 'VAE/Loss[loss]/280', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [53, 71, 66], 'sorted_id': 52}
        {
            Tensor c = (fprec)2.0;
            forward_result[52] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/e1', 'op': 'aten::binary_cross_entropy', 'in': [50, 0, 51, 52], 'output_id': 0, 'shape': [], 'out': [80], 'sorted_id': 53}
        {
            BCELossOp* op = new BCELossOp();
            forward_result[53] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[52] );
        }
        
        // {'name': 'VAE/215', 'op': 'prim::TupleUnpack', 'in': [49], 'output_id': 2, 'shape': [32, 2], 'out': [65, 61], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[54] = op;
            
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'VAE/214', 'op': 'prim::TupleUnpack', 'in': [49], 'output_id': 1, 'shape': [32, 2], 'out': [69, 60], 'sorted_id': 55}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[55] = op;
            
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'VAE/Loss[loss]/279', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [60, 61], 'sorted_id': 56}
        {
            Tensor c = (fprec)6.0;
            forward_result[56] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/278', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [60, 61], 'sorted_id': 57}
        {
            Tensor c = (fprec)0.0;
            forward_result[57] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/277', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [60, 61], 'sorted_id': 58}
        {
            forward_result[58] = NULL;
        }
        
        // {'name': 'VAE/Loss[loss]/276', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [60, 61], 'sorted_id': 59}
        {
            Tensor c = (fprec)0.0;
            forward_result[59] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/283', 'op': 'aten::zeros_like', 'in': [55, 56, 57, 58, 59, 51], 'output_id': 0, 'shape': [32, 2], 'out': [62], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {32,2};
            FullLikeOp* op = new FullLikeOp( 0.0 );
            forward_result[60] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'VAE/Loss[loss]/284', 'op': 'aten::ones_like', 'in': [54, 56, 57, 58, 59, 51], 'output_id': 0, 'shape': [32, 2], 'out': [62], 'sorted_id': 61}
        {
            Tensor::shape_type shape = {32,2};
            FullLikeOp* op = new FullLikeOp( 1.0 );
            forward_result[61] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[58] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'VAE/Loss[loss]/285', 'op': 'prim::ListConstruct', 'in': [60, 61], 'output_id': 0, 'shape': [], 'out': [63], 'sorted_id': 62}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[62] = op;
            
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[61] );
        }
        
        // {'name': 'VAE/Loss[loss]/286', 'op': 'aten::broadcast_tensors', 'in': [62], 'output_id': 0, 'shape': [], 'out': [67, 64], 'sorted_id': 63}
        {
            BroadcastTensorsOp* op = new BroadcastTensorsOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'VAE/Loss[loss]/value', 'op': 'prim::ListUnpack', 'in': [63], 'output_id': 1, 'shape': [32, 2], 'out': [70, 65], 'sorted_id': 64}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 1 );
            forward_result[64] = op;
            
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'VAE/Loss[loss]/289', 'op': 'aten::div', 'in': [54, 64], 'output_id': 0, 'shape': [32, 2], 'out': [66], 'sorted_id': 65}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'VAE/Loss[loss]/var_ratio', 'op': 'aten::pow', 'in': [65, 52], 'output_id': 0, 'shape': [32, 2], 'out': [72, 75], 'sorted_id': 66}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[65] );
            op->set_inputs( forward_result[52] );
        }
        
        // {'name': 'VAE/Loss[loss]/loc', 'op': 'prim::ListUnpack', 'in': [63], 'output_id': 0, 'shape': [32, 2], 'out': [69], 'sorted_id': 67}
        {
            Tensor::shape_type shape = {32,2};
            ListUnpackOp* op = new ListUnpackOp( 0 );
            forward_result[67] = op;
            
            op->set_inputs( forward_result[63] );
        }
        
        // {'name': 'VAE/Loss[loss]/275', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [74, 80, 76, 69, 72], 'sorted_id': 68}
        {
            Tensor c = (fprec)1.0;
            forward_result[68] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/291', 'op': 'aten::sub', 'in': [55, 67, 68], 'output_id': 0, 'shape': [32, 2], 'out': [70], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[69] = op;
            
            op->set_inputs( forward_result[55] );
            op->set_inputs( forward_result[67] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'VAE/Loss[loss]/292', 'op': 'aten::div', 'in': [69, 64], 'output_id': 0, 'shape': [32, 2], 'out': [71], 'sorted_id': 70}
        {
            Tensor::shape_type shape = {32,2};
            DivOp* op = new DivOp();
            forward_result[70] = op;
            
            op->set_inputs( forward_result[69] );
            op->set_inputs( forward_result[64] );
        }
        
        // {'name': 'VAE/Loss[loss]/t1', 'op': 'aten::pow', 'in': [70, 52], 'output_id': 0, 'shape': [32, 2], 'out': [72], 'sorted_id': 71}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[71] = op;
            
            op->set_inputs( forward_result[70] );
            op->set_inputs( forward_result[52] );
        }
        
        // {'name': 'VAE/Loss[loss]/294', 'op': 'aten::add', 'in': [66, 71, 68], 'output_id': 0, 'shape': [32, 2], 'out': [74], 'sorted_id': 72}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[72] = op;
            
            op->set_inputs( forward_result[66] );
            op->set_inputs( forward_result[71] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'VAE/Loss[loss]/274', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [74], 'sorted_id': 73}
        {
            Tensor c = (fprec)1.0;
            forward_result[73] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/295', 'op': 'aten::sub', 'in': [72, 73, 68], 'output_id': 0, 'shape': [32, 2], 'out': [76], 'sorted_id': 74}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[74] = op;
            
            op->set_inputs( forward_result[72] );
            op->set_inputs( forward_result[73] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'VAE/Loss[loss]/296', 'op': 'aten::log', 'in': [66], 'output_id': 0, 'shape': [32, 2], 'out': [76], 'sorted_id': 75}
        {
            Tensor::shape_type shape = {32,2};
            LogOp* op = new LogOp();
            forward_result[75] = op;
            
            op->set_inputs( forward_result[66] );
        }
        
        // {'name': 'VAE/Loss[loss]/297', 'op': 'aten::sub', 'in': [74, 75, 68], 'output_id': 0, 'shape': [32, 2], 'out': [78], 'sorted_id': 76}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[76] = op;
            
            op->set_inputs( forward_result[74] );
            op->set_inputs( forward_result[75] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'VAE/Loss[loss]/273', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [78], 'sorted_id': 77}
        {
            Tensor c = (fprec)0.5;
            forward_result[77] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/298', 'op': 'aten::mul', 'in': [76, 77], 'output_id': 0, 'shape': [32, 2], 'out': [79], 'sorted_id': 78}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[78] = op;
            
            op->set_inputs( forward_result[76] );
            op->set_inputs( forward_result[77] );
        }
        
        // {'name': 'VAE/Loss[loss]/e2', 'op': 'aten::sum', 'in': [78, 51], 'output_id': 0, 'shape': [], 'out': [80], 'sorted_id': 79}
        {
            SumOp*    op = new SumOp();
            forward_result[79] = op;
            
            op->set_inputs( forward_result[78] );
            op->set_inputs( forward_result[51] );
        }
        
        // {'name': 'VAE/Loss[loss]/300', 'op': 'aten::add', 'in': [53, 79, 68], 'output_id': 0, 'shape': [], 'out': [81], 'sorted_id': 80}
        {
            AddOp* op = new AddOp();
            forward_result[80] = op;
            
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[79] );
            op->set_inputs( forward_result[68] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [80], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 81}
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
        vector<MCTNode*> forward_result(82);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
    #ifdef _TRAIN
        do_train_loop( forward_result, input_var, 80 );
    #else
        do_train1( forward_result, input_var, 80 );
    #endif
        
        return 0;
    }
    