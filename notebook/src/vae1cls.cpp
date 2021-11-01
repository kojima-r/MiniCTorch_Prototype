
    //
    //  vae1cls
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
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [32, 64], 'out': [40, 3, 41], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {32,64};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc1]/weight/175', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {16,64};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc1]/bias/174', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
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
        
        // {'name': 'VAE/Net[net]/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [32, 16], 'out': [20, 7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/weight/179', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,16};
            fc2_mean_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_mean_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/bias/178', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {2};
            fc2_mean_bias.reshape( shape );
            forward_result[6] = new VariableTensor( fc2_mean_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_mean]/m1', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [32, 2], 'out': [11, 9, 34, 25], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'VAE/Net[net]/168', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [9], 'sorted_id': 8}
        {
            Tensor c = (fprec)0.0;
            forward_result[8] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/184', 'op': 'aten::size', 'in': [7, 8], 'output_id': 0, 'shape': [], 'out': [12], 'sorted_id': 9}
        {
            SizeOp* op = new SizeOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
        }
        
        // {'name': 'VAE/Net[net]/167', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [11, 25], 'sorted_id': 10}
        {
            Tensor c = (fprec)1.0;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/185', 'op': 'aten::size', 'in': [7, 10], 'output_id': 0, 'shape': [], 'out': [12], 'sorted_id': 11}
        {
            SizeOp* op = new SizeOp();
            forward_result[11] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'VAE/Net[net]/186', 'op': 'prim::ListConstruct', 'in': [9, 11], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 12}
        {
            ListConstructOp* op = new ListConstructOp();
            forward_result[12] = op;
            
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[11] );
        }
        
        // {'name': 'VAE/Net[net]/166', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 6.0, 'out': [17], 'sorted_id': 13}
        {
            Tensor c = (fprec)6.0;
            forward_result[13] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/165', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 14}
        {
            forward_result[14] = NULL;
        }
        
        // {'name': 'VAE/Net[net]/164', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [17], 'sorted_id': 15}
        {
            forward_result[15] = NULL;
        }
        
        // {'name': 'VAE/Net[net]/163', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.0, 'out': [17], 'sorted_id': 16}
        {
            Tensor c = (fprec)0.0;
            forward_result[16] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/eps', 'op': 'aten::randn', 'in': [12, 13, 14, 15, 16], 'output_id': 0, 'shape': [32, 2], 'out': [24], 'sorted_id': 17}
        {
            Tensor::shape_type shape = {32,2};
            RandnOp* op = new RandnOp();
            forward_result[17] = op;
            
            op->set_inputs( forward_result[12] );
            op->set_inputs( forward_result[13] );
            op->set_inputs( forward_result[14] );
            op->set_inputs( forward_result[15] );
            op->set_inputs( forward_result[16] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/weight/182', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 18}
        {
            Tensor::shape_type shape = {2,16};
            fc2_var_weight.reshape( shape );
            forward_result[18] = new VariableTensor( fc2_var_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/bias/181', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [20], 'sorted_id': 19}
        {
            Tensor::shape_type shape = {2};
            fc2_var_bias.reshape( shape );
            forward_result[19] = new VariableTensor( fc2_var_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc2_var]/v1', 'op': 'aten::linear', 'in': [4, 18, 19], 'output_id': 0, 'shape': [32, 2], 'out': [22, 34], 'sorted_id': 20}
        {
            Tensor::shape_type shape = {32,2};
            LinearOp* op = new LinearOp();
            forward_result[20] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[18] );
            op->set_inputs( forward_result[19] );
        }
        
        // {'name': 'VAE/Net[net]/162', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [22], 'sorted_id': 21}
        {
            Tensor c = (fprec)0.5;
            forward_result[21] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Net[net]/188', 'op': 'aten::mul', 'in': [20, 21], 'output_id': 0, 'shape': [32, 2], 'out': [23], 'sorted_id': 22}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[22] = op;
            
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[21] );
        }
        
        // {'name': 'VAE/Net[net]/189', 'op': 'aten::exp', 'in': [22], 'output_id': 0, 'shape': [32, 2], 'out': [24], 'sorted_id': 23}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[23] = op;
            
            op->set_inputs( forward_result[22] );
        }
        
        // {'name': 'VAE/Net[net]/190', 'op': 'aten::mul', 'in': [17, 23], 'output_id': 0, 'shape': [32, 2], 'out': [25], 'sorted_id': 24}
        {
            Tensor::shape_type shape = {32,2};
            MulOp* op = new MulOp();
            forward_result[24] = op;
            
            op->set_inputs( forward_result[17] );
            op->set_inputs( forward_result[23] );
        }
        
        // {'name': 'VAE/Net[net]/input.5', 'op': 'aten::add', 'in': [7, 24, 10], 'output_id': 0, 'shape': [32, 2], 'out': [28], 'sorted_id': 25}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[25] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[24] );
            op->set_inputs( forward_result[10] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/weight/193', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [28], 'sorted_id': 26}
        {
            Tensor::shape_type shape = {16,2};
            fc3_weight.reshape( shape );
            forward_result[26] = new VariableTensor( fc3_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/bias/192', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [28], 'sorted_id': 27}
        {
            Tensor::shape_type shape = {16};
            fc3_bias.reshape( shape );
            forward_result[27] = new VariableTensor( fc3_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc3]/input.7', 'op': 'aten::linear', 'in': [25, 26, 27], 'output_id': 0, 'shape': [32, 16], 'out': [29], 'sorted_id': 28}
        {
            Tensor::shape_type shape = {32,16};
            LinearOp* op = new LinearOp();
            forward_result[28] = op;
            
            op->set_inputs( forward_result[25] );
            op->set_inputs( forward_result[26] );
            op->set_inputs( forward_result[27] );
        }
        
        // {'name': 'VAE/Net[net]/input', 'op': 'aten::relu', 'in': [28], 'output_id': 0, 'shape': [32, 16], 'out': [32], 'sorted_id': 29}
        {
            Tensor::shape_type shape = {32,16};
            ReluOp* op = new ReluOp();
            forward_result[29] = op;
            
            op->set_inputs( forward_result[28] );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/weight/197', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [32], 'sorted_id': 30}
        {
            Tensor::shape_type shape = {64,16};
            fc4_weight.reshape( shape );
            forward_result[30] = new VariableTensor( fc4_weight );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/bias/196', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [32], 'sorted_id': 31}
        {
            Tensor::shape_type shape = {64};
            fc4_bias.reshape( shape );
            forward_result[31] = new VariableTensor( fc4_bias );
        }
        
        // {'name': 'VAE/Net[net]/Linear[fc4]/198', 'op': 'aten::linear', 'in': [29, 30, 31], 'output_id': 0, 'shape': [32, 64], 'out': [33], 'sorted_id': 32}
        {
            Tensor::shape_type shape = {32,64};
            LinearOp* op = new LinearOp();
            forward_result[32] = op;
            
            op->set_inputs( forward_result[29] );
            op->set_inputs( forward_result[30] );
            op->set_inputs( forward_result[31] );
        }
        
        // {'name': 'VAE/Net[net]/y', 'op': 'aten::sigmoid', 'in': [32], 'output_id': 0, 'shape': [32, 64], 'out': [34], 'sorted_id': 33}
        {
            Tensor::shape_type shape = {32,64};
            SigmoidOp* op = new SigmoidOp();
            forward_result[33] = op;
            
            op->set_inputs( forward_result[32] );
        }
        
        // {'name': 'VAE/200', 'op': 'prim::TupleConstruct', 'in': [33, 20, 7, 20, 7, 7], 'output_id': 0, 'shape': [], 'out': [54, 35, 58, 69, 51, 68], 'sorted_id': 34}
        {
            TupleConstructOp* op = new TupleConstructOp();
            forward_result[34] = op;
            
            op->set_inputs( forward_result[33] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[20] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[7] );
        }
        
        // {'name': 'VAE/154', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 0, 'shape': [32, 64], 'out': [38, 42], 'sorted_id': 35}
        {
            Tensor::shape_type shape = {32,64};
            TupleUnpackOp* op = new TupleUnpackOp( 0 );
            forward_result[35] = op;
            
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/Loss[loss]/207', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1e-07, 'out': [38, 43], 'sorted_id': 36}
        {
            Tensor c = (fprec)1e-07;
            forward_result[36] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/206', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [43, 38, 42, 46, 65, 60, 41, 57, 53], 'sorted_id': 37}
        {
            Tensor c = (fprec)1.0;
            forward_result[37] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/208', 'op': 'aten::add', 'in': [35, 36, 37], 'output_id': 0, 'shape': [32, 64], 'out': [39], 'sorted_id': 38}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[38] = op;
            
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/209', 'op': 'aten::log', 'in': [38], 'output_id': 0, 'shape': [32, 64], 'out': [40], 'sorted_id': 39}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[39] = op;
            
            op->set_inputs( forward_result[38] );
        }
        
        // {'name': 'VAE/Loss[loss]/210', 'op': 'aten::mul', 'in': [0, 39], 'output_id': 0, 'shape': [32, 64], 'out': [46], 'sorted_id': 40}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[40] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[39] );
        }
        
        // {'name': 'VAE/Loss[loss]/211', 'op': 'aten::rsub', 'in': [0, 37, 37], 'output_id': 0, 'shape': [32, 64], 'out': [45], 'sorted_id': 41}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[41] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/212', 'op': 'aten::rsub', 'in': [35, 37, 37], 'output_id': 0, 'shape': [32, 64], 'out': [43], 'sorted_id': 42}
        {
            Tensor::shape_type shape = {32,64};
            RsubOp* op = new RsubOp();
            forward_result[42] = op;
            
            op->set_inputs( forward_result[35] );
            op->set_inputs( forward_result[37] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/213', 'op': 'aten::add', 'in': [42, 36, 37], 'output_id': 0, 'shape': [32, 64], 'out': [44], 'sorted_id': 43}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[43] = op;
            
            op->set_inputs( forward_result[42] );
            op->set_inputs( forward_result[36] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/214', 'op': 'aten::log', 'in': [43], 'output_id': 0, 'shape': [32, 64], 'out': [45], 'sorted_id': 44}
        {
            Tensor::shape_type shape = {32,64};
            LogOp* op = new LogOp();
            forward_result[44] = op;
            
            op->set_inputs( forward_result[43] );
        }
        
        // {'name': 'VAE/Loss[loss]/215', 'op': 'aten::mul', 'in': [41, 44], 'output_id': 0, 'shape': [32, 64], 'out': [46], 'sorted_id': 45}
        {
            Tensor::shape_type shape = {32,64};
            MulOp* op = new MulOp();
            forward_result[45] = op;
            
            op->set_inputs( forward_result[41] );
            op->set_inputs( forward_result[44] );
        }
        
        // {'name': 'VAE/Loss[loss]/e', 'op': 'aten::add', 'in': [40, 45, 37], 'output_id': 0, 'shape': [32, 64], 'out': [48], 'sorted_id': 46}
        {
            Tensor::shape_type shape = {32,64};
            AddOp* op = new AddOp();
            forward_result[46] = op;
            
            op->set_inputs( forward_result[40] );
            op->set_inputs( forward_result[45] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/205', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [61, 48], 'sorted_id': 47}
        {
            forward_result[47] = NULL;
        }
        
        // {'name': 'VAE/Loss[loss]/217', 'op': 'aten::sum', 'in': [46, 47], 'output_id': 0, 'shape': [], 'out': [50], 'sorted_id': 48}
        {
            SumOp*    op = new SumOp();
            forward_result[48] = op;
            
            op->set_inputs( forward_result[46] );
            op->set_inputs( forward_result[47] );
        }
        
        // {'name': 'VAE/Loss[loss]/204', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 32.0, 'out': [50, 64], 'sorted_id': 49}
        {
            Tensor c = (fprec)32.0;
            forward_result[49] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/e1', 'op': 'aten::div', 'in': [48, 49], 'output_id': 0, 'shape': [], 'out': [65], 'sorted_id': 50}
        {
            DivOp* op = new DivOp();
            forward_result[50] = op;
            
            op->set_inputs( forward_result[48] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'VAE/155', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 1, 'shape': [32, 2], 'out': [53], 'sorted_id': 51}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 1 );
            forward_result[51] = op;
            
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/Loss[loss]/203', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [53], 'sorted_id': 52}
        {
            Tensor c = (fprec)1.0;
            forward_result[52] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/219', 'op': 'aten::add', 'in': [51, 52, 37], 'output_id': 0, 'shape': [32, 2], 'out': [57], 'sorted_id': 53}
        {
            Tensor::shape_type shape = {32,2};
            AddOp* op = new AddOp();
            forward_result[53] = op;
            
            op->set_inputs( forward_result[51] );
            op->set_inputs( forward_result[52] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/156', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 2, 'shape': [32, 2], 'out': [56], 'sorted_id': 54}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 2 );
            forward_result[54] = op;
            
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/Loss[loss]/202', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 2.0, 'out': [56], 'sorted_id': 55}
        {
            Tensor c = (fprec)2.0;
            forward_result[55] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/220', 'op': 'aten::pow', 'in': [54, 55], 'output_id': 0, 'shape': [32, 2], 'out': [57], 'sorted_id': 56}
        {
            Tensor::shape_type shape = {32,2};
            PowOp* op = new PowOp();
            forward_result[56] = op;
            
            op->set_inputs( forward_result[54] );
            op->set_inputs( forward_result[55] );
        }
        
        // {'name': 'VAE/Loss[loss]/221', 'op': 'aten::sub', 'in': [53, 56, 37], 'output_id': 0, 'shape': [32, 2], 'out': [60], 'sorted_id': 57}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[57] = op;
            
            op->set_inputs( forward_result[53] );
            op->set_inputs( forward_result[56] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/157', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 3, 'shape': [32, 2], 'out': [59], 'sorted_id': 58}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 3 );
            forward_result[58] = op;
            
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/Loss[loss]/222', 'op': 'aten::exp', 'in': [58], 'output_id': 0, 'shape': [32, 2], 'out': [60], 'sorted_id': 59}
        {
            Tensor::shape_type shape = {32,2};
            ExpOp* op = new ExpOp();
            forward_result[59] = op;
            
            op->set_inputs( forward_result[58] );
        }
        
        // {'name': 'VAE/Loss[loss]/g', 'op': 'aten::sub', 'in': [57, 59, 37], 'output_id': 0, 'shape': [32, 2], 'out': [61], 'sorted_id': 60}
        {
            Tensor::shape_type shape = {32,2};
            SubOp* op = new SubOp();
            forward_result[60] = op;
            
            op->set_inputs( forward_result[57] );
            op->set_inputs( forward_result[59] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/224', 'op': 'aten::sum', 'in': [60, 47], 'output_id': 0, 'shape': [], 'out': [63], 'sorted_id': 61}
        {
            SumOp*    op = new SumOp();
            forward_result[61] = op;
            
            op->set_inputs( forward_result[60] );
            op->set_inputs( forward_result[47] );
        }
        
        // {'name': 'VAE/Loss[loss]/201', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 0.5, 'out': [63], 'sorted_id': 62}
        {
            Tensor c = (fprec)0.5;
            forward_result[62] = new VariableTensor( c, false );
        }
        
        // {'name': 'VAE/Loss[loss]/225', 'op': 'aten::mul', 'in': [61, 62], 'output_id': 0, 'shape': [], 'out': [64], 'sorted_id': 63}
        {
            MulOp* op = new MulOp();
            forward_result[63] = op;
            
            op->set_inputs( forward_result[61] );
            op->set_inputs( forward_result[62] );
        }
        
        // {'name': 'VAE/Loss[loss]/e2', 'op': 'aten::div', 'in': [63, 49], 'output_id': 0, 'shape': [], 'out': [65], 'sorted_id': 64}
        {
            DivOp* op = new DivOp();
            forward_result[64] = op;
            
            op->set_inputs( forward_result[63] );
            op->set_inputs( forward_result[49] );
        }
        
        // {'name': 'VAE/Loss[loss]/227', 'op': 'aten::add', 'in': [50, 64, 37], 'output_id': 0, 'shape': [], 'out': [66], 'sorted_id': 65}
        {
            AddOp* op = new AddOp();
            forward_result[65] = op;
            
            op->set_inputs( forward_result[50] );
            op->set_inputs( forward_result[64] );
            op->set_inputs( forward_result[37] );
        }
        
        // {'name': 'VAE/Loss[loss]/228', 'op': 'aten::neg', 'in': [65], 'output_id': 0, 'shape': [], 'out': [67], 'sorted_id': 66}
        {
            NegOp* op = new NegOp();
            forward_result[66] = op;
            
            op->set_inputs( forward_result[65] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [66], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 67}
        {
        }
        
        // {'name': 'VAE/158', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 4, 'shape': [32, 2], 'out': [], 'sorted_id': 68}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 4 );
            forward_result[68] = op;
            
            op->set_inputs( forward_result[34] );
        }
        
        // {'name': 'VAE/159', 'op': 'prim::TupleUnpack', 'in': [34], 'output_id': 5, 'shape': [32, 2], 'out': [], 'sorted_id': 69}
        {
            Tensor::shape_type shape = {32,2};
            TupleUnpackOp* op = new TupleUnpackOp( 5 );
            forward_result[69] = op;
            
            op->set_inputs( forward_result[34] );
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
    
    int main()
    {
        vector<MCTNode*> forward_result(70);
    
        // input data
        Tensor::shape_type shape = {32,64};
        xin.reshape( shape );
        VariableTensor input_var(xin);
    
        defineOp( forward_result, input_var );
        do_train1( forward_result, input_var, 66 );
        
        return 0;
    }
    