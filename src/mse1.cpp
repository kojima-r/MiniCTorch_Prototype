
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  Constant1;
    
    int main()
    {
        vector<MCTNode*> forward_result(9);
    
        // input data
        Tensor::shape_type shape = {3,5};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [3, 5], 'out': [2], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {3,5};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/target.1', 'op': 'prim::Constant', 'in': [], 'shape': [3, 5], 'constant_value': [0.2673, -0.4212, -0.5107, -1.5727, -0.1232, 3.587, -1.8313, 1.5987, -1.277, 0.3255, -0.4791, 1.379, 2.5286, 0.4107, -0.988], 'out': [2], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {3,5};
            Constant1.reshape( shape );
            forward_result[1] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/19', 'op': 'prim::ListConstruct', 'in': [0, 1], 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {};
            ListConstructOp* op = new ListConstructOp();
            forward_result[2] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
        }
        
        // {'name': 'Net/20', 'op': 'aten::broadcast_tensors', 'in': [2], 'shape': [], 'out': [5, 4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {};
            MoveOp* op = new MoveOp( "broadcast_tensors" );
            forward_result[3] = op;
            
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input', 'op': 'prim::ListUnpack', 'in': [3], 'shape': [3, 5], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {3,5};
            ListUnpackOp* op = new ListUnpackOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/target', 'op': 'prim::ListUnpack', 'in': [3], 'shape': [3, 5], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {3,5};
            ListUnpackOp* op = new ListUnpackOp();
            forward_result[5] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/23', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[6] = new VariableTensor( c );
        }
        
        // {'name': 'Net/24', 'op': 'aten::mse_loss', 'in': [4, 5, 6], 'shape': [], 'out': [8], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {};
            MseLossOp* op = new MseLossOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [7], 'shape': [], 'out': [], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[7]->forward();
        for(int k=0;k<=7;k++) {
           if( forward_result[k] )  forward_result[k]->forward();
        }
        auto o = forward_result[7]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[7]->grad = xt::ones_like( forward_result[7]->output );
        //forward_result[7]->backward();
        for(int k=7;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    