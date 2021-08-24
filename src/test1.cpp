
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
        vector<MCTNode*> forward_result(11);
    
        // input data
        Tensor::shape_type shape = {2,2};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [2, 2], 'out': [7, 3, 2], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {2,2};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 10.0, 'out': [2], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)10.0;
            forward_result[1] = new VariableTensor( c );
        }
        
        // {'name': 'Net/5', 'op': 'aten::mul', 'in': [0, 1], 'shape': [2, 2], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[2] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
        }
        
        // {'name': 'Net/6', 'op': 'aten::mul', 'in': [2, 0], 'shape': [2, 2], 'out': [5], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[0] );
        }
        
        // {'name': 'Net/7', 'op': 'prim::Constant', 'in': [], 'shape': [2, 2], 'constant_value': [1.0, 2.0, 3.0, 4.0], 'out': [5], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {2,2};
            Constant1.reshape( shape );
            forward_result[4] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/f1', 'op': 'aten::mul', 'in': [3, 4], 'shape': [2, 2], 'out': [9], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[5] = op;
            
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
        }
        
        // {'name': 'Net/9', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 5.0, 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)5.0;
            forward_result[6] = new VariableTensor( c );
        }
        
        // {'name': 'Net/f2', 'op': 'aten::mul', 'in': [0, 6], 'shape': [2, 2], 'out': [9], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {2,2};
            MulOp* op = new MulOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/11', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [9], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[8] = new VariableTensor( c );
        }
        
        // {'name': 'Net/12', 'op': 'aten::add', 'in': [5, 7, 8], 'shape': [2, 2], 'out': [10], 'sorted_id': 9}
        {
            Tensor::shape_type shape = {2,2};
            AddOp* op = new AddOp();
            forward_result[9] = op;
            
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [9], 'shape': [2, 2], 'out': [], 'sorted_id': 10}
        {
            Tensor::shape_type shape = {2,2};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[9]->forward();
        for(int k=0;k<=9;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[9]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[9]->grad = xt::ones_like( forward_result[9]->output );
        //forward_result[9]->backward();
        for(int k=9;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    