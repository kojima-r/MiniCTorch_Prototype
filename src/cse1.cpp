
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
        vector<MCTNode*> forward_result(7);
    
        // input data
        Tensor::shape_type shape = {3,5};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [3, 5], 'out': [5], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {3,5};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [3], 'constant_value': [2.0, 3.0, 2.0], 'out': [5], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {3};
            Constant1.reshape( shape );
            forward_result[1] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/5', 'op': 'prim::Constant', 'in': [], 'shape': [], 'out': [5], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {};
            forward_result[2] = NULL;
        }
        
        // {'name': 'Net/6', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': 1.0, 'out': [5], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)1.0;
            forward_result[3] = new VariableTensor( c );
        }
        
        // {'name': 'Net/7', 'op': 'prim::Constant', 'in': [], 'shape': [], 'constant_value': -100.0, 'out': [5], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {};
            Tensor c = (float)-100.0;
            forward_result[4] = new VariableTensor( c );
        }
        
        // {'name': 'Net/8', 'op': 'aten::cross_entropy_loss', 'in': [0, 1, 2, 3, 4], 'shape': [], 'out': [6], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {};
            CrossEntropyLossOp* op = new CrossEntropyLossOp();
            forward_result[5] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
            op->set_inputs( forward_result[3] );
            op->set_inputs( forward_result[4] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [5], 'shape': [], 'out': [], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[5]->forward();
        for(int k=0;k<=5;k++) {
           if( forward_result[k] )  forward_result[k]->forward();
        }
        auto o = forward_result[5]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[5]->grad = xt::ones_like( forward_result[5]->output );
        //forward_result[5]->backward();
        for(int k=5;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    