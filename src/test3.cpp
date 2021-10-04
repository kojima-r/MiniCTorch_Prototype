
    //
    //  test3
    //
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
        vector<MCTNode*> forward_result(6);
    
        // input data
        Tensor::shape_type shape = {1,3};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [1, 3], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {1,3};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [2, 3], 'constant_value': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'out': [2], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {2,3};
            Constant1.reshape( shape );
            forward_result[1] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/tt', 'op': 'aten::t', 'in': [1], 'output_id': 0, 'shape': [3, 2], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {3,2};
            TransposeOp* op = new TransposeOp();
            forward_result[2] = op;
            
            op->set_inputs( forward_result[1] );
        }
        
        // {'name': 'Net/tensor', 'op': 'aten::matmul', 'in': [0, 2], 'output_id': 0, 'shape': [1, 2], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {1,2};
            MatMulOp* op = new MatMulOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/56', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [1, 2], 'out': [5], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {1,2};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [4], 'output_id': 0, 'shape': [1, 2], 'out': [], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {1,2};
        }
        
        cout<<"### forward computation ..."<<endl;
        //forward_result[4]->forward();
        for(int k=0;k<=4;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[4]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[4]->grad = xt::ones_like( forward_result[4]->output );
        //forward_result[4]->backward();
        for(int k=4;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
    
        return 0;
    }
    