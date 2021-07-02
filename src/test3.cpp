
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;

    int main()
    {
        
    #include "test3_param.h"
    
        // input data
        VariableTensor input_var(xin);
        vector<MCTNode*> forward_result(6);
    
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'shape': [1, 3], 'out': [3], 'sorted_id': 0}
        {
        
            Tensor::shape_type shape = {1,3};
            forward_result[0]=&input_var;
        }
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [2, 3], 'constant_value': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'out': [2], 'sorted_id': 1}
        {
        
            Tensor::shape_type shape = {2,3};
            Tensor t= {5.0,6.0,7.0,8.0,9.0,10.0};
            t=t.reshape(shape);
            forward_result[1]=new VariableTensor(t);
        }
        
        // {'name': 'Net/tt', 'op': 'aten::t', 'in': [1], 'shape': [3, 2], 'out': [3], 'sorted_id': 2}
        {
        
            Tensor::shape_type shape = {3,2};
            TransposeOp* op = new TransposeOp();
            forward_result[2]=op;
            MCTNode* p_in;
            p_in=forward_result[1];
            op->inputs.push_back(p_in);
        }
        
        // {'name': 'Net/z1', 'op': 'aten::matmul', 'in': [0, 2], 'shape': [1, 2], 'out': [4], 'sorted_id': 3}
        {
        
            Tensor::shape_type shape = {1,2};
            MatMulOp* op = new MatMulOp();
            forward_result[3]=op;
            MCTNode* p_in;
            p_in=forward_result[0];
            op->inputs.push_back(p_in);
            p_in=forward_result[2];
            op->inputs.push_back(p_in);
        }
        
        // {'name': 'Net/7', 'op': 'aten::relu', 'in': [3], 'shape': [1, 2], 'out': [5], 'sorted_id': 4}
        {
        
            Tensor::shape_type shape = {1,2};
            ReluOp* op = new ReluOp();
            forward_result[4]=op;
            MCTNode* p_in;
            p_in=forward_result[3];
            op->inputs.push_back(p_in);
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [4], 'shape': [1, 2], 'out': [], 'sorted_id': 5}
        {
        
            Tensor::shape_type shape = {1,2};
        }
        
        cout<<"### forward computation ..."<<endl;
        forward_result[4]->forward();
        auto o = forward_result[4]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[4]->grad=xt::ones_like(forward_result[4]->output); // 210702 mod mari
        forward_result[4]->backward();
        cout<<input_var.grad<<endl;
    
        return 0;
    }
    