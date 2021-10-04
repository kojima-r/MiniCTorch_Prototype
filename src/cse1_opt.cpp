
    //
    //  cse1_opt
    //
    #include<stdio.h>
    #include<iostream>
    #include<fstream>
    #include<string>
    #include<vector>
    #include"minictorch.hpp"

    using namespace std;
    
    extern Tensor  xin;
    extern Tensor  fc1_weight;
    extern Tensor  fc1_bias;
    extern Tensor  fc2_weight;
    extern Tensor  fc2_bias;
    extern Tensor  Constant1;
    
    int main()
    {
        vector<MCTNode*> forward_result(14);
    
        // input data
        Tensor::shape_type shape = {112,4};
        xin.reshape( shape );
        VariableTensor input_var(xin);
        
        // {'name': 'input/x', 'op': 'IO Node', 'in': [], 'output_id': 0, 'shape': [112, 4], 'out': [3], 'sorted_id': 0}
        {
            Tensor::shape_type shape = {112,4};
            forward_result[0] = &input_var;
        }
        
        // {'name': 'Net/Linear[fc1]/weight/35', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 1}
        {
            Tensor::shape_type shape = {64,4};
            fc1_weight.reshape( shape );
            forward_result[1] = new VariableTensor( fc1_weight );
        }
        
        // {'name': 'Net/Linear[fc1]/bias/34', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [3], 'sorted_id': 2}
        {
            Tensor::shape_type shape = {64};
            forward_result[2] = new VariableTensor( fc1_bias );
        }
        
        // {'name': 'Net/Linear[fc1]/input.1', 'op': 'aten::linear', 'in': [0, 1, 2], 'output_id': 0, 'shape': [112, 64], 'out': [4], 'sorted_id': 3}
        {
            Tensor::shape_type shape = {112,64};
            LinearOp* op = new LinearOp();
            forward_result[3] = op;
            
            op->set_inputs( forward_result[0] );
            op->set_inputs( forward_result[1] );
            op->set_inputs( forward_result[2] );
        }
        
        // {'name': 'Net/input.3', 'op': 'aten::relu', 'in': [3], 'output_id': 0, 'shape': [112, 64], 'out': [7], 'sorted_id': 4}
        {
            Tensor::shape_type shape = {112,64};
            ReluOp* op = new ReluOp();
            forward_result[4] = op;
            
            op->set_inputs( forward_result[3] );
        }
        
        // {'name': 'Net/Linear[fc2]/weight/38', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 5}
        {
            Tensor::shape_type shape = {3,64};
            fc2_weight.reshape( shape );
            forward_result[5] = new VariableTensor( fc2_weight );
        }
        
        // {'name': 'Net/Linear[fc2]/bias/37', 'op': 'prim::GetAttr', 'in': [], 'output_id': 0, 'shape': [], 'out': [7], 'sorted_id': 6}
        {
            Tensor::shape_type shape = {3};
            forward_result[6] = new VariableTensor( fc2_bias );
        }
        
        // {'name': 'Net/Linear[fc2]/input', 'op': 'aten::linear', 'in': [4, 5, 6], 'output_id': 0, 'shape': [112, 3], 'out': [12], 'sorted_id': 7}
        {
            Tensor::shape_type shape = {112,3};
            LinearOp* op = new LinearOp();
            forward_result[7] = op;
            
            op->set_inputs( forward_result[4] );
            op->set_inputs( forward_result[5] );
            op->set_inputs( forward_result[6] );
        }
        
        // {'name': 'Net/17', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [112], 'constant_value': [1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0], 'out': [12], 'sorted_id': 8}
        {
            Tensor::shape_type shape = {112};
            Constant1.reshape( shape );
            forward_result[8] = new VariableTensor( Constant1 );
        }
        
        // {'name': 'Net/18', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'out': [12], 'sorted_id': 9}
        {
            forward_result[9] = NULL;
        }
        
        // {'name': 'Net/19', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': 1.0, 'out': [12], 'sorted_id': 10}
        {
            Tensor c = (fprec)1.0;
            forward_result[10] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/20', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [], 'constant_value': -100.0, 'out': [12], 'sorted_id': 11}
        {
            Tensor c = (fprec)-100.0;
            forward_result[11] = new VariableTensor( c, false );
        }
        
        // {'name': 'Net/21', 'op': 'aten::cross_entropy_loss', 'in': [7, 8, 9, 10, 11], 'output_id': 0, 'shape': [], 'out': [13], 'sorted_id': 12}
        {
            CrossEntropyLossOp* op = new CrossEntropyLossOp();
            forward_result[12] = op;
            
            op->set_inputs( forward_result[7] );
            op->set_inputs( forward_result[8] );
            op->set_inputs( forward_result[9] );
            op->set_inputs( forward_result[10] );
            op->set_inputs( forward_result[11] );
        }
        
        // {'name': 'output/output.1', 'op': 'IO Node', 'in': [12], 'output_id': 0, 'shape': [], 'out': [], 'sorted_id': 13}
        {
        }
        
        /*
        cout<<"### forward computation ..."<<endl;
        //forward_result[12]->forward();
        for(int k=0;k<=12;k++) {
            if( forward_result[k] )  
            {
                //forward_result[k]->set_id( k );
                forward_result[k]->forward();
                forward_result[k]->zerograd();
            }
        }
        auto o = forward_result[12]->output;
        cout<<o<<endl;
    
        cout<<"### backward computation ..."<<endl;
        forward_result[12]->grad = xt::ones_like( forward_result[12]->output );
        //forward_result[12]->backward();
        for(int k=12;k>=0;k--) {
           if( forward_result[k] )  forward_result[k]->backward();
        }
        cout<<"input_grad"<<input_var.grad<<endl;
        */
        
    // optimization  210819 add
    auto exec_forward=[]( vector<MCTNode*> &op, int n ) 
    {
        cout<<"### forward computation ..."<<endl;
        //op[n]->forward();
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->forward();
        }
    };
    auto exec_backward=[]( vector<MCTNode*> &op, int n ) 
    {
        cout<<"### backward computation ..."<<endl;
        op[n]->grad = xt::ones_like( op[n]->output );
        //op[n]->backward();
        for(int k=n;k>=0;k--) {
          if( op[k] )  op[k]->backward();
        }
    };
    auto exec_zerograd=[]( vector<MCTNode*> &op, int n ) 
    {
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->zerograd();
        }
    };
    auto update_params=[]( vector<MCTNode*> &op, int n, fprec lr=0.01 ) 
    {
        for(int k=0;k<=n;k++) {
          if( op[k] )  op[k]->update( lr );
        }
    };
    
    auto lbs = forward_result[8]->get_output();
    auto sh = lbs.shape();
    //cout<<"sh"<<sh[0]<<endl;
        
    ofstream outputfile("cse1.out");
    for(int epoch=0;epoch<300;epoch++)
    {
        fprec lr= 0.01;
        int   NL = 12;  // loss
        int   NS =  7;  // result
        
        exec_forward( forward_result, NL );
        
        CrossEntropyLossOp *op = dynamic_cast<CrossEntropyLossOp*>(forward_result[NL]);
        
        auto  lb2 = op->get_classes();
        auto  eq  = xt::equal( lbs, lb2 );
        auto  nc2 = xt::sum( eq );
        fprec acc = fprec(nc2[0]) / fprec(sh[0]);
        //cout<<"lbs"<<lb2<<endl;
        //cout<<"nc2"<<nc2<<endl;
        
        fprec o = op->get_loss();
        cout<<"epoch "<<epoch<<" - loss "<<o<<" - accuracy "<<acc<<endl;
        outputfile<<to_string(o)<<","<<to_string(acc)<<endl;
        
        exec_backward( forward_result, NL );
        update_params( forward_result, NL, lr );
        exec_zerograd( forward_result, NL );
    }
    outputfile.close();
    
        return 0;
    }
    