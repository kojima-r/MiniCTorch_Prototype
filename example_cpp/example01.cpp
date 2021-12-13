#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include"minictorch.hpp"

using namespace std;

void print_shape(Tensor x){
    auto s=x.shape();
    cout<<"(";
    for(int i=0;i<s.size();i++){
        cout<<s[i]<<",";
    }
    cout<<")";
}

void test01(){
    Tensor a=
            {{{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}},
            {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}}};
    Tensor b=
            {{{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}},
            {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}}};
    Tensor c({2,3,3});
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                a[i, j,k] = 1;
                b[i, j,k] = 1;
            }
        }
    }
    c=a+b;
    cout<<c<<endl;
    const auto& d = c.shape();
    cout << "Dim size: " << d.size() <<endl;
    for(int i=0;i<d.size();i++){
        cout << ">>" << d[i] <<endl;
    }
    /////////////
    {
        cout<<"=="<<endl;
        VariableTensor va(a);
        VariableTensor vb(b);
        AddOp op_a_b;
        op_a_b.inputs.push_back(&va);
        op_a_b.inputs.push_back(&vb);
        op_a_b.forward();
        cout<<op_a_b.output<<endl;
        cout<<"=="<<endl;
        op_a_b.backward();
    }
};

void test02(){
    Tensor a=
            {{{1, 2, 3,4},
            {4, 5, 6,7},
            {7, 8, 9,10}},
            {{1, 2, 3,4},
            {4, 5, 6,7},
            {7, 8, 9,9}}};
    Tensor b={{1, 2, 3},
            {4, 5, 6},
            {4, 5, 6},
            {7, 8, 9}};
    auto bb=xt::expand_dims(b,1);
    auto z=xt::repeat(bb,2,1);
    cout<<z<<endl;
    {
        cout<<"=="<<endl;
        VariableTensor va(a);
        VariableTensor vb(b);
        MatMulOp op_a_b;
        op_a_b.inputs.push_back(&va);
        op_a_b.inputs.push_back(&vb);
        op_a_b.forward();
        cout<<op_a_b.output<<endl;
        cout<<"=="<<endl;
        op_a_b.grad=op_a_b.output;;
        op_a_b.backward();
        cout<<va.grad<<endl;
        cout<<vb.grad<<endl;
    }
}


