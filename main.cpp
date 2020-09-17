#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include"xtensor/xarray.hpp"
#include"xtensor/xio.hpp"
#include"xtensor-blas/xlinalg.hpp"
#include"nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;
typedef xt::xarray<float> Tensor;

class MCTNode;

class MCTNode{
    public:
    virtual bool forward(){
        cout<<"not implemented: forward"<<endl;
    }
    virtual bool backward(){
        cout<<"not implemented: backward"<<endl;
        return false;
    }
    std::vector<MCTNode*> inputs;
    std::string name;
    Tensor output;
    Tensor grad;
};

class VariableTensor:public MCTNode{
    public:
    VariableTensor(){
    }
    VariableTensor(Tensor tensor){
        this->output=tensor;
    }
    VariableTensor(string name,Tensor tensor){
        this->output=tensor;
        this->name=name;
    }
    bool forward(){
        return true;
    }
    bool backward(){
        return true;
    }
};

class ExpOp:public MCTNode{
    public:
    ExpOp(){
    }
    int axis;
    bool forward(){
        for(auto& itr:inputs){
            itr->forward();
        }
        this->output=xt::exp(inputs[0]->output);
        return true;
    }
    bool backward(){
        for(auto& itr:inputs){
            itr->grad+=this->grad*this->output;
        }
        for(auto& itr:inputs){
            itr->backward();
        }
    }
};


class SumOp:public MCTNode{
    public:
    SumOp(int axis=0){
        this->axis=axis;
        this->size=1;
    }
    int axis;
    int size;
    bool forward(){
        for(auto& itr:inputs){
            itr->forward();
        }
        auto s=inputs[0]->output.shape();
        this->size=s[this->axis];
        this->output=xt::sum(inputs[0]->output,{this->axis});
        return true;
    }
    bool backward(){
        for(auto& itr:inputs){
            auto bb=xt::expand_dims(this->grad,this->axis);
            auto g=xt::repeat(bb,this->size,this->axis);
            itr->grad+=g;
        }
        for(auto& itr:inputs){
            itr->backward();
        }
    }
};


class AddOp:public MCTNode{
    public:
    AddOp(){
    }
    bool forward(){
        for(auto& itr:inputs){
            itr->forward();
        }
        this->output=inputs[0]->output+inputs[1]->output;
        //for(int i=1;i<inputs.size();i++){
        //    this->output=this->output+inputs[i]->output;
        //}
        return true;
    }
    bool backward(){
        this->inputs[0]->grad+=this->grad;
        this->inputs[1]->grad+=this->grad;
        /*
        for(auto& itr:inputs){
            itr->grad+=this->grad;
        }*/
        for(auto& itr:inputs){
            itr->backward();
        }
    }
};

class MulOp:public MCTNode{
    public:
    MulOp(){
    }
    bool forward(){
        for(auto& itr:inputs){
            itr->forward();
        }
        this->output=inputs[0]->output*inputs[1]->output;
        //this->output=inputs[0]->output;
        //for(int i=1;i<inputs.size();i++){
        //    this->output=this->output*inputs[i]->output;
        //}
        return true;
    }
    bool backward(){
        this->inputs[0]->grad+=this->grad*this->output/this->inputs[0]->output;
        this->inputs[1]->grad+=this->grad*this->output/this->inputs[1]->output;
        /*
        for(auto& itr:inputs){
            itr->grad+=this->grad*this->output/itr->output;
        }*/
        for(auto& itr:inputs){
            itr->backward();
        }
    }
};


void print_shape(Tensor x){
    auto s=x.shape();
    cout<<"(";
    for(int i=0;i<s.size();i++){
        cout<<s[i]<<",";
    }
    cout<<")";
}

class MatMulOp:public MCTNode{
    public:
    MatMulOp(){
    }
    
    bool forward(){
        if(inputs.size()!=2){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        auto a=inputs[0]->output;
        auto b=inputs[1]->output;
        auto as=a.shape();
        auto bs=b.shape();
        int ar=as.size();
        int br=bs.size();
        if(ar==2 && br==2){
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            this->output =xt::linalg::dot(a,b);
        }else if(ar > 2 && br==2){
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            this->output =xt::linalg::dot(a,b);
            print_shape(a);cout<<"x";
            print_shape(b);cout<<"=>";
            print_shape(this->output);cout<<endl;
        }else{
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    Tensor _transpose(Tensor a,int ar){
        vector<int> perm;
        for(int i=0;i<ar-2;i++){
            perm.push_back(i);
        }
        perm.push_back(ar-1);
        perm.push_back(ar-2);
        auto a_t=xt::transpose(a,perm);
        return a_t;
    }
    Tensor::shape_type _batch_shape(Tensor a,Tensor::shape_type a_shape){
        int b=1;
        int ar=a_shape.size();
        for(int i=0;i<ar-2;i++){
            b*=a_shape[i];
        }
        Tensor::shape_type new_s(3);
        new_s[0]=b;
        new_s[1]=a_shape[ar-2];
        new_s[2]=a_shape[ar-1];
        return new_s;
    }
    bool backward(){
        if(inputs.size()!=2){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        auto a=inputs[0]->output;
        auto b=inputs[1]->output;
        auto& ga=inputs[0]->grad;
        auto& gb=inputs[1]->grad;
        auto& gc=this->grad;
        auto as=a.shape();
        auto bs=b.shape();
        int ar=as.size();
        int br=bs.size();
        if(ar==2 && br==2){
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            ga+=xt::linalg::dot(gc,xt::transpose(b));
            gb+=xt::linalg::dot(xt::transpose(a),gc);
        }else if(ar > 2 && br==2){
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            ga+=xt::linalg::dot(gc,xt::transpose(b));
            cout<<"ga"<<endl;
            cout<<ga<<endl;
            cout<<"gc"<<endl;
            cout<<gc<<endl;
            cout<<"=="<<endl;
            auto a_t=this->_transpose(a,ar);
            cout<<"a_t"<<endl;
            print_shape(a_t);cout<<endl;
            auto new_a_shape=_batch_shape(a_t,a_t.shape());
            auto batch_a_t=a.reshape(new_a_shape);
            auto new_gc_shape=_batch_shape(gc,gc.shape());
            auto batch_gc=gc.reshape(new_gc_shape);
            //
            cout<<"gc"<<endl;
            print_shape(gc);cout<<endl;
            cout<<"a_t"<<endl;
            print_shape(a_t);cout<<endl;
            cout<<"b:gc"<<endl;
            print_shape(batch_gc);cout<<endl;
            cout<<"b:a_t"<<endl;
            print_shape(batch_a_t);cout<<endl;
            //
            Tensor::shape_type out_s={new_a_shape[0],new_a_shape[1],new_gc_shape[2]};
            Tensor out(out_s);
            for(int i=0;i<new_a_shape[0];i++){
                auto v = xt::view(a_t, i, xt::all(), xt::all());
                auto w = xt::view(gc, i, xt::all(), xt::all());
                auto temp=xt::linalg::dot(v,w);//a.T gc
                print_shape(temp);cout<<endl;
                xt::view(out, i, xt::all(), xt::all())=temp;
            }
            vector<int> r;
            for(int i=0;i<ar-2;i++){
                r.push_back(i);
            }
            //cout<<temp<<endl;
            //gb+=xt::sum(temp,r);
        }else{
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
};



class EinsumOp:public MCTNode{
    public:
    EinsumOp(){
    }
    bool forward(){
        return true;
    }
};
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

json read_computational_graph(const std::string& filename){
    std::ifstream reading(filename, std::ios::in);
    json j;
    reading >> j;
    for (auto& elem : j) {
        std::cout << elem << std::endl;
    }
    return j;
}

int main(){
    cout<<"### reading computational graph..."<<endl;
    json cg=read_computational_graph("network/example01.json");
    // input data
    Tensor x={{1, 2},
            {3, 4}};
    VariableTensor var_x(x);
    cout<<"### computational graph construction..."<<endl;
    int graph_size=cg.size();
    vector<MCTNode*> forward_result(graph_size);
    int output_id=-1;
    for (int i=0;i<graph_size;i++) {
        cout<<"=="<<i<<" :"<<endl;
        if(cg[i]["op"]=="IO Node"){
            if(cg[i]["name"]=="input/x"){
                cout<<x<<endl;
                forward_result[i]=&var_x;
            }else if(cg[i]["name"]=="output/output.1"){
                int num_inputs=cg[i]["in"].size();
                if(num_inputs>0){
                    output_id=cg[i]["in"][0];
                }
                //forward_result[i]=&var_x;
            }else{
                cout<<"unknown IO:"<<cg[i]["name"]<<endl;
            }
        }else if(cg[i]["op"]=="prim::Constant"){
            if(cg[i]["shape"].size()==0){ // float
                Tensor c=(float)cg[i]["constant_value"];
                forward_result[i]=new VariableTensor(c);
                cout<<c<<endl;
            }else{ //tensor
                Tensor::shape_type shape=cg[i]["shape"];
                size_t shape_flat=1;
                for(auto s: shape){
                    shape_flat*=s;
                }
                vector<size_t> ss={shape_flat};
                Tensor t(ss);
                for(int j=0;j<shape_flat;j++){
                    t[j]=(float)cg[i]["constant_value"][j];
                }
                t=t.reshape(shape);
                forward_result[i]=new VariableTensor(t);
                cout<<t<<endl;
            }
        }else if(cg[i]["op"]=="aten::mul"){
            int num_inputs=cg[i]["in"].size();
            MulOp* op=new MulOp();
            cout<<"Mul:";
            for(int j=0;j<num_inputs;j++){
                int id=cg[i]["in"][j];
                cout<<id<<" ";
                MCTNode* p_in=forward_result[id];
                op->inputs.push_back(p_in);
            }
            cout<<endl;
            forward_result[i]=op;
        }else if(cg[i]["op"]=="aten::add"){
            int num_inputs=cg[i]["in"].size();
            AddOp* op=new AddOp();
            cout<<"Add:";
            for(int j=0;j<num_inputs;j++){
                int id=cg[i]["in"][j];
                cout<<id<<" ";
                MCTNode* p_in=forward_result[id];
                op->inputs.push_back(p_in);
            }
            cout<<endl;
            forward_result[i]=op;
        }else{
            cout<<"unknown op:"<<cg[i]["op"]<<endl;
        }
    }
    
    cout<<"### forward computation ..."<<endl;
    forward_result[output_id]->forward();
    auto o = forward_result[output_id]->output;
    cout<<o<<endl;

    cout<<"### backward computation ..."<<endl;
    forward_result[output_id]->grad=xt::ones_like(forward_result[output_id]->grad);
    forward_result[output_id]->backward();
    cout<<var_x.grad<<endl;
    return 0;
}
