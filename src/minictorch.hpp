#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>  // 210806 add mari
#include <xtensor/xbroadcast.hpp>    // 210806 add mari
#include <xtensor-blas/xlinalg.hpp>

#define fprec float   // 210702 add mari
typedef xt::xarray<fprec> Tensor;

class MCTNode;

using namespace std;

#define DOT(a,b)  (xt::linalg::dot(a,b))   // 210702 add mari


class MCTNode{
    public:
    MCTNode(){
        frontcnt = 0;  // 210806 add
        backcnt  = 0;  // 210722 add
    }
    virtual bool forward(){
        cout<<"not implemented: forward"<<endl;
        return true;
    }
    virtual bool backward(){
        cout<<"not implemented: backward"<<endl;
        return false;
    }
    std::vector<MCTNode*> inputs;
    std::string name;
    Tensor output;
    Tensor grad;
    int    frontcnt;  // 210806 add mari
    int    backcnt;   // 210722 add mari
    
    // utility function
    void print_message( const char* msg ) // 210726 add
    {
#ifdef _DEBUG
        cout<<msg<<endl;
#endif
    }
    void print_tensor( const char* title, Tensor a ) // 210726 add
    {
#ifdef _DEBUG
        cout<<title<<a<<endl;
#endif
    }
    void print_shape( const char *title, Tensor x ){
        auto s=x.shape();
        cout<<title<<" (";
        for(int i=0;i<s.size();i++){
            cout<<s[i]<<",";
        }
        cout<<")"<<endl; // 210702 add mari
    }
    Tensor _transpose(Tensor a,int ar){ // 210702 move from MatMulOp
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
    void _forward_inputs()
    {
        //cout<<"forward_inputs"<<endl;
        /*for(auto& itr:inputs){  // 210806 mod mari
            if( itr )  itr->forward();
        }*/
    }
    void _forward_inputs2()
    {
        //cout<<"forward_inputs -- "<<id<<endl;
        for(auto& itr:inputs){
            if( itr )  
            {
                //cout<<"forward : itr - "<<id<<" = "<<itr->id<<","<<itr->frontcnt<<endl;
                if( itr->frontcnt == 0 )  
                {
                    itr->frontcnt++;
                    itr->forward();
                }
            }
        }
    }
    void _backward_inputs()
    {
        //cout<<"backward_inputs"<<endl;
        /*for(auto& itr:inputs){ // 210806 mod mari
            if( itr )  itr->backward(); 
        }*/
    }
    void _backward_inputs2()
    {
        //cout<<"backward_inputs"<<endl;
        for(auto& itr:inputs){
            if( itr )
            {
                itr->backcnt--;
                if( itr->backcnt < 1 )  itr->backward();
            }
        }
    }
    void set_inputs( MCTNode *node )  // 210723 add mari
    {
        inputs.push_back( node );
        if( node )  node->backcnt++;
    }
};

class VariableTensor:public MCTNode{
    public:
    VariableTensor(){}
    
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
    ExpOp(){}
    int axis;
    
    bool forward(){
        _forward_inputs();
        print_message( "exp(forward)" );
        this->output = xt::exp(inputs[0]->output);
        return true;
    }
    bool backward(){
        print_message( "exp(backward)" );
        for(auto& itr:inputs){
            itr->grad += this->grad*this->output;
        }
        _backward_inputs();
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
        _forward_inputs();
        print_message( "sum(forward)" );
        if( inputs[1] ) {  // 210724 add
            auto ax = (int)inputs[1]->output[0];
            if( ax == 1 )  this->axis = ax;
        }
        auto s = inputs[0]->output.shape();
        this->size   = s[this->axis];
        this->output = xt::sum(inputs[0]->output,{this->axis});
        return true;
    }
    bool backward(){
        print_message( "sum(backward)" );
        for(auto& itr:inputs){
            if( itr ) {
                auto bb = xt::expand_dims( this->grad,this->axis );
                auto g  = xt::repeat( bb,this->size,this->axis );
                itr->grad += g;  // 210723 mod mari
            }
        }
        _backward_inputs();
    }
};


class AddOp:public MCTNode{
    public:
    AddOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "add(forward)" );
        this->output = inputs[0]->output + inputs[1]->output;
        return true;
    }
    bool backward(){
        print_message( "add(backward)" );
        this->inputs[0]->grad += this->grad;
        this->inputs[1]->grad += this->grad;
        _backward_inputs();
    }
};

class MulOp:public MCTNode{
    public:
    MulOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "mul(forward)" );
        this->output = inputs[0]->output * inputs[1]->output;
        return true;
    }
    bool backward(){
        print_message( "mul(backward)" );
        this->inputs[0]->grad += this->grad * this->output / inputs[0]->output;
        this->inputs[1]->grad += this->grad * this->output / inputs[1]->output;
        _backward_inputs();
    }
};

class NegOp:public MCTNode{
    public:
    NegOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "neg(forward)" );
        output = -inputs[0]->output;
        return true;
    }
    bool backward(){
        print_message( "neg(backward)" );
        inputs[0]->grad -= this->grad;
        _backward_inputs();
        return true;
    }
};

class SubOp:public MCTNode{
    public:
    SubOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "sub(forward)" );
        output = inputs[0]->output - inputs[1]->output;
        return true;
    }
    bool backward(){
        print_message( "sub(backward)" );
        inputs[0]->grad += this->grad;
        inputs[1]->grad -= this->grad;
        _backward_inputs();
        return true;
    }
};

class DivOp:public MCTNode{
    public:
    DivOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "div(forward)" );
        output = inputs[0]->output / inputs[1]->output;
        return true;
    }
    bool backward(){
        print_message( "div(backward)" );
        auto x0 = inputs[0]->output;
        auto x1 = inputs[1]->output;
        inputs[0]->grad += this->grad / x1;
        inputs[1]->grad += this->grad * ( -x0 / (x1*x1) );
        _backward_inputs();
        return true;
    }
};

class PowOp:public MCTNode{  // 210705 add mari
    public:
    PowOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "pow(forward)" );
        output = pow( inputs[0]->output, inputs[1]->output );
        return true;
    }
    bool backward(){
        print_message( "pow(backward)" );
        auto x = inputs[0]->output;
        auto c = inputs[1]->output;
        inputs[0]->grad += this->grad * c * pow( x, c-1.0 );
        _backward_inputs();
        return true;
    }
};

class MatMulOp:public MCTNode{
    public:
    MatMulOp(){}
    
    bool forward(){
        if( inputs.size()!=2 ){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        _forward_inputs();
        print_message( "matmul(forward)" );
        auto a=inputs[0]->output;
        auto b=inputs[1]->output;
        auto as=a.shape();
        auto bs=b.shape();
        int  ar=as.size();
        int  br=bs.size();
        if( ( ar==1 || ar==2 ) && br==2 ){ // 210702 mod mari
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            this->output = DOT(a,b);
            //this->output =xt::linalg::dot(a,b);
        }else if(ar > 2 && br==2){
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            this->output = DOT(a,b);
            //this->output =xt::linalg::dot(a,b);
            print_shape( "a", a );cout<<"x";
            print_shape( "b", b );cout<<"=>";
            print_shape( "output", this->output );
        }else{
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward(){
        if(inputs.size()!=2){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        print_message( "matmul(backward)" );
        auto  a =inputs[0]->output;
        auto  b =inputs[1]->output;
        auto& ga=inputs[0]->grad;
        auto& gb=inputs[1]->grad;
        auto& gc=this->grad;
        auto  as=a.shape();
        auto  bs=b.shape();
        int   ar=as.size();
        int   br=bs.size();
        if(ar==2 && br==2){
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            ga+= DOT( gc, xt::transpose(b) );
            gb+= DOT( xt::transpose(a), gc );
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            //gb+=xt::linalg::dot(xt::transpose(a),gc);
        } else if( ar==1 && br==2 ){ // 210702 add mari
            // matmul:
            //  a.rank: 1
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            //gb+=xt::linalg::dot(a2,g2);
        } else if(ar > 2 && br==2){
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            ga+= DOT(gc,xt::transpose(b));
            print_tensor( "ga", ga );
            print_tensor( "gc", gc );
            cout<<"=="<<endl;
            auto a_t=this->_transpose(a,ar);
            print_shape( "a_t", a_t );
            auto new_a_shape=_batch_shape(a_t,a_t.shape());
            auto batch_a_t=a.reshape(new_a_shape);
            auto new_gc_shape=_batch_shape(gc,gc.shape());
            auto batch_gc=gc.reshape(new_gc_shape);
            //
            print_shape( "gc", gc );
            print_shape( "a_t", a_t );
            print_shape( "b:gc", batch_gc );
            print_shape( "b:a_t", batch_a_t);
            //
            Tensor::shape_type out_s={new_a_shape[0],new_a_shape[1],new_gc_shape[2]};
            Tensor out(out_s);
            for(int i=0;i<new_a_shape[0];i++){
                auto v = xt::view(a_t, i, xt::all(), xt::all());
                auto w = xt::view(gc, i, xt::all(), xt::all());
                //auto temp=xt::linalg::dot(v,w);//a.T gc
                auto temp = DOT(v,w);  //a.T gc
                print_shape( "temp", temp);
                xt::view( out, i, xt::all(), xt::all() ) = temp;
            }
            /*
            vector<int> r;
            for(int i=0;i<ar-2;i++){
                r.push_back(i);
            }
            //cout<<temp<<endl;
            //gb+=xt::sum(temp,r);
            */
        }else{
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
};

class LinearOp:public MCTNode{ // 210702 add mari
    public:
    LinearOp(){}
    
    bool forward(){
        if( inputs.size() != 3 ){
            cout<<"Error:LinearOp input size"<<endl;
            return false;
        }
        _forward_inputs();  
        print_message( "linear(forward)" );
        auto a = inputs[0]->output;  // x
        auto b = inputs[1]->output;  // weight
        auto d = inputs[2]->output;  // bias
        auto as= a.shape();
        auto bs= b.shape();
        auto ds= d.shape();
        int  ar= as.size();
        int  br= bs.size();
        /*
        print_tensor( "linear a", a );
        print_tensor( "linear b", a );
        print_tensor( "linear d", a );
        cout<<"ar:"<<ar<<endl;
        cout<<"br:"<<br<<endl;
        print_shape( "a", a );
        print_shape( "b", b );
        print_shape( "d", d );*/
        if(( ar==1 || ar==2 ) && br==2 ){ 
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            output = DOT(a, xt::transpose(b) ) + d;
            //output =xt::linalg::dot(a,b) + d;
        }else if( ar > 2 && br==2 ){  // yet 
            // batched admm:
            //  a.rank: >2
            //  b.rank: 2 
            output = DOT( a,xt::transpose(b) ) + d;
            //output =xt::linalg::dot(a,b) + d;
            print_shape( "a", a );cout<<"x";
            print_shape( "b", b );cout<<"=>";
            print_shape( "output", this->output );
        }else{
            cout<<"Error:LinearOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward(){
        if( inputs.size() != 3 ){
            cout<<"Error:LinearOp input size"<<endl;
            return false;
        }
        print_message( "linear(backward)" );
        auto  a =inputs[0]->output; // x
        auto  b =inputs[1]->output; // weight
        auto  d =inputs[2]->output; // bias
        auto& ga=inputs[0]->grad;
        auto& gb=inputs[1]->grad;
        auto& gd=inputs[2]->grad;
        auto& gc=this->grad;
        auto  as=a.shape();
        auto  bs=b.shape();
        int   ar=as.size();
        int   br=bs.size();
        //print_shape( "gc", gc );
        //print_shape( "a", a );
        //print_shape( "b", b );
        //cout<<gc<<endl;
        if( ar==2 && br==2 ){
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            //gb+=xt::linalg::dot(xt::transpose(a),gc);
            ga += DOT( gc, b );
            gb += DOT( xt::transpose(gc), a );
            gd += xt::sum( gc,{0} ); // 210721 mod
        } else if( ar==1 && br==2 ){ // yet
            // addmm:
            //  a.rank: 1
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            //gb +=xt::linalg::dot(a2,g2);
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} ); // 210721 mod
        } else if( ar > 2 && br==2 ){  // yet gd
            // batched addmm:
            //  a.rank: >2
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gd,xt::transpose(b));
            ga += DOT( gd, xt::transpose(b) );
            gd += xt::sum( gc,{0} ); // 210721 mod
            
            print_tensor( "linear ga", ga );
            print_tensor( "linear gc", gc );
            cout<<"=="<<endl;
            auto a_t=this->_transpose(a,ar);
            print_shape( "a_t", a_t );
            
            auto new_a_shape  =_batch_shape(a_t,a_t.shape());
            auto batch_a_t    = a.reshape(new_a_shape);
            auto new_gc_shape =_batch_shape(gc,gc.shape());
            auto batch_gc     = gd.reshape(new_gc_shape);
            //
            print_shape( "gc", gc );
            print_shape( "a_t", a_t );
            print_shape( "b:gc", batch_gc );
            print_shape( "b:a_t", batch_a_t );
            //
            Tensor::shape_type out_s={new_a_shape[0],new_a_shape[1],new_gc_shape[2]};
            Tensor out(out_s);
            for(int i=0;i<new_a_shape[0];i++)  // yet gd
            {
                auto v = xt::view(a_t, i, xt::all(), xt::all());
                auto w = xt::view(gc,  i, xt::all(), xt::all());
                //auto temp=xt::linalg::dot(v,w);//a.T gc
                auto temp = DOT(v,w); // a.T gc
                print_shape( "temp", temp );
                xt::view(out, i, xt::all(), xt::all() ) = temp;
            }
            
            /*
            vector<int> r;
            for(int i=0;i<ar-2;i++){
                r.push_back(i);
            }
            //cout<<temp<<endl;
            //gb+=xt::sum(temp,r);
            */
        }else{
            cout<<"Error:LinearOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
};

class AddMmOp:public MCTNode{ 
    public:
    AddMmOp(){}
    
    bool forward(){
        if( inputs.size()!=5 ){
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
         _forward_inputs();  
        print_message( "addmm(forward)" );
        auto d=inputs[0]->output;  // bias
        auto a=inputs[1]->output;  // x
        auto b=inputs[2]->output;  // weight
        auto as=a.shape();
        auto bs=b.shape();
        auto ds=d.shape();
        int  ar=as.size();
        int  br=bs.size();
        /*
        print_tensor( "a", a );
        print_tensor( "b", b );
        print_tensor( "d", d );
        cout<<"ar:"<<ar<<endl;
        cout<<"br:"<<br<<endl;
        print_shape( "a", a );
        print_shape( "b", b );
        print_shape( "d", d );*/
        if(( ar==1 || ar==2 ) && br==2 ){ 
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            output = DOT(a,b) + d;
            //this->output =xt::linalg::dot(a,b) + d;
        }else if( ar > 2 && br==2 ){
            // batched admm:
            //  a.rank: >2
            //  b.rank: 2 
            output = DOT(a,b) + d;
            //this->output =xt::linalg::dot(a,b + d;
            /*
            print_shape( "a", a );cout<<"x";
            print_shape( "b", b );cout<<"=>";
            print_shape( "output", this->output );*/
        }else{
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward(){
        if( inputs.size()!=5 ){
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
        print_message( "addmm(backward)" );
        auto  d =inputs[0]->output; // bias
        auto  a =inputs[1]->output; // x
        auto  b =inputs[2]->output; // weight
        auto& gd=inputs[0]->grad;
        auto& ga=inputs[1]->grad;
        auto& gb=inputs[2]->grad;
        auto& gc=this->grad;
        auto  as=a.shape();
        auto  bs=b.shape();
        int   ar=as.size();
        int   br=bs.size();
        //print_shape( "gc", gc );
        //print_shape( "a", a );
        //print_shape( "b", b );
        //print_tensor( "gc", gc );
        if( ar==2 && br==2 ){
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            //gb+=xt::linalg::dot(xt::transpose(a),gc);
            ga += DOT( gc, xt::transpose(b) );
            gb += DOT( xt::transpose(a), gc );
            gd += xt::sum( gc,{0} ); // 210721 mod
        } else if( ar==1 && br==2 ){
            // addmm:
            //  a.rank: 1
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gc,xt::transpose(b));
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            //gb +=xt::linalg::dot(a2,g2);
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} ); // 210721 mod
        } else if( ar > 2 && br==2 ){  // yet gd
            // batched addmm:
            //  a.rank: >2
            //  b.rank: 2 
            //ga+=xt::linalg::dot(gd,xt::transpose(b));
            ga += DOT( gd, xt::transpose(b) );
            gd += xt::sum( gc,{0} ); // 210721 mod
            
            print_tensor( "ga", ga );
            print_tensor( "gc", gc );
            cout<<"=="<<endl;
            auto a_t=this->_transpose(a,ar);
            print_shape( "a_t", a_t);
            
            auto new_a_shape  =_batch_shape(a_t,a_t.shape());
            auto batch_a_t    = a.reshape(new_a_shape);
            auto new_gc_shape =_batch_shape(gc,gc.shape());
            auto batch_gc     = gd.reshape(new_gc_shape);
            //
            print_shape( "gc", gc );
            print_shape( "a_t", a_t );
            print_shape( "b:gc", batch_gc );
            print_shape( "b:a_t", batch_a_t );
            //
            Tensor::shape_type out_s={new_a_shape[0],new_a_shape[1],new_gc_shape[2]};
            Tensor out(out_s);
            for(int i=0;i<new_a_shape[0];i++)  // yet gd
            {
                auto v = xt::view(a_t, i, xt::all(), xt::all());
                auto w = xt::view(gc,  i, xt::all(), xt::all());
                //auto temp=xt::linalg::dot(v,w);//a.T gc
                auto temp = DOT(v,w); // a.T gc
                print_shape( "temp", temp );
                xt::view(out, i, xt::all(), xt::all() ) = temp;
            }
            
            /*
            vector<int> r;
            for(int i=0;i<ar-2;i++){
                r.push_back(i);
            }
            //cout<<temp<<endl;
            //gb+=xt::sum(temp,r);
            */
        }else{
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
};

class TransposeOp:public MCTNode{
    public:
    TransposeOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "transpose(forward)" );
        output = xt::transpose( inputs[0]->output );
        return true;
    }
    bool backward(){
        print_message( "transpose(backward)" );
        inputs[0]->grad += xt::transpose( this->grad );
        _backward_inputs();
        return true;
    }
};

class SigmoidOp:public MCTNode{
    public:
    SigmoidOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "sigmoid(forward)" );
        output = 1.0 / ( 1.0+xt::exp( -inputs[0]->output ) );
        //print_tensor( "sigmoid", output );
        return true;
    }
    bool backward(){
        print_message( "sigmoid(backward)" );
        auto y = output;
        inputs[0]->grad += this->grad * y * (1.0-y);
        //print_tensor( "sigmoid grad", inputs[0]->grad );
        _backward_inputs();
        return true;
    }
};

class ReluOp:public MCTNode{
    public:
    ReluOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "relu(forward)" );
        output = xt::maximum( inputs[0]->output, 0 );
        return true;
    }
    bool backward(){
        print_message( "relu(backward)" );
        inputs[0]->grad += this->grad * ( inputs[0]->output > 0 );
        _backward_inputs();
        return true;
    }
};

class HardTanhOp:public MCTNode{ // 210806 add mari
    public:
    HardTanhOp( fprec v1=-1.0, fprec v2=1.0 ){
        min_val = v1;
        max_val = v2;
    }
    fprec  min_val;
    fprec  max_val;

    bool forward(){
        _forward_inputs();
        print_message( "hardtanh(forward)" );
        min_val = (fprec)inputs[1]->output[0];
        max_val = (fprec)inputs[2]->output[0];
        output = xt::maximum( inputs[0]->output, min_val );
        output = xt::minimum( output, max_val );
        return true;
    }
    bool backward(){
        print_message( "hardtanh(backward)" );
        auto  y  = inputs[0]->output;
        auto &gd = inputs[0]->grad;
        gd += this->grad * ( y > min_val && y < max_val );
        _backward_inputs();
        return true;
    }
};

class EluOp:public MCTNode{  // 210806 add mari
    public:
    EluOp(fprec a=1.0){
       alpha = a;
    }
    fprec alpha;
    xt::xarray<bool> mask;
    
    bool forward(){
        _forward_inputs();
        print_message( "elu(forward)" );
        auto y = inputs[0]->output;
        alpha  = (fprec)inputs[1]->output[0];
        output = xt::maximum( y, 0.0 );
        mask = ( y < 0.0 );
        auto m = xt::masked_view( output, mask );
        m = xt::minimum( alpha * ( xt::exp( y ) - 1.0 ), 0.0 );
        return true;
    }
    bool backward(){
        print_message( "elu(backward)" );
        auto  y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd = this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = alpha * xt::exp( y );
        _backward_inputs();
        return true;
    }
};

class LeakyReluOp:public MCTNode{ // 210806 add mari
    public:
    LeakyReluOp( fprec s=0.01 ){
        slope=s;
    }
    fprec slope;
    xt::xarray<bool> mask;
    
    bool forward(){
        _forward_inputs();
        print_message( "leakyrelu(forward)" );
        auto y = inputs[0]->output;
        slope  = (fprec)inputs[1]->output[0];
        mask = ( y < 0.0 );
        output = y;
        auto m = xt::masked_view( output, mask );
        m = y * slope;
        return true;
    }
    bool backward(){
        print_message( "leakyrelu(backward)" );
        auto  y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd += this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = gd * slope;
        _backward_inputs();
        return true;
    }
};

class SoftmaxOp:public MCTNode{
    public:
    SoftmaxOp(int ax=1){
        this->axis = ax;
    }
    int axis;
    
    bool forward(){
        _forward_inputs();
        print_message( "softmax(forward)" );
        auto  a = inputs[0]->output;
        auto  as = a.shape();
        int   ar = as.size();
        Tensor sm = xt::amax( a, {axis} );
        Tensor sa = a - sm;
        Tensor se = xt::exp( sa );
        Tensor sd = xt::sum( se, {axis} );
        /*
        print_tensor( "a ", a );
        cout<<"sm"<<sm<<axis<<endl;
        print_tensor( "sa", sa );
        print_tensor( "se", se );
        print_tensor( "sd", sd );*/
        output = se / sd;
        return true;
    }
    bool backward(){
        print_message( "softmax(backward)" );
        auto  as = output.shape();
        int   ar = as.size();
        Tensor ga = output * this->grad;
        Tensor sg = xt::sum( ga, {axis} );
        /*
        print_tensor( "softmax output", output );
        print_tensor( "softmax grad", grad );
        print_tensor( "softmax ga", ga );
        print_tensor( "softmax sg", sg );*/
        inputs[0]->grad += ( ga - output * sg );
        _backward_inputs();
        return true;
    }
};

class LogSoftmaxOp:public MCTNode{
    public:
    LogSoftmaxOp(int ax=1){
        axis=ax;
    }
    int axis;
    
    bool forward(){
        _forward_inputs();
        print_message( "log_softmax(forward)" );
        auto  a = inputs[0]->output;
        auto  as = a.shape();
        int   ar = as.size();
        Tensor sm = xt::amax( a, {axis} );
        Tensor sa = a - sm;
        Tensor se = xt::exp( sa );
        Tensor sd = xt::sum( se, {axis} );
        Tensor sl = xt::log( sd );
        /*
        print_tensor( "a ", a );
        cout<<"sm"<<sm<<axis<<endl;
        print_tensor( "sa", sa );
        print_tensor( "se", se );
        print_tensor( "sd", sd );
        print_tensor( "sl", sl );*/
        output = a - ( sm + sl );
        return true;
    }
    bool backward(){
        print_message( "log_softmax(backward)" );
        auto  as = output.shape();
        int   ar = as.size();
        Tensor& ga = this->grad;
        Tensor  sg = xt::sum( ga, {axis} );
        Tensor  se = xt::exp( output );
        inputs[0]->grad += ( ga - se * sg );
        _backward_inputs();
        return true;
    }
};

class TanhOp:public MCTNode{
    public:
    TanhOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "tanh(forward)" );
        output = xt::tanh( inputs[0]->output );
        return true;
    }
    bool backward(){
        print_message( "tanh(backward)" );
        inputs[0]->grad += this->grad * ( 1.0 - output * output );
        _backward_inputs();
        return true;
    }
};

class RandnOp:public MCTNode{  // 210806 add below mari
    public:
    RandnOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "randn(forward)" );
        auto sh = inputs[0]->output;
        int s0 = (int)sh[0];
        int s1 = (int)sh[1];
        output = xt::random::randn<float>( {s0,s1} );
        print_tensor( "randn", output );
        return true;
    }
    bool backward(){
        _backward_inputs();
        return true;
    }
};

class NormalOp:public MCTNode{
    public:
    NormalOp(){
        shape = {1,1};
    }
    Tensor::shape_type shape;
    
    void set_shape( Tensor::shape_type sh ){
        shape = sh;
    }
    bool forward(){
        _forward_inputs();
        print_message( "normal(forward)" );
        fprec mean = (fprec)inputs[0]->output[0];
        fprec std  = (fprec)inputs[1]->output[0];
        output = xt::random::randn<float>( shape, mean, std );
        //print_shape("randn", shape );
        cout<<"normal mean std"<<mean<<std<<endl;
        print_tensor( "normal", output );
        return true;
    }
    bool backward(){
        _backward_inputs();
        return true;
    }
};

class MseLossOp:public MCTNode{
    public:
    MseLossOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "mseloss(forward)" );
        auto a = inputs[0]->output;
        auto b = inputs[1]->output;
        int  type = (int)inputs[2]->output[0];
        auto diff = a - b;
        output = xt::sum( xt::pow( diff, 2.0 ) );
        if( type == 1 ) {
            output = output/(double)a.size();
        }
        cout<<"mseloss "<<a.size()<<","<<type<<endl;
        print_tensor( "mesloss", output );
        return true;
    }
    bool backward(){
        print_message( "mseloss(backward)" );
        auto a = inputs[0]->output;
        auto b = inputs[1]->output;
        int  type = (int)inputs[2]->output[0];
        auto diff = a - b;
        auto& ga = inputs[0]->grad;
        auto& gb = inputs[1]->grad;
        auto& gc = this->grad;
        ga = gc * diff * 2.0;
        if( type == 1 ) {
            ga = ga / (double)a.size();
        }
        gb = -ga;
        print_tensor( "mseloss_grad", ga );
        _backward_inputs();
        return true;
    }
};

class CrossEntropyLossOp:public MCTNode{  // #210806 == log_softmax
    public:
    CrossEntropyLossOp( int ax=1 ){
        axis = ax;
    }
    xt::xarray<bool> mask;
    int axis;
    
    bool forward(){
        _forward_inputs();
        print_message( "cross_entropy_loss(forward)" );
        auto   a  = inputs[0]->output;
        auto   as = a.shape();
        //cout<<"a "<<a<<endl;
        cout<<"ashape"<<as[0]<<as[1]<<endl;
        Tensor sm = xt::amax( a, {axis} );
        sm.reshape({-1,1} );
        //print_tensor( "sm", sm );
        auto   sa = a - sm;
        auto   se = xt::exp( sa );
        auto   sd = xt::sum( se, {axis} );
        Tensor sl = xt::log( sd );
        sl.reshape( {-1,1} );
        auto   sz = a - ( sm + sl );
        //print_tensor( "sa", sa );
        //print_tensor( "se", se );
        //print_tensor( "sd", sd );
        //print_tensor( "sl", sl );
        //print_tensor( "sz", sz );
        
        fprec  h = (fprec)inputs[4]->output[0];
        //cout<<"h"<<h<<endl;
        //mask = ( output < h );
        //auto m = xt::masked_view( output, mask );
        //m = h;
        
        xt::xarray<float> t = xt::zeros<float>( {as[0]} );
        for(int i=0;i<as[0];i++){
            int j = (int)inputs[1]->output[i];
            t[i] = sz( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / (fprec)as[0];
        print_tensor( "crossloss", output );
        return true;
    }
    bool backward(){
        print_message( "cross_entropy_loss(backward)" );
        auto  a  = inputs[0]->output;
        auto  as = a.shape();
        auto  ga = this->grad;
        fprec sc = 1.0 / (fprec)as[0];
        
        //cout<<"a "<<a<<endl;
        cout<<"ashape"<<as[0]<<","<<as[1]<<endl;
        Tensor sm = xt::amax( a, {axis} );
        sm.reshape( {-1,1} );
        //cout<<"sm"<<sm<<endl;
        auto   sa = a - sm;
        auto   se = xt::exp( sa );
        Tensor sd = xt::sum( se, {axis} );
        sd.reshape( {-1,1} );
        //print_tensor( "sa", sa );
        //print_tensor( "se", se );
        //print_tensor( "sd", sd );
        auto sd2 = xt::broadcast( sd, as );
        //print_tensor( "sd2", sd2 );
        auto y = se / sd2;
        print_tensor( "y", y );
        
        Tensor one = xt::zeros<float>( as );
        for(int i=0;i<as[0];i++){
            int j = (int)inputs[1]->output[i];
            one(i,j) = 1.0;
        }
        //print_tensor( "one", one );
        //cout<<"ga"<<ga<<","<<sc<<endl;
        
        inputs[0]->grad = ( y - one ) * ga * sc;
        print_tensor( "crossloss_grad", inputs[0]->grad );
        _backward_inputs();
      
        return true;
    }
};

class ListConstructOp:public MCTNode{
    public:
    ListConstructOp() {}
    ListConstructOp( string na ) { name = na; }
    
    bool forward(){
        _forward_inputs();
        print_message( "ListConstruct(forward)" );
        if( inputs.size() < 2 )
        {
            cout<<"Error:ListConstructOp"<<endl;
            return false;
        }
        int err = 0;
        int sz = -1;
        for( int i=0;i<inputs.size();i++)
        {
            if( inputs[i] ) {
                auto a = inputs[i]->output;
                auto as = a.shape();
                int  ar = as.size();
                if( sz < 0 ) {
                    sz = ar;
                } else {
                    if( sz != ar )  err++;
                }
            } else {
                if( sz < 0 ) {
                    sz = 0;
                } else {
                    if( sz != 0 )  err++;
                }
            }
        }
        if( err > 0 ) {
            cout<<"Error:ListConstructOp(mismatch)"<<endl;
            return false;
        }
        cout<<"list size "<<sz<<endl;
        
        Tensor f0,f1;
        if( sz == 0 ) {
            Tensor f0 = {(float)0.0};
            Tensor f1 = {(float)0.0};
            if( inputs[0] ) f0 = {(float)inputs[0]->output[0]};
            if( inputs[1] ) f1 = {(float)inputs[1]->output[0]};
            cout<<"f0,f1"<<f0<<","<<f1<<endl;
            output = xt::concatenate(xt::xtuple(f0,f1));
        } else {
            Tensor f0 = inputs[0]->output;
            Tensor f1 = inputs[1]->output;
            cout<<"f0"<<f0<<endl;
            cout<<"f1"<<f1<<endl;
            auto f00 = xt::expand_dims( f0, {0} );
            auto f11 = xt::expand_dims( f1, {0} );
            output = xt::concatenate(xt::xtuple(f00,f11));
        }
        
        for(int i=2;i<inputs.size();i++)
        {
            Tensor f2;
            if( sz == 0 ) {
                Tensor f2 = {(float)0.0};
                if( inputs[i] )  f2 = {(float)inputs[i]->output[0]};
                output = xt::concatenate(xt::xtuple(output,f2));
            } else {
                Tensor f2 = inputs[0]->output;
                auto  f22 = xt::expand_dims( f2, {0} );
                output = xt::concatenate(xt::xtuple(output,f22));
            }
        }
        cout<<"ListConstruct "<<name<<output<<endl;
        return true;
    }
    bool backward(){
        print_message( "ListConstruct(backward)" );
        auto& gc = this->grad;
        auto gs = gc.shape();
        int  gz = gs.size();
        cout<<"gz"<<gz<<endl;
        if( gz == 0 ) {
            // yet
        } else if( gz < inputs.size() ) {
            // yet
        } else {
            for(int i=0;i<inputs.size();i++){
                if( inputs[i] ) {
                    inputs[i]->grad = xt::view( gc, i );
                    cout<<"ListConstruct "<<name<<","<<i<<" "<<inputs[i]->grad<<endl;
                }
            }
        }
        _backward_inputs();
        return true;
    }
};

using namespace xt::placeholders; // 210806 add

class ListUnpackOp:public MCTNode{
    public:
    ListUnpackOp() {}
    ListUnpackOp( string na ) { name = na; }
    
    bool forward(){
        _forward_inputs();
        print_message( "ListUnpack(forward)" );
        auto a = inputs[0]->output;
        int ar = a.size();
        if( ar < 1 ) {
            cout<<"Error:ListUnpackOp"<<endl;
            return false;
        }
        output = xt::view( a, 0 );
        inputs[0]->output = xt::view( a, xt::range(1,_) );
        cout<<"ListUnpack "<<name<<output<<endl;
        return true;
    }
    bool backward(){
        print_message( "ListUnpack(backward)" );
        auto  a  = inputs[0]->output;
        auto& ga = inputs[0]->grad;
        auto& gc = this->grad;
        int  nga = ga.size();
        cout<<"ListUnpack size"<<nga<<","<<ga<<endl;
        if( nga == 1 ) {  // yet
            ga = xt::expand_dims( gc, {0} );
        } else {
            auto gb = xt::expand_dims( gc, {0} );
            ga = xt::concatenate( xt::xtuple(gb,ga) );
        }
        cout<<"ListUnpack "<<name<<ga<<endl;
        _backward_inputs();
        return true;
    }
};

class SizeOp:public MCTNode{ // 210802 add
    public:
    SizeOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "size(forward)" );
        auto  a  = inputs[0]->output;
        auto  as = a.shape();
        int   ar = as.size();
        auto  no = (int)inputs[1]->output[0];
        output = (float)as[no];
        cout<<"size "<<no<<","<<output<<endl;
        return true;
    }
    bool backward(){
        _backward_inputs();
        return true;
    }
};

class ExpandOp:public MCTNode{  // 210802 add
    public:
    ExpandOp(){}
    
    bool forward(){
        _forward_inputs();
        print_message( "expand(forward)" );
        Tensor f0 = { (float)0.0};
        int l = 0;
        for(int i=0;i<inputs.size();i++)
        {
            for(int j=0;j<inputs[i]->output.size();j++)
            {
                Tensor f1 = { (float)inputs[i]->output[j] };
                if( l == 0 ) {
                    f0 = f1;
                } else if( l == 1 ) {
                    output = xt::concatenate( xt::xtuple(f0,f1) );
                } else if( l > 1 ) {
                    output = xt::concatenate( xt::xtuple(output,f1) );
                }
                l++;
            }
        }
        cout<<"expand"<<output<<endl;
        return true;
    }
    bool backward(){
        _backward_inputs();
        return true;
    }
};

// move input[0] to output simply.
//   prim::NumtoTensor
//   aten::Int
//   aten::detach
//   aten::broadcast_tensors
class MoveOp:public MCTNode{
    public:
    MoveOp() {}
    MoveOp( string na ){ name = na; }
    
    bool forward(){
        _forward_inputs();
        print_message( "Move(forward)" );
        output = inputs[0]->output;
        cout<<"MoveOp "<<name<<output<<endl;
        return true;
    }
    bool backward(){
        print_message( "Move(backward)" );
        auto& ga = inputs[0]->grad;
        inputs[0]->grad += this->grad;
        cout<<"MoveOp "<<name<<ga<<endl;
        _backward_inputs();
        return true;
    }
};


class EinsumOp:public MCTNode{
    public:
    EinsumOp(){}
    
    bool forward(){
        return true;
    }
};

