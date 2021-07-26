#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define fprec float   // 210702 add mari
typedef xt::xarray<fprec> Tensor;

class MCTNode;

using namespace std;

#define DOT(a,b)    (xt::linalg::dot(a,b))   // 210702 add mari


class MCTNode{
    public:
    MCTNode(){
        backcnt=0;  // 210722 add
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
    //std::vector<MCTNode*> unique_inputs;  // 210723 del
    std::string name;
    Tensor output;
    Tensor grad;
    int    backcnt;  // 210722 add mari
    
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
    void _forward_inputs()
    {
        //cout<<"forward_inputs"<<endl;
        for(auto& itr:inputs){
            if( itr )  itr->forward();
        }
    }
    void _backward_inputs()
    {
        //cout<<"backward_inputs"<<endl;
        //for(auto& itr:unique_inputs){  // 210723 mod
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
    void set_unique_inputs( MCTNode *node )  // 210723 del
    {
        //unique_inputs.push_back( node );
        //if( node ) node->backcnt++;
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
        this->output = xt::exp(inputs[0]->output);
        return true;
    }
    bool backward(){
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
        if( inputs[1] ) {  // 210723 add
            auto ax = (int)inputs[1]->output[0];
            if( ax == 1 )  this->axis = ax;
        }
        auto s = inputs[0]->output.shape();
        this->size   = s[this->axis];
        this->output = xt::sum(inputs[0]->output,{this->axis});
        return true;
    }
    bool backward(){
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
        this->output = inputs[0]->output + inputs[1]->output;
        //for(int i=1;i<inputs.size();i++){
        //    this->output=this->output+inputs[i]->output;
        //}
        return true;
    }
    bool backward(){
        this->inputs[0]->grad += this->grad;
        this->inputs[1]->grad += this->grad;
        //for(auto& itr:inputs){
        //    itr->grad+=this->grad;
        //}
        _backward_inputs();
    }
};

class MulOp:public MCTNode{
    public:
    MulOp(){}
    
    bool forward(){
        _forward_inputs();
        this->output = inputs[0]->output * inputs[1]->output;
        //this->output=inputs[0]->output;
        //for(int i=1;i<inputs.size();i++){
        //    this->output=this->output*inputs[i]->output;
        //}
        return true;
    }
    bool backward(){
        this->inputs[0]->grad += this->grad * this->output / inputs[0]->output;
        this->inputs[1]->grad += this->grad * this->output / inputs[1]->output;
        //for(auto& itr:inputs){
        //    itr->grad+=this->grad*this->output/itr->output;
        //}
        _backward_inputs();
    }
};

class NegOp:public MCTNode{
    public:
    NegOp(){}
    
    bool forward(){
        _forward_inputs();
        output = -inputs[0]->output;
        return true;
    }
    bool backward(){
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
        output = inputs[0]->output - inputs[1]->output;
        return true;
    }
    bool backward(){
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
        output = inputs[0]->output / inputs[1]->output;
        return true;
    }
    bool backward(){
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
        output = pow( inputs[0]->output, inputs[1]->output );
        return true;
    }
    bool backward(){
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
    /* 210702 move to MCTNode
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
    }*/
    bool backward(){
        if(inputs.size()!=2){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
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
        //cout<<output<<endl;
        return true;
    }
    bool backward(){
        print_message( "sigmoid(backward)" );
        auto y = output;
        inputs[0]->grad += this->grad * y * (1.0-y);
        //cout<<"sigmoid grad"<<inputs[0]->grad<<endl;
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

class EinsumOp:public MCTNode{
    public:
    EinsumOp(){}
    
    bool forward(){
        return true;
    }
};

