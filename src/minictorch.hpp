#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xbroadcast.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define UINT  long unsigned int

#define fprec float
typedef xt::xarray<fprec> Tensor;

extern bool train_mode;

using namespace std;

#define DOT(a,b)  (xt::linalg::dot(a,b))


class MCTNode{
    public:
    MCTNode(){
        frontcnt = 0;
        backcnt  = 0;
        id = -1; // for check
        grad_calc = true;
    }
    virtual bool forward()
    {
        cout<<"not implemented: forward"<<endl;
        return true;
    }
    virtual bool backward()
    {
        cout<<"not implemented: backward"<<endl;
        return false;
    }
    std::vector<MCTNode*> inputs;
    std::string name;
    Tensor output;
    Tensor grad;
    bool   grad_calc;
    int    frontcnt;
    int    backcnt;
    int    id; // for check
    
    void set_inputs( MCTNode *node )
    {
        inputs.push_back( node );
        if( node )  node->backcnt++;
    }
    
    virtual void update( fprec delta ) {};
    virtual void set_output1( fprec o1 ) {};
    
    virtual vector<Tensor>*   get_tlist() { return NULL; };
    virtual vector<Tensor>*   get_glist() { return NULL; };
    virtual vector<MCTNode*>* get_ndlist(){ return NULL; };
    
    // utility function
    void set_id( int n ) { id = n; };
    void zerograd()
    {
        //this->grad = xt::zeros_like( output );
        this->grad = 0.;
    }
    Tensor& get_output()    { return output; }
    bool is_grad()          { return grad_calc; };
    void set_grad( bool g ) { grad_calc = g; };
    
    // for debug
    void print_message( const char* msg ) 
    {
#ifdef _DEBUG
        cout<<msg<<endl;
#endif
    }
    void print_tensor( const char* title, Tensor a )
    {
#ifdef _DEBUG
        cout<<title<<a<<endl;
#endif
    }
    void print_shape( const char *title, Tensor x )
    {
#ifdef _DEBUG
        auto s=x.shape();
        cout<<title<<" (";
        for(int i=0;i<s.size();i++){
            cout<<s[i]<<",";
        }
        cout<<")"<<endl;
#endif
    }

    void _forward_inputs()
    {
        //cout<<"forward_inputs -- "<<id<<endl;
        /*for(auto& itr:inputs){ 
            if( itr )  itr->forward();
        }*/
        /*for(auto& itr:inputs){
            if( itr ){
                //cout<<"forward : itr - "<<id<<" = "<<itr->id<<","<<itr->frontcnt<<endl;
                if( itr->frontcnt == 0 ){
                    itr->frontcnt++;
                    itr->forward();
                }
            }
        }*/
    }
    void _backward_inputs()
    {
        //cout<<"backward_inputs"<<endl;
        /*for(auto& itr:inputs){ 
            if( itr )  itr->backward(); 
        }*/
        /*for(auto& itr:inputs){
            if( itr ){
                itr->backcnt--;
                if( itr->backcnt < 1 )  itr->backward();
            }
        }*/
    }
    void get_items( UINT *q, int n, int dum=0 )
    {
        std::vector<Tensor>* ptr = get_tlist(); 
        if( ptr )
        {
            for(int i=0;i<n;i++)
            {
                auto t = ptr->at(i);
                q[i] = (UINT)t[0];
            }
        } else {
            for(int i=0;i<n;i++)  q[i] = dum;
        }
    }
};

class VariableTensor:public MCTNode{
    public:
    VariableTensor(){}
    
    VariableTensor( Tensor tensor, bool g=true ){
        this->output = tensor;
        this->grad_calc = g;
        this->grad = xt::zeros_like( output );
    }
    VariableTensor( string name, Tensor tensor, bool g=true ){
        this->output = tensor;
        this->name = name;
        this->grad_calc = g;
        this->grad = xt::zeros_like( output );
    }
    bool forward()  { return true; }
    bool backward() { return true; }
    void set_output1( fprec o1 ) { output[0] = o1; }
};

class ExpOp:public MCTNode{
    public:
    ExpOp(){}
    int axis;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "exp(forward)" );
        output = xt::exp( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "exp(backward)" );
        for(auto& itr:inputs){
            itr->grad += this->grad * this->output;
        }
        _backward_inputs();
        return true;
    }
};

class LogOp:public MCTNode{ 
    public:
    LogOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "log(forward)" );
        output = xt::log(inputs[0]->output);
        return true;
    }
    bool backward()
    {
        print_message( "log(backward)" );
        inputs[0]->grad += this->grad / inputs[0]->output;
        _backward_inputs();
        return true;
    }
};

class Log1pOp:public MCTNode{ 
    public:
    Log1pOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "log1p(forward)" );
        output = xt::log( inputs[0]->output + 1.0 );
        return true;
    }
    bool backward()
    {
        print_message( "log1p(backward)" );
        inputs[0]->grad += this->grad / ( inputs[0]->output + 1.0 );
        _backward_inputs();
        return true;
    }
};

class SumOp:public MCTNode{
    public:
    SumOp( int ax=-1 ){ axis = ax; }
    int axis;  // sum kind  -1:all 0:row 1:column
    
    bool forward()
    {
        _forward_inputs();
        print_message( "sum(forward)" );
        if( inputs[1] ) { 
            axis = (int)inputs[1]->output[0];
            output = xt::sum( inputs[0]->output, {axis} );
        } else {
            if( axis < 0 ) {
                auto o = xt::sum( inputs[0]->output );
                output = xt::expand_dims( o, 0 );
            } else {
                output = xt::sum( inputs[0]->output, {axis} );
            }
        }
        return true;
    }
    bool backward()
    {
        print_message( "sum(backward)" );
        auto sh = inputs[0]->output.shape();
        int  sz = sh.size();
        
        if( axis >=0  ) { 
            auto e = xt::expand_dims( this->grad, axis );
            auto g = xt::repeat( e, sh[0], axis );
            inputs[0]->grad = g;
        } else {
            auto e  = xt::expand_dims( this->grad, 0 );
            auto g1 = xt::repeat( e, sh[0], 0 );
            if( sz > 1 && sh[1] > 1 ) {
                auto g2 = xt::repeat( g1, sh[1], 1 );
                inputs[0]->grad = g2;
            } else {
                inputs[0]->grad = g1;
            }
        }
        _backward_inputs();
        return true;
    }
};

class MeanOp:public MCTNode{
    public:
    MeanOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "mean(forward)" );
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                auto& a = tlist->at(0);
                cout<<"mean a"<<a<<","<<a.size()<<endl;
                output = a / a.size();
                cout<<"mean o"<<output<<endl;
                return true;
            } else {
                auto& a = inputs[0]->output;
                output = a / a.size();
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        //print_message( "mean(backward)" );
        _backward_inputs();
        return true;
    }
};

class SelectOp:public MCTNode{
    public:
    SelectOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "select(forward)" );
        if( inputs[0] )
        {
            int dim = (int)inputs[1]->output[0];
            int idx = (int)inputs[2]->output[0];
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                auto& a  = tlist->at(0);
                auto  as = a.shape();
                int   ar = as.size();
                xt::xstrided_slice_vector sv;
                for( int i=0;i<ar;i++)
                {
                    if( i == dim ) {
                        sv.push_back( idx );
                    } else {
                        sv.push_back( xt::all() );
                    }
                }
                output = xt::strided_view(a, sv);
                cout<<"select a"<<a<<endl;
                cout<<"select o"<<output<<endl;
                return true;
            } else {
                cout<<"no implement select tensor type"<<endl;
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        //print_message( "select(backward)" );
        _backward_inputs();
        return true;
    }
};

class Copy_Op:public MCTNode{
    public:
    Copy_Op(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "copy_(forward)" );
        auto& a = inputs[0]->output;
        auto& b = inputs[1]->output;
        a = b;
        return true;
    }
    bool backward()
    {
        print_message( "copy_(backward)" );
        auto& ga = inputs[0]->grad;
        auto& gb = inputs[1]->grad;
        gb = ga;
        _backward_inputs();
        return true;
    }
};
    
class AddOp:public MCTNode{
    public:
    AddOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "add(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        output = inputs[0]->output + inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "add(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad;
        if( inputs[1]->is_grad() )
            inputs[1]->grad += this->grad * s;
        _backward_inputs();
        return true;
    }
};

class MulOp:public MCTNode{
    public:
    MulOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "mul(forward)" );
        this->output = inputs[0]->output * inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "mul(backward)" );
        if( inputs[0]->is_grad() )
            inputs[0]->grad += this->grad * inputs[1]->output;
        if( inputs[1]->is_grad() )
            inputs[1]->grad += this->grad * inputs[0]->output;
        _backward_inputs();
        return true;
    }
};

class NegOp:public MCTNode{
    public:
    NegOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "neg(forward)" );
        output = -inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "neg(backward)" );
        inputs[0]->grad -= this->grad;
        _backward_inputs();
        return true;
    }
};

class SubOp:public MCTNode{
    public:
    SubOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "sub(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        output = inputs[0]->output - inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "sub(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() )
            inputs[0]->grad += this->grad;
        if( inputs[1]->is_grad() )
            inputs[1]->grad -= this->grad * s;
        _backward_inputs();
        return true;
    }
};

class RsubOp:public MCTNode{
    public:
    RsubOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "rsub(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        output = inputs[1]->output - inputs[0]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "rsub(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        inputs[0]->grad -= this->grad * s;
        _backward_inputs();
        return true;
    }
};

class DivOp:public MCTNode{
    public:
    DivOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "div(forward)" );
        output = inputs[0]->output / inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "div(backward)" );
        auto& x0 = inputs[0]->output;
        auto& x1 = inputs[1]->output;
        if( inputs[0]->is_grad() )
            inputs[0]->grad += this->grad / x1;
        if( inputs[1]->is_grad() )
            inputs[1]->grad += this->grad * ( -x0 / (x1*x1) );
        _backward_inputs();
        return true;
    }
};

class PowOp:public MCTNode{
    public:
    PowOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "pow(forward)" );
        output = pow( inputs[0]->output, inputs[1]->output );
        return true;
    }
    bool backward()
    {
        print_message( "pow(backward)" );
        auto& x = inputs[0]->output;
        fprec c = (fprec)inputs[1]->output[0];
        inputs[0]->grad += this->grad * c * pow( x, c-1.0 );
        _backward_inputs();
        return true;
    }
};

class MatMulBase:public MCTNode{
    public:
    MatMulBase(){}
    
    Tensor _batch_transpose( Tensor a ) 
    {
        auto as = a.shape();
        int  ar = as.size();
        vector<int> perm;
        for(int i=0;i<ar-2;i++)  perm.push_back(i);
        perm.push_back(ar-1);
        perm.push_back(ar-2);
        return ( xt::transpose(a,perm) );
    }
    Tensor::shape_type _batch_shape( Tensor a )
    {
        auto as = a.shape();
        int  ar = as.size();
        int  b=1;
        for(int i=0;i<ar-2;i++)  b *= as[i];
        Tensor::shape_type new_s(3);
        new_s[0] = b;
        new_s[1] = as[ar-2];
        new_s[2] = as[ar-1];
        return new_s;
    }
    Tensor::shape_type _restore_shape( Tensor::shape_type as, Tensor::shape_type bs )
    {
        int  ar = as.size();
        int  br = bs.size();
        Tensor::shape_type new_s(ar);
        for(int i=0;i<ar-2;i++)  new_s[i] = as[i];
        new_s[ar-2] = bs[ar-2];
        new_s[ar-1] = bs[ar-1];
        return new_s;
    }
    bool forward() { return true; }
    bool backward(){ return true; }
    
    Tensor _batch_grad( Tensor &x, Tensor &y, Tensor::shape_type in_s )
    {
        auto x_t  = _batch_transpose( x ); 
        auto bx_s = _batch_shape( x_t );
        auto bx   =  x_t.reshape( bx_s );
        auto by_s = _batch_shape( y );
        auto by   =  y.reshape( by_s );
        //
        print_shape(" x_t  shape ", x_t );
        print_shape(" y    shape ", y );
        print_shape("b:x_t shape ", bx );
        print_shape("b:y   shape ", by );
        //
        Tensor::shape_type out_s = { bx_s[0], bx_s[1], by_s[2] };
        Tensor out(out_s);
        for(int i=0;i<bx_s[0];i++)
        {
            auto v = xt::view( bx, i, xt::all(), xt::all() );
            auto w = xt::view( by, i, xt::all(), xt::all() );
            auto temp = DOT(v,w); // ba.T * bb
            print_shape("temp shape ", temp);
            xt::view( out, i, xt::all(), xt::all() ) = temp;
        }
        
        Tensor::shape_type gb_s = _restore_shape( in_s, out_s ); 
        Tensor gb = out.reshape( gb_s );
        print_shape("gb shape ", gb);
        print_tensor( "gb", gb );
        
        return gb;
    }
};

class MatMulOp:public MatMulBase{
    public:
    MatMulOp(){}
    
    bool forward()
    {
        if( inputs.size() != 2 ){
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        _forward_inputs();
        print_message( "matmul(forward)" );
        auto& a  = inputs[0]->output;
        auto& b  = inputs[1]->output;
        auto  as = a.shape();
        auto  bs = b.shape();
        int   ar = as.size();
        int   br = bs.size();
        if( ( ar==1 || ar==2 ) && br==2 ){ 
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            output = DOT(a,b);
            //output =xt::linalg::dot(a,b);
        } else if( ar > 2 && br==2 ) {
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            output = DOT(a,b);
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size() != 2 ) {
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        print_message( "matmul(backward)" );
        auto& a  = inputs[0]->output;
        auto& b  = inputs[1]->output;
        auto& ga = inputs[0]->grad;
        auto& gb = inputs[1]->grad;
        auto& gc = this->grad;
        auto  as = a.shape();
        auto  bs = b.shape();
        int   ar = as.size();
        int   br = bs.size();
        if( ar==2 && br==2 ){
            // matmul:
            //  a.rank: 2
            //  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            gb += DOT( xt::transpose(a), gc );
            
        } else if( ar==1 && br==2 ){
            // matmul:
            //  a.rank: 1
            //  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            
        } else if( ar > 2 && br==2 ){
            // batched matmul:
            //  a.rank: >2
            //  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            print_tensor( "ga", ga );
            print_tensor( "gc", gc );
            
            gb += _batch_grad( a, gc, as );
            
            /* 210824 mod
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
            */
            /*
            vector<int> r;
            for(int i=0;i<ar-2;i++){
                r.push_back(i);
            }
            //cout<<temp<<endl;
            //gb+=xt::sum(temp,r);
            */
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
};

class LinearOp:public MatMulBase{
    public:
    LinearOp(){}
    
    bool forward()
    {
        if( inputs.size() != 3 )
        {
            cout<<"Error:LinearOp input size"<<endl;
            return false;
        }
        _forward_inputs();  
        print_message( "linear(forward)" );
        auto& a  = inputs[0]->output;  // x
        auto& b  = inputs[1]->output;  // weight
        auto& d  = inputs[2]->output;  // bias
        auto  as = a.shape();
        auto  bs = b.shape();
        auto  ds = d.shape();
        int   ar = as.size();
        int   br = bs.size();
        /*
        print_tensor( "linear a", a );
        print_tensor( "linear b", a );
        print_tensor( "linear d", a );
        cout<<"ar:"<<ar<<endl;
        cout<<"br:"<<br<<endl;
        print_shape( "a", a );
        print_shape( "b", b );
        print_shape( "d", d );*/
        
        if(( ar==1 || ar==2 ) && br==2 ) { 
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            output = DOT(a, xt::transpose(b) ) + d;
            
        } else if( ar > 2 && br==2 ){  // yet 
            // batched admm:
            //  a.rank: >2
            //  b.rank: 2 
            output = DOT( a,xt::transpose(b) ) + d;
            
        } else {
            cout<<"Error:LinearOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size() != 3 ){
            cout<<"Error:LinearOp input size"<<endl;
            return false;
        }
        print_message( "linear(backward)" );
        auto& a  = inputs[0]->output; // x
        auto& b  = inputs[1]->output; // weight
        auto& d  = inputs[2]->output; // bias
        auto& ga = inputs[0]->grad;
        auto& gb = inputs[1]->grad;
        auto& gd = inputs[2]->grad;
        auto& gc = this->grad;
        auto  as = a.shape();
        auto  bs = b.shape();
        int   ar = as.size();
        int   br = bs.size();
        
        if( ar==2 && br==2 ){
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            ga += DOT( gc, b );
            gb += DOT( xt::transpose(gc), a );
            gd += xt::sum( gc,{0} );
            
        } else if( ar==1 && br==2 ){ // yet
            // addmm:
            //  a.rank: 1
            //  b.rank: 2
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} );
            
        } else if( ar > 2 && br==2 ){
            // batched addmm:
            //  a.rank: >2
            //  b.rank: 2 
            ga += DOT( gd, xt::transpose(b) );
            gd += xt::sum( gc,{0} );
            
            print_tensor( "linear ga", ga );
            print_tensor( "linear gc", gc );
            cout<<"=="<<endl;
            
            gb += _batch_grad( gc, a, as );
            
        } else {
            cout<<"Error:LinearOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
    
    void update( fprec delta )
    {
        auto& b  = inputs[1]->output;  // weight
        auto& d  = inputs[2]->output;  // bias
        auto& gb = inputs[1]->grad;
        auto& gd = inputs[2]->grad;
        
        b = b - delta * gb;
        d = d - delta * gd;
    }
};

class AddMmOp:public MatMulBase{
    public:
    AddMmOp(){}
    
    bool forward()
    {
        if( inputs.size() != 5 ){
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
         _forward_inputs();  
        print_message( "addmm(forward)" );
        auto& d  = inputs[0]->output;  // bias
        auto& a  = inputs[1]->output;  // x
        auto& b  = inputs[2]->output;  // weight
        auto  as = a.shape();
        auto  bs = b.shape();
        auto  ds = d.shape();
        int   ar = as.size();
        int   br = bs.size();
    
        if(( ar==1 || ar==2 ) && br==2 ){ 
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            output = DOT(a,b) + d;
            
        } else if( ar > 2 && br==2 ){
            // batched admm:
            //  a.rank: >2
            //  b.rank: 2 
            output = DOT(a,b) + d;
            
        } else {
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size()!=5 ){
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
        print_message( "addmm(backward)" );
        auto& d  = inputs[0]->output; // bias
        auto& a  = inputs[1]->output; // x
        auto& b  = inputs[2]->output; // weight
        auto& gd = inputs[0]->grad;
        auto& ga = inputs[1]->grad;
        auto& gb = inputs[2]->grad;
        auto& gc = this->grad;
        auto  as = a.shape();
        auto  bs = b.shape();
        int   ar = as.size();
        int   br = bs.size();
       
        if( ar==2 && br==2 ) {
            // addmm:
            //  a.rank: 2
            //  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            gb += DOT( xt::transpose(a), gc );
            gd += xt::sum( gc,{0} );
            
        } else if( ar==1 && br==2 ) {
            // addmm:
            //  a.rank: 1
            //  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} );
            
        } else if( ar > 2 && br==2 ) {
            // batched addmm:
            //  a.rank: >2
            //  b.rank: 2 
            ga += DOT( gd, xt::transpose(b) );
            gd += xt::sum( gc,{0} );
            gb += _batch_grad( a, gc, as );
            
        } else {
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
    
    bool forward()
    {
        _forward_inputs();
        print_message( "transpose(forward)" );
        output = xt::transpose( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "transpose(backward)" );
        inputs[0]->grad += xt::transpose( this->grad );
        _backward_inputs();
        return true;
    }
};

class MaxOp:public MCTNode{
    public:
    MaxOp( int ax=1 ) { axis = ax; }
    int axis;
    xt::xarray<bool> cond;
    
    bool forward(){
        _forward_inputs();
        print_message( "max(forward)" );
        axis = (int)inputs[1]->output[0];
        output = xt::amax( inputs[0]->output, {axis} );
        return true;
    }
    bool backward(){
        print_message( "max(backward)" );
        auto& x  = inputs[0]->output;
        auto  o  = output;
        auto& gd = inputs[0]->grad;
        auto  xs = x.shape();
        
        auto& gc = this->grad;
        print_tensor( "gd1", gc );
        if( axis == 1 ){
            o.reshape( {-1,1} );
            gd.reshape( {-1,1} );
        }
        cond = xt::equal( x, o );
        cout<<"cond"<<cond<<endl;
        gd = gc * cond;
        print_tensor( "gd2", gd );
        _backward_inputs();
        return true;
    }
};

class MinOp:public MCTNode{
    public:
    MinOp( int ax=1 ){ axis = ax; }
    int axis;
    xt::xarray<bool> cond;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "min(forward)" );
        axis = (int)inputs[1]->output[0];
        output = xt::amin( inputs[0]->output, {axis} );
        return true;
    }
    bool backward() // same as MaxOp::backward()
    {
        print_message( "min(backward)" );
        auto& x  = inputs[0]->output;
        auto  o  = output;
        auto& gd = inputs[0]->grad;
        auto  xs = x.shape();
        
        auto& gc = this->grad;
        //print_tensor( "gd1", gc );
        if( axis == 1 ){
            o.reshape( {-1,1} );
            gd.reshape( {-1,1} );
        }
        cond = xt::equal( x, o );
        //cout<<"cond"<<cond<<endl;
        gd = gc * cond;
        print_tensor( "gd2", gd );
        _backward_inputs();
        return true;
    }
};

class SigmoidOp:public MCTNode{
    public:
    SigmoidOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "sigmoid(forward)" );
        output = 1.0 / ( 1.0+xt::exp( -inputs[0]->output ) );
        return true;
    }
    bool backward()
    {
        print_message( "sigmoid(backward)" );
        inputs[0]->grad += this->grad * output * ( 1.0 - output );
        _backward_inputs();
        return true;
    }
};

class ReluOp:public MCTNode{
    public:
    ReluOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "relu(forward)" );
        output = xt::maximum( inputs[0]->output, 0 );
        return true;
    }
    bool backward()
    {
        print_message( "relu(backward)" );
        inputs[0]->grad += this->grad * ( inputs[0]->output > 0 );
        _backward_inputs();
        return true;
    }
};

class HardTanhOp:public MCTNode{
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
        auto& y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd += this->grad * ( y > min_val && y < max_val );
        _backward_inputs();
        return true;
    }
};

class EluOp:public MCTNode{
    public:
    EluOp(fprec a=1.0) { alpha = a; }
    fprec alpha;
    xt::xarray<bool> mask;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "elu(forward)" );
        auto& y = inputs[0]->output;
        alpha   = (fprec)inputs[1]->output[0];
        output  = xt::maximum( y, 0.0 );
        mask = ( y < 0.0 );
        auto m = xt::masked_view( output, mask );
        m = xt::minimum( alpha * ( xt::exp( y ) - 1.0 ), 0.0 );
        return true;
    }
    bool backward()
    {
        print_message( "elu(backward)" );
        auto& y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd = this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = alpha * xt::exp( y );
        _backward_inputs();
        return true;
    }
};

class LeakyReluOp:public MCTNode{
    public:
    LeakyReluOp( fprec s=0.01 ) { slope=s; }
    fprec slope;
    xt::xarray<bool> mask;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "leakyrelu(forward)" );
        auto& y = inputs[0]->output;
        slope   = (fprec)inputs[1]->output[0];
        mask = ( y < 0.0 );
        output = y;
        auto m = xt::masked_view( output, mask );
        m = y * slope;
        return true;
    }
    bool backward()
    {
        print_message( "leakyrelu(backward)" );
        auto& y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd += this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = gd * slope;
        _backward_inputs();
        return true;
    }
};

class SoftplusOp:public MCTNode{
    public:
    SoftplusOp(){}
    xt::xarray<bool> mask;
    
    bool forward()
    {
        fprec beta = (fprec)inputs[1]->output[0];
        if( beta == 0.0 ){
            cout<<"Error:Beta is 0.0"<<endl;
            return false;
        }
        _forward_inputs();
        print_message( "softplus(forward)" );
        fprec threshold = (fprec)inputs[2]->output[0];
        output = inputs[0]->output;
        mask = ( output < threshold );
        auto m = xt::masked_view( output, mask );
        m = xt::log( 1.0 + xt::exp( beta * output ) ) / beta;
        return true;
    }
    bool backward()
    {
        print_message( "softplus(backward)" );
        auto& a  = inputs[0]->output;
        auto& ga = this->grad;
        fprec beta      = (fprec)inputs[1]->output[0];
        fprec threshold = (fprec)inputs[2]->output[0];
        ga = 1.0;
        auto  y = inputs[0]->output;
        mask = ( y < threshold );
        auto m = xt::masked_view( ga, mask );
        m = 1.0/( 1.0 + 1.0/xt::exp( beta * a ) );
        _backward_inputs();
        return true;
    }
};

class SoftmaxBase:public MCTNode{
    public:
    SoftmaxBase(){}
    
    bool forward()  { return true; }
    bool backward() { return true; }
    
    Tensor _row2col( Tensor& b, Tensor::shape_type as  )
    {
        auto bs = b.shape();
        int  az = as.size();
        int  bz = bs.size();
        if( az == bz )  return b;
        
        if( az == 2 && bz == 1 ) {
            return xt::view( b, xt::all(), xt::newaxis() );
            //return xt::expand_dims( b, 1 );
            //return b.reshape({-1,1} );
        }
        return b; // error
    }
    virtual Tensor _softmax( Tensor& a, int ax )
    {
        //cout<<"_softmax"<<endl;
        auto  as = a.shape();
        auto  ga = this->grad;
       
        Tensor sm = xt::amax( a, {ax} );
        sm = _row2col( sm, as );
        auto   sa = a - sm;
        auto   se = xt::exp( sa );
        Tensor sd = xt::sum( se, {ax} );
        sd = _row2col( sd, as );
        Tensor y = se / sd;  // softmax(a)
        return y;
    }
    virtual Tensor _log_softmax( Tensor& a, int ax )
    {
        //cout<<"_log_softmax"<<endl;
        auto  as = a.shape();
        int   ar = as.size();
        Tensor sm = xt::amax( a, {ax} );
        sm = _row2col( sm, as );
        auto sa = a - sm;
        auto se = xt::exp( sa );
        auto sd = xt::sum( se, {ax} );
        Tensor sl = xt::log( sd );
        sl = _row2col( sl, as );
        return ( a - (sm + sl) );
    }
};

class SoftmaxOp:public SoftmaxBase{
    public:
    SoftmaxOp(int ax=1) { axis = ax; }
    int axis;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "softmax(forward)" );
        output = _softmax( inputs[0]->output, axis ); 
        return true;
    }
    bool backward()
    {
        print_message( "softmax(backward)" );
        auto   as = output.shape();
        int    ar = as.size();
        Tensor ga = output * this->grad;
        Tensor sg = xt::sum( ga, {axis} );
        
        inputs[0]->grad += ( ga - output * sg );
        _backward_inputs();
        return true;
    }
};

class LogSoftmaxOp:public SoftmaxBase{
    public:
    LogSoftmaxOp( int ax=1 ) { axis=ax; }
    int axis;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "log_softmax(forward)" );
        output = _log_softmax( inputs[0]->output, axis );
        return true;
    }
    bool backward()
    {
        print_message( "log_softmax(backward)" );
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
    
    bool forward()
    {
        _forward_inputs();
        print_message( "tanh(forward)" );
        output = xt::tanh( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "tanh(backward)" );
        inputs[0]->grad += this->grad * ( 1.0 - output * output );
        _backward_inputs();
        return true;
    }
};

class FullLikeOp:public MCTNode{
    public:
    FullLikeOp( fprec v=0.0 ) { value = v; }
    fprec value;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "full_like(forward)" );
        if( inputs[0] )
        {
            auto& a  = inputs[0]->output;
            auto& as = a.shape();
            if( as[0] > 0 && as[1] > 0 )
            {
                output = xt::full_like( a, value );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        //print_message( "fulllike(backward)" );
        _backward_inputs();
        return true;
    }
};

class ZerosOp:public MCTNode{
    public:
    ZerosOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "zeros(forward)" );
        if( inputs[0] )
        {
            Tensor::shape_type shape;
            std::vector<size_t> v;
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                std::vector<size_t> v;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
                    cout<<"zeros shape "<<t1[0]<<endl;
                    v.push_back( (int)t1[0] );
                }
                shape = v;
                output = xt::zeros<fprec>( shape );
                cout<<"zeros"<<output<<endl;
                return true;
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                for(int k=0;k<as.size();k++)
                {
                    v.push_back( as[k]);
                }
                shape = v;
                output = xt::zeros<fprec>( shape );
                return true;
            }
            /*
            int s0 = 0;
            int s1 = 0;
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                auto& t0 = tlist->at(0);
                auto& t1 = tlist->at(1);
                s0 = (int)t0[0];
                s1 = (int)t1[0];
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                s0 = as[0];
                s1 = as[1];
            }
            if( s0 > 0 && s1 > 0 )
            {
                output = xt::zeros<fprec>( {s0,s1} );
                return true;
            }*/
        }
        return false;
    }
    bool backward()
    {
        //print_message( "zeros(backward)" );
        _backward_inputs();
        return true;
    }
};

class OnesOp:public MCTNode{
    public:
    OnesOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "ones(forward)" );
        if( inputs[0] )
        {
            Tensor::shape_type shape;
            std::vector<size_t> v;
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                std::vector<size_t> v;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
                    v.push_back( (int)t1[0] );
                }
                shape = v;
                output = xt::ones<fprec>( shape );
                return true;
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                for(int k=0;k<as.size();k++)
                {
                    v.push_back( as[k] );
                }
                shape = v;
                output = xt::ones<fprec>( shape );
                return true;
            }
            /*
            int s0 = 0;
            int s1 = 0;
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                auto& t0 = tlist->at(0);
                auto& t1 = tlist->at(1);
                s0 = (int)t0[0];
                s1 = (int)t1[0];
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                s0 = as[0];
                s1 = as[1];
            }
            if( s0 > 0 && s1 > 0 )
            {
                output = xt::ones<fprec>( {s0,s1} );
                return true;
            }*/
        }
        return false;
    }
    bool backward(){
        //print_message( "ones(backward)" );
        _backward_inputs();
        return true;
    }
};

class RandnOp:public MCTNode{
    public:
    RandnOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "randn(forward)" );
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                auto& t0 = tlist->at(0);
                auto& t1 = tlist->at(1);
                int s0 = (int)t0[0];
                int s1 = (int)t1[0];
                output = xt::random::randn<fprec>( {s0,s1} );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        //print_message( "randn(backward)" );
        _backward_inputs();
        return true;
    }
};

class NormalOp:public MCTNode{
    public:
    NormalOp(){}
  
    bool forward()
    {
        _forward_inputs();
        if( !inputs[0] )  return false;
        if( !inputs[1] )  return false;
        print_message( "normal(forward)" );
        if( inputs[0]->is_grad() ) 
        {
            if( inputs[1]->is_grad() )  // inputs[0,1] are tensors
            {
                auto& mean = inputs[0]->output;
                auto& std  = inputs[1]->output;
                auto  shape = mean.shape();
                Tensor out( shape );
                for(int i=0;i<shape[0];i++)
                {
                    auto o = xt::random::randn<fprec>( {1,1}, (fprec)mean[i], (fprec)std[i] );
                    out(i,0) = o(0,0);
                    out(i,1) = o(0,1);
                }
                output = out;
                return true;
            }
        } else {
            if( !inputs[1]->is_grad() )  // inputs[0,1] are scalar
            {
                if( inputs[2] )  // inputs[2] is tensor shape
                {
                    std::vector<Tensor>* tlist = inputs[2]->get_tlist();
                    if( tlist )
                    {
                        fprec mean = (fprec)inputs[0]->output[0];
                        fprec std  = (fprec)inputs[1]->output[0];
                        auto& t0 = tlist->at(0);
                        auto& t1 = tlist->at(1);
                        int s0 = (int)t0[0];
                        int s1 = (int)t1[0];
                        output = xt::random::randn<fprec>( {s0,s1}, mean, std );
                        //print_tensor( "randn", output );
                        return true;
                    }
                }
            }
        }
        return false;
    }
    bool backward()
    {
        _backward_inputs();
        return true;
    }
};

class BatchNormOp:public MCTNode{
public:
    BatchNormOp() {}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "batchnorm(forward)" );
        Tensor   x  = inputs[0]->output;
        auto     xs = x.shape();
        int   n_dim = xs.size();
        if( n_dim != 2 && n_dim != 4 )
        {
            cout<<"Error:BatchNormOp input dimension(not 2 or 4)"<<endl;
            cout<<"ndim="<<n_dim<<endl;
            return false;
        }
        
        UINT N,C,H,W;
        if( n_dim == 4 )
        {
            N = xs[0]; C = xs[1]; H = xs[2]; W = xs[3];
            vector<UINT> perm1{ 0, 2, 3, 1 };
            x = xt::transpose( x, perm1 );
            x.reshape( { -1, (int)C } );
        }
        
        auto& gamma = inputs[1]->output;  // gamma (weight)
        auto& beta  = inputs[2]->output;  // beta  (bias)
        auto& running_mean = inputs[3]->output;
        auto& running_var  = inputs[4]->output;
        fprec momentum = (fprec)inputs[6]->output[0];
        fprec eps      = (fprec)inputs[7]->output[0]; 
        
        Tensor xn;
        if( train_mode )
        {
            Tensor mean = xt::mean( x, {0} );
            Tensor var  = xt::variance( x, {0} );
            xn = ( x - mean ) / xt::sqrt( var + eps );
            int   gz = gamma.size();
            auto  xs = x.shape();
            fprec m  = (fprec)xs[1] / (fprec)gz;
            fprec adjust = ( m > 2.0 ) ? (m-1.0): 1.0;  
            running_mean = running_mean * (1-momentum) + momentum * mean;
            running_var  = running_var  * (1-momentum) + momentum * var * adjust;
        } else {
            xn = ( x - running_mean ) / ( xt::sqrt( running_var + eps ) );
        }
    
        if( n_dim == 4 ) {
            Tensor u = gamma * xn + beta;
            u = u.reshape( { N, H, W, C });
            vector<UINT> perm2{ 0, 3, 1, 2 };
            output = xt::transpose( u, perm2 );
        } else {
            output = gamma * xn + beta;
        }
        //cout<<"batchnorm"<<output<<endl;
        return true;
    }
    bool backward()
    {
        print_message( "batchnorm(backward)" );
        Tensor   gc = this->grad;
        auto     gs = gc.shape();
        int n_dim   = gs.size();
        int n_batch = gs[0];
        
        UINT N,C,H,W;
        vector<UINT> perm1{ 0, 2, 3, 1 };
        if( n_dim == 4 )
        {
            N = gs[0]; C = gs[1]; H = gs[2]; W = gs[3];
            gc = xt::transpose( gc, perm1 );
            gc.reshape( { -1, (int)C } );
        }
        Tensor x = inputs[0]->output;
        if( n_dim == 4 )
        {
            x = xt::transpose( x, perm1 );
            x.reshape( { -1, (int)C } );
        }
        
        auto& gamma = inputs[1]->output;  // gamma (weight)
        auto& beta  = inputs[2]->output;  // beat  (bias)
        auto& running_mean = inputs[3]->output;
        auto& running_var  = inputs[4]->output;
        fprec eps   = (fprec)inputs[7]->output[0]; 
        
        Tensor mean = xt::mean( x, {0} );
        Tensor var  = xt::variance( x, {0} );
        Tensor xc,xn,std;
        if( train_mode ) {
            xc  = x - mean;
            std = xt::sqrt( var + eps );
            xn  = xc / std;
        } else {
            xc  = x - running_mean ;
            std = xt::sqrt( running_var + eps );
            xn  = xc / std;
        }
            
        auto& gx = inputs[0]->grad; // x
        auto& gm = inputs[1]->grad; // gamma
        auto& gb = inputs[2]->grad; // beta
            
        Tensor dxn  = gamma * gc;
        Tensor dxc  = dxn / std;
        Tensor dvar = xt::sum( ( dxn*xc )/(std*std), {0} );
        dvar = dvar / std;
        dxc  = dxc - xc * dvar / n_batch;
        Tensor dmu = xt::sum( dxc, {0} );
        gx = dxc - dmu / n_batch;
        gm = xt::sum( gc*xn , {0} );
        gb = xt::sum( gc, {0} );
           
        if( n_dim == 4 )
        {
            gx = gx.reshape( { N, H, W, C });
            vector<UINT> perm2{ 0, 3, 1, 2 };
            gx = xt::transpose( gx, perm2 );
        }
    
        _backward_inputs();
        return true;
    }
    void update( fprec delta )
    {
        auto& g  = inputs[1]->output;  // gamma
        auto& b  = inputs[2]->output;  // beta
        auto& gm = inputs[1]->grad;
        auto& gb = inputs[2]->grad;
        
        g = g - delta * gm;
        b = b - delta * gb;
    }
};

class DropoutOp:public MCTNode{
public:
    DropoutOp( int kd=1 ) { kind = kd; }
    Tensor dropout;
    int kind;  // dropout-type ( 0: normal 1: inverted[pytorch] )
    
    bool forward()
    {
        _forward_inputs();
        print_message( "dropout(forward)" );
        auto& x = inputs[0]->output;
        fprec ratio = (fprec)inputs[1]->output[0];
        
        if( train_mode ) {
            Tensor r = xt::random::rand<fprec>( x.shape() );
            if( kind == 1 ) { // inverted
                fprec scale = 1.0 / (1.0 - ratio);
                dropout = xt::where( r > ratio, 1, 0 );
                output  = x * dropout * scale;
            } else {
                dropout = xt::where( r > ratio, 1, 0 );
                output  = x * dropout;
            }
            print_tensor("drop2",output);
            print_tensor("dropout",dropout);
        } else {
            if( kind == 1 ) { // inverted
                output = x;
            } else {
                output = x * ( 1.0 - ratio );
            }
        }
        return true;
    }
    bool backward()
    {
        if( train_mode ) {
            print_message( "dropout(forward)" );
            if( kind == 1 ) { // inverted
                fprec ratio = (fprec)inputs[1]->output[0];
                fprec scale = 1.0 / (1.0 - ratio);
                inputs[0]->grad = this->grad * dropout * scale;
            } else {
                inputs[0]->grad = this->grad * dropout;
            }
        } else {
            inputs[0]->grad = this->grad;
        }
        _backward_inputs();
        return true;
    }
};

class MseLossOp:public MCTNode{
public:
    MseLossOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "mseloss(forward)" );
        auto& a = inputs[0]->output;
        auto& b = inputs[1]->output;
        int  type = (int)inputs[2]->output[0];
        auto diff = a - b;
        output = xt::sum( xt::pow( diff, 2.0 ) );
        if( type == 1 )  output = output/(fprec)a.size();
        //cout<<"mseloss"<<output<<endl;
        return true;
    }
    bool backward()
    {
        print_message( "mseloss(backward)" );
        auto& a = inputs[0]->output;
        auto& b = inputs[1]->output;
        int  type = (int)inputs[2]->output[0];
        auto diff = a - b;
        auto& ga = inputs[0]->grad;
        auto& gb = inputs[1]->grad;
        auto& gc = this->grad;
        ga = gc * diff * 2.0;
        if( type == 1 )  ga = ga / (fprec)a.size();
        gb = -ga;
        //print_tensor( "mseloss_grad", ga );
        _backward_inputs();
        return true;
    }
    fprec get_loss() { return output[0]; }
};

class CrossEntropyLossOp:public SoftmaxBase{ // ( == log_softmax )
public:
    CrossEntropyLossOp( int ax=1 ) { axis = ax; }
    int axis;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "cross_entropy_loss(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        print_shape("ashape", a);
        
        auto sz = _log_softmax( a, axis );
        
        fprec  h = ( inputs[4] ) ? (fprec)inputs[4]->output[0] : -100.0;
        
        xt::xarray<float> t = xt::zeros<float>( {as[0]} );
        for(int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            t[i] = sz( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / (fprec)as[0];
        print_tensor( "crossloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "cross_entropy_loss(backward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        auto& ga = this->grad;
        fprec sc = 1.0 / (fprec)as[0];
        
        auto y = _softmax( a, axis );
        
        Tensor one = xt::zeros<float>( as );
        for(int i=0;i<as[0];i++){
            int j = (int)inputs[1]->output[i];
            one(i,j) = 1.0;
        }
        
        inputs[0]->grad = ( y - one ) * ga * sc;
        //print_tensor( "crossloss_grad", inputs[0]->grad );
        _backward_inputs();
      
        return true;
    }
    Tensor get_classes()
    {
        auto sm = _softmax( inputs[0]->output, axis );
        auto sh = sm.shape();
        xt::xarray<fprec>::shape_type shape = {sh[0]};
        
        Tensor lbs( shape );
        for(int i=0;i<sh[0];i++)
        {
            fprec smax = sm(i,0);
            int jl = (int)lbs(i);
            int jm = 0;
            for(int j=1;j<sh[1];j++)
            {
                if( smax < sm(i,j) ) {
                    smax = sm(i,j);
                    jm = j;
                }
            }
            lbs[i] = fprec(jm);
        }
        return lbs;
    }
    fprec get_loss() { return output[0]; }
};

class BCELossOp:public MCTNode {
public:
    BCELossOp(){};
    
    bool forward()
    {
        _forward_inputs();
        print_message( "bceloss(forward)" );
        auto& y = inputs[0]->output;
        auto& t = inputs[1]->output;
        
        fprec eps = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0e-7;
        fprec d = fprec( y.size() );
        if( inputs[3] ) 
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        Tensor q = t * xt::log(y+eps) + (1-t) * xt::log(1-y+eps);
      
        auto ys = y.shape();
        output = -xt::sum( q )/ d;
        print_tensor( "bceloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "bceloss(backward)" );
        auto& y = inputs[0]->output;
        auto& t = inputs[1]->output;
        
        fprec eps = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0e-7;
        fprec d = fprec( y.size() );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        auto  ys = y.shape();
        auto& gy = inputs[0]->grad;
        auto& gt = inputs[1]->grad;
        gy += this->grad * ( -t/(y+eps) + (1-t)/(1-y+eps) ) / d;
        gt += this->grad * ( -xt::log(y+eps) + xt::log(1-y+eps) ) / d; // 211004 mod
        //print_tensor( "bce grad", gy );
        _backward_inputs();
        return true;
    }
};

class NLLLossOp:public MCTNode{  // #220104 add
    public:
    NLLLossOp(){}
    
    bool forward() 
    {
        _forward_inputs();
        print_message( "nullload(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        fprec h  = (fprec)inputs[4]->output[0];
        
        xt::xarray<float> t = xt::zeros<float>( {as[0]} );
        for(int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            t[i] = a( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / (fprec)as[0];
        print_tensor( "nllloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "nullload(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        fprec sc = 1.0 / (fprec)as[0];
        
        Tensor& gc = this->grad;
        print_tensor( "gc ", gc );
        
        xt::xarray<fprec> one = xt::zeros<float>( as );
        for(int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            one(i,j) = -gc[0];
        }
        inputs[0]->grad = one * sc;
        print_tensor( "nllloss grad", inputs[0]->grad );
        _backward_inputs();
        return true;
    }
};

class SizeOp:public MCTNode{
public:
    SizeOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "size(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        int   ar = as.size();
        int   no = (int)inputs[1]->output[0];
        output = (float)as[no];
        return true;
    }
    bool backward()
    {
        _backward_inputs();
        return true;
    }
};

class ExpandOp:public MCTNode{
public:
    ExpandOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "expand(forward)" );
        auto& a = inputs[0]->output;
        
        if( inputs[1] )  // shape
        {
            std::vector<Tensor>* tlist = inputs[1]->get_tlist();
            if( tlist )
            {
                auto& t0 = tlist->at(0);
                auto& t1 = tlist->at(1);
                int s0 = (int)t0[0];
                int s1 = (int)t1[0];
                output = xt::broadcast( a, {s0,s1} );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        _backward_inputs();
        return true;
    }
};

class NumToTensorOp:public MCTNode{
public:
    NumToTensorOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "numtotensor(forward)" );
        output = inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "numtotensor(backward)" );
        inputs[0]->grad += this->grad;
        _backward_inputs();
        return true;
    }
};

class IntOp:public MCTNode{
public:
    IntOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "int(forward)" );
        output = inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "int(backward)" );
        inputs[0]->grad += this->grad;
        _backward_inputs();
        return true;
    }
};

class ViewOp:public MCTNode{
public:
    ViewOp(){}
    Tensor::shape_type  org_shape;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "view(forward)" );
        auto& a = inputs[0]->output;
        org_shape = a.shape();
        
        if( inputs[1] )
        {
            std::vector<Tensor>* tlist1 = inputs[1]->get_tlist();
            if( tlist1 )
            {
                std:vector<int> sv;
                for(int i=0;i<tlist1->size();i++)
                {
                    auto& t = tlist1->at(i);
                    sv.push_back( (int)t[0] );
                }
                output = a.reshape( sv );
            }
            return true;
        }
        return false;
    }
    bool backward()
    {
        print_message( "view(backward)" );
        Tensor& gc = this->grad;
        inputs[0]->grad = gc.reshape( org_shape );
        _backward_inputs();
        return true;
    }
};

class BroadcastTensorsOp:public MCTNode{
public:
    BroadcastTensorsOp() 
    {
        broadcast_shape = {0}; 
        result = 0;
    }
    std::vector<Tensor>  tlist;  // tensor list
    std::vector<Tensor>  glist;  // grad tensor list
    
    std::vector<Tensor>* get_tlist() { return &tlist; };
    std::vector<Tensor>* get_glist() { return &glist; };
    Tensor::shape_type   broadcast_shape;
    int  result;
    
    void check_shape( Tensor::shape_type as, vector<unsigned int>&na, int n_dim )
    {
        unsigned int  az = as.size();
        for(int i=0;i<n_dim;i++)  na[i] = 1;
        if( az > 0 ) 
        {
            int inc = n_dim - az;
            for(int i=0;i<az;i++)  na[inc+i] = as[i];
        }
    }
    void print_ints( string s, vector<unsigned int>v, int nv )
    {
        cout<<s;
        for(int i=0;i<nv;i++)  cout<<v[i]<<",";
        cout<<endl;
    }
    
    // return value   0: no broadcast
    //               >0; broadcast by shape
    //               <0: broadcast error
    int check( std::vector<Tensor> &a, int num, Tensor::shape_type &shape, int chk=0 )
    {
        shape = {0};
        if( num <  1 )  return -1;
        if( num == 1 )
        {
            shape = a[0].shape();
            return -3;  // Tensor only one
        }
        
        std::vector<Tensor::shape_type>  as(num);
        std::vector<int>  az(num);
        int n_dim = -1;
        for(int i=0;i<num;i++)  
        {
            as[i] = a[i].shape();
            az[i] = as[i].size();
            if( n_dim < az[i] )  n_dim = az[i];
        }
        if( n_dim < 1 )  return 0; // all numbers
        
        int equal = 1;
        int err   = 0;
        std::vector<unsigned int>  na(n_dim);
        std::vector<unsigned int>  nb(n_dim);
        check_shape( as[0], na, n_dim );
        if( chk > 0 )
        {
            cout<<"--------------------------"<<endl;
            cout<<"broadcast n_dim="<<n_dim<<endl;
            print_ints( "broadcast tensor-1 ", na, n_dim );
        }
        
        for(int k=1;k<num;k++)
        {
            check_shape( as[k], nb, n_dim );
            
            // check broadcast
            int ne = 0;
            int n1 = 0;
            int n2 = 0;
            for(int i=0;i<n_dim;i++){
                if( na[i] == nb[i] ){
                    ne += 1;
                } else if( na[i] == 1 ){
                    n1 += 1;
                } else if( nb[i] == 1 ){
                    n1 += 1;
                } else {
                    //cout<<"broadcast mismatch dimension no."<<i<<endl;
                    n2 += 1;
                }
            }
            if( chk > 0 )
            {
                string ss = "broadcast tensor-" + to_string(k+1) + " ";
                print_ints( ss, nb, n_dim );
                cout<<"broadcast ("<<k<<")  equal="<<ne<<" eq1="<<n1<<" other="<<n2<<endl;
            }
            
            int status = 0;
            if( ne == n_dim ) {  
                status = 2;
                equal++;
            } else if( (ne+n1) == n_dim ) {
                status = 1;
            } else {
                cout<<"broadcast error. ("<<k<<")"<<endl; 
                err += 1;
            }
            if( status == 1 )
            {
                for(int i=0;i<n_dim;i++)
                {
                    na[i] = ( na[i] > nb[i] ) ? na[i] : nb[i];
                }
            }
        }
        
        int result = 1;
        if( err == 0 )
        {
            if( equal == num ) {
                result = 0;
            } else {
                // set shape size by std::vector
                //   ex. shape = { na[0], na[1], na[2] } 
                //   vector<size_t> v= { na[0], na[1], na[2] }
                std::vector<size_t> v;
                for(int i=0;i<n_dim;i++)  v.push_back( na[i] );
                shape = v;
                result = 1;
                print_ints( "broadcast shape ", na, n_dim );
            }
        } else {
            for(int i=0;i<num;i++)
            {
                string ss = "tensor(" + to_string(i) + ")  shape=";
                print_shape( ss.c_str(), a[i] );
            }
            result = -1;
        }
        
        return result;
    }
    int restore_broadcast( Tensor &ga, Tensor::shape_type as, Tensor::shape_type os )
    {
        int  az = as.size();
        int  oz = os.size();
        cout<<"az "<<az<<", oz"<<oz<<endl;
        if( oz == 0 )
        {
            ga = (fprec)ga[0];
            return 1;
        }
        if( az == oz )
        {
            int equal = 0;
            for(int i=0;i<oz;i++)
            {
                if( as[i] == os[i] )  equal++;
            }
            cout<<"equal"<<equal<<endl;
            if( oz == equal )  return 0;  // no change
        } 
        
        int inc = az - oz;
        cout<<"inc"<<inc<<endl;
        // return original tensor row size 
        //  ex. delete first row
        //      ga = xt::view( ga, 0, xt::ellipsis() );
        //  cf. numpy : ga = ga[ 0, ... ]
        if( inc > 0 ) {
            xt::xstrided_slice_vector sv1;
            for(int i=0;i<inc;i++)  sv1.push_back( 0 );  // 0 == decrease tensor row
            sv1.push_back( xt::ellipsis() );  // xt:all() == add all column
            ga = xt::strided_view( ga, sv1 );
        } else if( inc < 0 ) {
            return -2;
        }
        // return original tensor column size
        //  ex. original tensor is sencond tensor. os is original column size array.
        //      ga = xt::view(ga, xt::range(0,os[0]), xt::range(0,os[1]) );
        //  cf. numpy : ga = ga[ 0:os[0], 0:os[1] ]
        if( oz > 0 )
        {
            xt::xstrided_slice_vector sv2;
            for(int i=0;i<oz;i++)  sv2.push_back( xt::range( 0, os[i] ) );
            ga = xt::strided_view( ga, sv2 );
        }
        return 1;
    }
    
    bool forward()
    {
        _forward_inputs();
        print_message( "broadcast_tensors(forward)" );
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            if( tlist1 )
            {
                result = check( *tlist1, tlist1->size(), broadcast_shape, 0 );
                if( result != 0 )  cout<<"broadcast check="<<result<<endl;
                if( result < 0 )   return false;
                
                tlist.clear();
                glist.clear();
                for(int i=0;i<tlist1->size();i++)
                {
                    if( result == 0 ) {
                        tlist.push_back( tlist1->at(i) );
                    } else { // result > 0  
                        auto &a1 = tlist1->at(i);
                        auto  a2 = xt::broadcast( a1, broadcast_shape );
                        tlist.push_back( a2 );
                    }
                }
            }
        }
/*#ifdef _DEBUG
        cout<<"BroadcastTensors"<<endl;
        for(int i=0;i<tlist.size();i++) {
            cout<<"Broadcast "<<","<<i<<" "<<tlist[i]<<endl;
        }
#endif*/
        return true;
    }
    bool backward()
    {
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            std::vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(int i=0;i<glist.size();i++)
                {
                    if( result == 0 ) {
                        glist1->push_back( glist.at(i) );
                    } else { // result > 0 
                        auto& a1 = tlist1->at(i);
                        auto& g1 = glist.at(i);
                        Tensor g2 = g1;
                        int status = restore_broadcast( g2, g1.shape(), a1.shape() );
                        if( status < 0 )  return false;
                        glist1->push_back( g2 );
                    }
                }
            }
            glist.clear();
        }
        print_message( "broadcast_tensors(backward)" );
        _backward_inputs();
        return true;
    }
};

// create tensor list
class ListConstructOp:public MCTNode{
public:
    ListConstructOp() {}
    ListConstructOp( string na ) { name = na; }
    std::vector<Tensor>  tlist;  // tensor list
    std::vector<Tensor>  glist;  // grad tensor list
    
    std::vector<Tensor>* get_tlist() { return &tlist; };
    std::vector<Tensor>* get_glist() { return &glist; };
    
    bool forward()
    {
        _forward_inputs();
        print_message( "list_contruct(forward)" );
        if( inputs.size() < 1 ) // 220106 mod
        {
            cout<<"Error:ListConstructOp"<<endl;
            return false;
        }
        
        tlist.clear();
        glist.clear();
        for(int i=0;i<inputs.size();i++)
        {
            if( inputs[i] ) {
                tlist.push_back( inputs[i]->output );
            } else {
                Tensor v = (fprec)0.0;
                tlist.push_back( v );
            }
        }
/*#ifdef _DEBUG
        cout<<"ListConstruct "<<name<<endl;
        for(int i=0;i<tlist.size();i++) {
            //cout<<"ListConstruct "<<i<<endl;
            cout<<"ListConstruct "<<","<<i<<" "<<tlist[i]<<endl;
        }
#endif*/
        return true;
    }
    bool backward()
    {
        print_message( "list_contruct(backward)" );
        if( glist.size() > 0 )
        {
            for(int i=0;i<inputs.size();i++)
            {
                if( inputs[i] )
                {
                    if( inputs[i]->is_grad() )
                    {
                        inputs[i]->grad = glist[i];
                        //glist.erase( glist.begin() );
                        //cout<<"ListConstruct "<<","<<i<<" "<<inputs[i]->grad<<endl;
                    }
                }
            }
        }
        _backward_inputs();
        return true;
    }
};

class ListUnpackOp:public MCTNode{
public:
    ListUnpackOp( int id=0 ) { out_id = id; };
    ListUnpackOp( string na, int id=0 ) 
    { 
        name = na;
        out_id = id;
    }
    int  out_id;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "list_unpack(forward)" );
        
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            if( tlist1 )
            {
                output = tlist1->at( out_id );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        print_message( "list_unpack(backward)" );
        std::vector<Tensor>* glist1 = inputs[0]->get_glist();
        if( glist1 )
        {
            int sz = glist1->size();
            if( sz <= out_id )
            {
                for(int i=sz;i<out_id;i++)  // temporary add
                {
                    glist1->push_back( (Tensor)0 );
                }
                glist1->push_back( this->grad );
            } else {
                glist1->at(out_id)= this->grad;
            }
        }
        _backward_inputs();
        return true;
    }
};


class TupleConstructOp:public MCTNode {
public:
    TupleConstructOp() {}
    TupleConstructOp( string na ) { name = na; }
    std::vector<MCTNode*>  ndlist;  // MCTnode pointer list
  //std::vector<MCTNode*>  nglist;  // MCTNode* grad list
    
    std::vector<MCTNode*>* get_ndlist() { return &ndlist; };
  //std::vector<MCTNode*>* get_nglist() { return &nglist; };
    
    bool forward()
    {
        _forward_inputs();
        print_message( "tuplecontruct(forward)" );
        
        ndlist.clear();  
        //nglist.clear(); 
    
        for(int i=0;i<inputs.size();i++)
        {
            if( inputs[i] ) {
                ndlist.push_back( inputs[i] );
            } else {
                ndlist.push_back( NULL );
            }
        }
/*#ifdef _DEBUG
        cout<<"TupleConstruct "<<name<<endl;
        for(int i=0;i<ndlist.size();i++) {
            cout<<"TupleConstruct "<<i<<endl;
        }
#endif*/
        return true;
    }
    bool backward()
    {
        /*print_message( "tuplecontruct(backward)" );
        if( nglist.size() > 0 )
        {
            for(int i=0;i<inputs.size();i++)
            {
                if( inputs[i] )
                {
                    if( inputs[i]->is_grad() )
                    {
                        inputs[i]->grad = nglist[i]->output;
                        //glist.erase( glist.begin() );
                        //cout<<"ListConstruct "<<","<<i<<" "<<inputs[i]->grad<<endl;
                    }
                }
            }
        }*/
        _backward_inputs();
        return true;
    }
};

class TupleUnpackOp:public MCTNode {
public:
    TupleUnpackOp( int id=0 ) { out_id = id; }
    TupleUnpackOp( string na, int id=0  ) 
    { 
        name = na; 
        out_id = id;
    }
    int out_id;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "tupleunpack(forward)" );
        
        if( inputs[0] )
        {
            std::vector<MCTNode*>* ndlist1 = inputs[0]->get_ndlist();
            if( ndlist1 )
            {
                MCTNode *node = ndlist1->at( out_id );
                if( node )  output = node->output;
/*#ifdef _DEBUG
                cout<<"TupleUnpack "<<output<<endl;
#endif*/
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        print_message( "tupleunpack(backward)" );
        std::vector<MCTNode*>* ndlist1 = inputs[0]->get_ndlist();
        if( ndlist1 )
        {
            MCTNode* node = ndlist1->at( out_id );
            if( node )  node->grad += this->grad;
                
            /*int sz = nglist1->size();
            //cout<<"tuple unpack "<<sz<<","<<out_id<<endl;
            if( sz <= out_id )
            {
                for(int i=sz;i<out_id;i++)  // temporary add
                {
                    nglist1->push_back( NULL );
                }
                nglist1->push_back( this->grad );
            } else {
                nglist1->at(out_id) = this->grad;
            }*/
        }
        _backward_inputs();
        return true;
    }
};

class ToOp:public MCTNode{
public:
    ToOp() {}
    std::vector<Tensor>  tlist;  // tensor list
    std::vector<Tensor>* get_tlist() { return &tlist; };
    
    bool forward()
    {
        _forward_inputs();
        print_message( "to(forward)" );
        if( inputs.size() < 2 )
        {
            cout<<"Error:ToOp"<<endl;
            return false;
        }
        
        tlist.clear();
        for(int i=0;i<inputs.size();i++)
        {
            if( inputs[i] ) {
                tlist.push_back( inputs[i]->output );
            } else {
                Tensor v = (fprec)0.0;
                tlist.push_back( v );
            }
        }
        return true;
    }
    bool backward()
    {
        _backward_inputs();
        return true;
    }
};

class DetachOp:public MCTNode{
public:
    DetachOp( int id=0 ) { out_id = id; }
    int out_id;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "detach(forward)" );
        
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            if( tlist1 )
            {
                output = tlist1->at( out_id );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
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

