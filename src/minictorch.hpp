//
//   minictorch.hpp
//
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xbroadcast.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>



typedef  float  fprec;
typedef  xt::xarray<fprec>  Tensor;
typedef  xt::xarray<fprec>::shape_type  Tshape;
typedef  xt::xarray<bool>   Tbool;

extern bool train_mode;

using namespace std;

#define DOT(a,b)  (xt::linalg::dot(a,b))


enum Evariable  // VariableTensor type
{
    VAR_NODE     = 0,  // node
    VAR_CONST    = 1,  // constant
    VAR_ATTR     = 2,  // attribute
    VAR_INPUT    = 3,  // input_var
    VAR_RUNNING  = 4   // running_*(batchnorm)
};

enum Eshape  // Broadcast type
{
    SHAPE_ACCEPT     =  0,  // そのまま継続実行
    SHAPE_BROADCAST  =  1,  // Broadcastして実行 deprecated (1:A,2:B,3:A and B)

    SHAPE_REJECT     = -1,  // Broadcastできないの中断
    SHAPE_ERROR      = -2   // Errorなので中断
};


class MCTNode {
public:
    MCTNode()
    {
        id = -1;
        grad = 0;
        ntype = VAR_NODE;
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
    vector<MCTNode*>  inputs;
    string  name;
    Tensor  output;
    Tensor  grad;
    int    id;
    
    void set_inputs( MCTNode *node )
    {
        inputs.push_back( node );
    }
    
    virtual void update( fprec delta )   {};
    virtual void set_output1( fprec o1 ) {};
    
    virtual vector<Tensor>*   get_tlist() { return NULL; };
    virtual vector<Tensor>*   get_glist() { return NULL; };
    virtual vector<MCTNode*>* get_ndlist(){ return NULL; };
    
    // utility function
    void set_id( int n ) { id = n; };
    void zerograd()
    {
        this->grad = 0.;
    }
    Tensor& get_output() { return output; }
    void set_ntype( enum Evariable t ) { ntype=t; };
    
    // Whether to calculate the gradient
    // No further backpropagation for nodes with a gradient of 0
    bool is_grad()
    {
        if( ntype == VAR_CONST || ntype == VAR_RUNNING){
            return false;
        }
        return true;
    };
    
    // extend shape for the first step of broadcasting
    // as: (2,3,4,5)
    // n_dim: 7
    // out: (1,1,1,2,3,4,5)
    vector<size_t> extend_shape(const Tshape as, int n_dim )
    {
        unsigned int  az = as.size();
        vector<size_t> out(n_dim,1);
        if( az > 0 ) 
        {
            int inc = n_dim - az;
            for(unsigned int i=0;i<az;i++){
                out[inc+i] = as[i];
            }
        }
        return out;
    }
    
    // shape check utility functions
    Tshape out_shape;
    void set_shape( Tshape s ) { out_shape = s; };
    
    enum Eshape display_shape_size( string ss, int id, Tshape sh1, Tshape sh2, int size2 )
    {
        int sz1 = sh1.size();
        if( sz1 < 1 )  return SHAPE_ACCEPT;
        int sz2 = sh2.size();
        if( sz1 == 1 && sz2 == 0 ) 
        {
            cout<<"# "<<ss<<" shape "<<"id="<<id<<"  (1) - (scalar) ->0"<<endl;
            return SHAPE_BROADCAST;
        }
        if( sz1 != sz2 )
        {
            int nsz = 1;
            for(int i=0;i<sz1;i++)  nsz *= sh1[i];
            cout<<"# "<<ss<<" shape size  "<<"id="<<id<<" "<<nsz<<","<<size2<<","<<sz1<<","<<sz2<<endl;
            if( nsz != size2 )
            {
                cout<<"# "<<ss<<" shape dim error "<<"id="<<id<<"  "<<sz1<<"-"<<sz2<<endl;
                return SHAPE_REJECT;
            }
        }
        int err = 0;
        if( sz1 > 0 && sz2 > 0 ) 
        {
            for(int k=0;k<sz1;k++)
            {
                if( sh1[k] != sh2[k] )  err++;
            }
            if( err >= 0 ) // all output
            {
                cout<<"# "<<ss<<" shape "<<"id="<<id<<"  (";
                for(int i=0;i<sz1;i++)  cout<<sh1[i]<<",";
                cout<<")  -  (";
                for(int j=0;j<sz2;j++)  cout<<sh2[j]<<",";
                cout<<")"<<" -> "<<err<<endl;
                return SHAPE_REJECT;
            }
        }
        return SHAPE_BROADCAST;
    }
    enum Eshape display_grad_shape_size( string s, int id, Tensor& ga, Tensor& gb )
    {
        return display_shape_size( s, id, ga.shape(), gb.shape(), gb.size() );
    }
    enum Eshape display_shape1( string s )
    {
        return display_shape_size( s, id, out_shape, output.shape(), output.size() );
    }
    enum Eshape display_grad_shape1( string s, int k=0 )
    {
        if( inputs[k] )
        {
            return display_grad_shape_size( s, id, inputs[k]->output, inputs[k]->grad );
        }
        return SHAPE_ERROR;
    }
    virtual void display_shape()      {}
    virtual void display_grad_shape() {}
    
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
    void print_shape( string s, Tensor x )
    {
#ifdef _DEBUG
        Tshape xs = x.shape();
        cout<<s<<" (";
        for(int i=0;i<s.size();i++){
            cout<<xs[i]<<",";
        }
        cout<<")"<<endl;
#endif
    }
    void print_shape1( string s, Tshape &xs )
    {
        cout<<s<<" (";
        for(unsigned int i=0;i<xs.size();i++){
            cout<<xs[i]<<",";
        }
        cout<<")"<<endl;
    }
    void print_ints( string s, vector<unsigned int> &v, int nv )
    {
        cout<<s;
        for(int i=0;i<nv;i++)  cout<<v[i]<<",";
        cout<<endl;
    }
    void print_ints( string s, vector<size_t> &v, int nv )
    {
        cout<<s;
        for(int i=0;i<nv;i++)  cout<<v[i]<<",";
        cout<<endl;
    }

    void get_items( unsigned int *q, int n, int dum=0 )
    {
        vector<Tensor>* ptr = get_tlist(); 
        if( ptr )
        {
            for(int i=0;i<n;i++)
            {
                Tensor t = ptr->at(i);
                q[i] = (unsigned int)t[0];
            }
        } else {
            for(int i=0;i<n;i++)  q[i] = dum;
        }
    }
protected:
    enum Evariable ntype;
};

class VariableTensor : public MCTNode {
public:
    VariableTensor(){}
    
    VariableTensor( Tensor tensor, enum Evariable t=VAR_NODE ) 
    {
        this->output = tensor;
        this->ntype = t;
        this->grad = xt::zeros_like( output );
    }
    VariableTensor( string name, Tensor tensor, enum Evariable t=VAR_NODE )
    {
        this->output = tensor;
        this->name   = name;
        this->ntype  = t;
        this->grad   = xt::zeros_like( output );
    }
    bool forward()  { return true; }
    bool backward() { return true; }
    void set_output1( fprec o1 ) { output[0] = o1; }
    
    void update( fprec delta ) 
    {
        switch( ntype ) {
        case VAR_ATTR:  // attribute
            output = output - delta * grad;
            break;
        }
    }
};

class SumOp : public MCTNode {
public:
    SumOp( int ax=-1 ) { axis = ax; }
    int axis;  // sum kind  -1:all, 0:row, 1:column
    
    bool forward()
    {
        print_message( "sum(forward)" );
        if( inputs[1] ) { 
            axis = (int)inputs[1]->output[0];
            output = xt::sum( inputs[0]->output, {axis} );
        } else {
            if( axis < 0 ) {
                Tensor o = xt::sum( inputs[0]->output );
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
        Tensor a  = inputs[0]->output;
        Tshape as = a.shape();
        
        if( axis >=0  ) { 
            Tensor e = xt::expand_dims( this->grad, axis );
            Tensor g = xt::repeat( e, as[axis], axis );
            inputs[0]->grad = g;
        } else {
            inputs[0]->grad = xt::full_like( a, (fprec)this->grad[0] );
        }
        return true;
    }
};

class MeanOp : public MCTNode {
public:
    MeanOp( int ax=-1 ) { axis = ax; };
    int axis;  // mean kind  -1:all, 0:row, 1:column
    
    bool forward()
    {
        print_message( "mean(forward)" );
        
        if( inputs[0] )
        {
            if( inputs[1] )
            {
                vector<Tensor>* tlist = inputs[1]->get_tlist();
                if( tlist )
                {
                    if( tlist->size() > 1 ) 
                    {
                        cout<<"sum argumenst 1 error: list length > 1 "<<endl;
                        return false;
                    }
                    Tensor& t0 = tlist->at(0);
                    axis = (int)t0[0];
                } else {
                    axis = (int)inputs[1]->output[0];
                }
            } else {
                axis = -1;
            }
            if( axis < 0 ) {
                output = xt::mean( inputs[0]->output );
            } else {
                output = xt::mean( inputs[0]->output, {axis} );
            }
            return true;
        }
        return false;
    }
    bool backward()
    {
        print_message( "mean(backward)" );
         if( inputs[0] )
        {
            if( inputs[1] )
            {
                vector<Tensor>* tlist = inputs[1]->get_tlist();
                if( tlist )
                {
                    if( tlist->size() > 1 ) 
                    {
                        cout<<"sum argumenst 1 error: list length > 1 "<<endl;
                        return false;
                    }
                    Tensor& t0 = tlist->at(0);
                    axis = (int)t0[0];
                } else {
                    axis = (int)inputs[1]->output[0];
                }
            } else {
                axis = -1;
            }
            Tensor& a  = inputs[0]->output;
            Tshape  as = a.shape();
            if( axis >=0  ) {
                Tensor gd = this->grad / (fprec)as[axis];
                Tensor e  = xt::expand_dims( gd, axis );
                Tensor g  = xt::repeat( e, as[axis], axis );
                inputs[0]->grad = g;
            } else {
                fprec gd = this->grad[0] / a.size();
                inputs[0]->grad = xt::full_like( a, gd );
            }
        }
        return true;
    }
};

class StackOp : public MCTNode {
public:
    StackOp(){}
    
    bool forward()
    {
        print_message( "stack(forward)" );
        if( inputs[0] )
        {
            int dim = (int)inputs[1]->output[0];
            vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                int    tr = tlist->size();
                Tensor a0 = tlist->at(0);
                Tshape as = a0.shape();
                int    az = as.size();
                
                vector<size_t> sh;
                for(int i=0;i<dim;i++)   sh.push_back( as[i] ); 
                sh.push_back( tr );
                for(int i=dim;i<az;i++)  sh.push_back( as[i] );
                
                Tensor out( sh );
                for(int k=0;k<tr;k++) 
                {
                    xt::xstrided_slice_vector sv;
                    for(int i=0;i<az;i++)
                    {
                        if( i == dim )  sv.push_back( k );
                        else            sv.push_back( xt::all() );
                    }
                    xt::strided_view( out, sv ) = tlist->at(k);
                }
                output = out;
                return true;
            } else {
                output = inputs[0]->output;
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        print_message( "stack(backward)" );
        if( inputs[0] )
        {
            Tensor& gc = this->grad;
            Tshape  gs = gc.shape();
            int     gz = gs.size();
                
            int dim = (int)inputs[1]->output[0];
            vector<Tensor>* tlist = inputs[0]->get_tlist();
            vector<Tensor>* glist = inputs[0]->get_glist();
            if( tlist && glist )
            {
                int tr = tlist->size();
                for( int k=0;k<tr;k++)
                {
                    xt::xstrided_slice_vector sv;
                    for(int i=0;i<gz;i++)
                    {
                        if( i == dim )  sv.push_back( k );
                        else            sv.push_back( xt::all() );
                    }
                    Tensor g = xt::strided_view( gc, sv );
                    glist->push_back( g );
                }
            } else {
                inputs[0]->grad = gc;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "stack" );
    }
    void display_grad_shape()
    {
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(unsigned int i=0;i<tlist1->size();i++)
                {
                    Tensor& a1 = tlist1->at(i);
                    Tensor& g1 = glist1->at(i);
                    string  s  = "stack_" + std::to_string(i) + " grad";
                    display_grad_shape_size( s, id, a1, g1 );
                }
            }
        }
    }
};

class NegOp : public MCTNode {
public:
    NegOp(){}
    
    bool forward()
    {
        print_message( "neg(forward)" );
        output = -inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "neg(backward)" );
        inputs[0]->grad -= this->grad;
        return true;
    }
};

struct BroadcastResult
{
    enum Eshape status;
    unsigned int ope_type[2];
    Tshape       shape;
};

enum EBtype  // broadcast column check type
{
    EB_NORMAL = 0,  // all column check
    EB_MATMUL = 1   // matmul only ( last-2 column check )
};

class BroadcastBase : public MCTNode {
public:
    BroadcastBase() {}
    BroadcastResult  bc;
    
    bool forward()  { return true; };
    bool backward() { return true; };
    
    //broadcast
    //B_MATMUL:assign the value after matrix multiplication for the last two dimensions
    //         If it is regarded as a normal broadcast, an error will occur due to a dimension mismatch.
    BroadcastResult broadcast_check( Tensor& a, Tensor &b, enum EBtype eb_type=EB_NORMAL )
    {
        Tshape as = a.shape();
        Tshape bs = b.shape();
        int    az = as.size();
        int    bz = bs.size();
        
        Tshape shape = {1};
        
        // scalar check
        int ns = 0;
        if( az == 0 )  ns += 1;
        if( bz == 0 )  ns += 1;
        if( ns > 0 ) 
        {
#ifdef _DEBUG
            cout<<"scalar check = "<<ns<<endl;
            cout<<"broadcast check status="<<2<<endl;
#endif
            return { SHAPE_ACCEPT, 0, 0, shape };
        }
        
        if( eb_type == EB_MATMUL )
        {
            if( az < 3 && bz < 3 )
            {
#ifdef _DEBUG
                cout<<"broadcast check (matmul) status="<<0<<endl;
#endif
                if( az == 2 && bz == 2 )
                {
                    vector<size_t> sv;
                    sv.push_back( as[0] );
                    sv.push_back( bs[1] );
                    shape = sv;
                }
                return { SHAPE_ACCEPT, 0, 0, shape };
            }
        }
        
        int n_dim = std::max( az, bz );
        if( n_dim < 1 )  return { SHAPE_ACCEPT, 0, 0, shape };
        
        vector<size_t> na=extend_shape( as, n_dim );
        vector<size_t> nb=extend_shape( bs, n_dim );
        
#ifdef _DEBUG
        cout<<"n_dim "<<n_dim<<" <- "<<az<<","<<bz<<endl;
        print_ints( "broadcast tensor-a ", na, n_dim );
        print_ints( "broadcast tensor-b ", nb, n_dim );
#endif
        
        // num = last dimension for broadcast
        int num = ( eb_type == EB_MATMUL ) ? n_dim-2 : n_dim;
        
        //  matmul check
        if( eb_type == EB_MATMUL )
        {
            if( na[num+1] != nb[num] )
            {
                cout<<"error: matmul matrix mismatch "<<na[num+1]<<","<<nb[num]<<endl;
                return { SHAPE_REJECT, 0, 0, shape };
            }
        }
        
        vector<unsigned int> nc(n_dim);
        
        // check broadcast
        enum Eshape status = SHAPE_ACCEPT;
        unsigned int ope_a = 0;
        unsigned int ope_b = 0;
        {
            int n_equal = 0;
            int n_broadcast = 0;
            int n_miss = 0;
            for(int i=0;i<num;i++){
                if( na[i] == nb[i] ){
                    n_equal += 1;
                } else if( na[i] == 1 ){
                    n_broadcast += 1;
                } else if( nb[i] == 1 ){
                    n_broadcast += 1;
                } else {
                    cout<<"broadcast mismatch dimension no."<<i<<endl;
                    n_miss += 1;
                }
            }
#ifdef _DEBUG
            cout<<"broadcast check:  equal="<<n_equal<<" broadcast="<<n_broadcast<<" miss="<<n_miss<<endl;
#endif
            if(n_equal == num){
                // success without broadcast (na=nb)
            }else if( (n_equal+n_broadcast) == num ) {
                status = SHAPE_BROADCAST;
            } else {
                cout<<"broadcast error. " <<endl; 
                return { SHAPE_REJECT, 0, 0, shape };
            }
            for(int i=0;i<num;i++)
            {
                nc[i] = std::max(na[i],nb[i]);
            }
            if( status ==  SHAPE_BROADCAST )
            {
                // broadcast shape
                for(int i=0;i<num;i++)
                {
                    unsigned int q = 1 << i;
                    if( nc[i] != na[i] )  ope_a += q;
                    if( nc[i] != nb[i] )  ope_b += q;
                }
            }
            if( eb_type == EB_MATMUL )  // matmul case
            {
                nc[num]   = na[num];
                nc[num+1] = nb[num+1];
            }
        }
        //if( status >= 0 )
        {
            vector<size_t> sv;
            for(int i=0;i<n_dim;i++)  sv.push_back( nc[i] );
            shape = sv;
        }
        
#ifdef _DEBUG
        cout<<"matmul broadcast check"<<endl;
        cout<<"   status="<<status<<endl;
        cout<<"   ope_a="<<ope_a<<", ope_b="<<ope_b<<endl;
        cout<<"   shape (";
        for(unsigned int i=0;i<shape.size();i++) cout<<shape[i]<<",";
        cout<<")"<<endl;
#endif
        return { status, ope_a, ope_b, shape };
    }
    Tensor broadcast_sum( Tensor& g, unsigned int ope_type, int n_dim )
    {
        unsigned int q = 1 << n_dim;
        q--;
        
        if( ope_type == 0 )  return g;
        if( ope_type == q )  return xt::sum( g );
        
        vector<size_t> sv;
        for(int i=0;i<n_dim;i++)
        {
            q = 1 << i;
            if( ope_type & q )  sv.push_back( i );
        }
        if( sv.size() > 0 )
        {
#ifdef _DEBUG
            cout<<"broadcast sum shape (";
            for(unsigned int i=0;i<sv.size();i++) cout<<sv.at(i)<<",";
            cout<<")"<<endl;
#endif
            return xt::sum( g, sv );
        }
        return g;
    }
};
    
class AddOp : public BroadcastBase {
public:
    AddOp(){}
    
    bool forward()
    {
        print_message( "add(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        bc = broadcast_check( inputs[0]->output, inputs[1]->output );
        output = inputs[0]->output + inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "add(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() ) 
        {
            if( bc.ope_type[0] != 0 ) {
                //inputs[0]->grad += xt::sum( this->grad );
                Tensor ga = broadcast_sum( this->grad, bc.ope_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad;
            }
        }
        if( inputs[1]->is_grad() )
        {
            if( bc.ope_type[1] != 0 ) {
                //inputs[1]->grad += xt::sum( this->grad *s );
                Tensor gb = broadcast_sum( this->grad, bc.ope_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += gb * s;
            } else {
                inputs[1]->grad += this->grad * s;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "add" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "add grad a", 0 );
        display_grad_shape1( "add grad b", 1 );
    }
};

class SubOp : public BroadcastBase {
public:
    SubOp(){}
    
    bool forward()
    {
        print_message( "sub(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        bc = broadcast_check( inputs[0]->output, inputs[1]->output );
        output = inputs[0]->output - inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "sub(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() )
        {
            if( bc.ope_type[0] != 0 ) {
                //inputs[0]->grad += xt::sum( this->grad );
                Tensor ga = broadcast_sum( this->grad, bc.ope_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad;
            }
        }
        if( inputs[1]->is_grad() )
        {
            if( bc.ope_type[1] != 0 ) {
                //Tensor tmp = this->grad * s;
                //inputs[1]->grad -= xt::sum( tmp );
                Tensor gb = broadcast_sum( this->grad, bc.ope_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad -= gb * s;
            } else {
                inputs[1]->grad -= this->grad * s;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "sub" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "sub grad a", 0 );
        display_grad_shape1( "sub grad b", 1 );
    }
};

class MulOp : public BroadcastBase {
public:
    MulOp(){}
    
    bool forward()
    {
        print_message( "mul(forward)" );
        bc = broadcast_check( inputs[0]->output, inputs[1]->output );
        this->output = inputs[0]->output * inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "mul(backward)" );
        if( inputs[0]->is_grad() ) 
        {
            if( bc.ope_type[0] != 0 ) {
                //Tensor tmp = this->grad * inputs[1]->output;
                //inputs[0]->grad += xt::sum( tmp );
                Tensor tmp = this->grad * inputs[1]->output;
                Tensor ga = broadcast_sum( tmp, bc.ope_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad * inputs[1]->output;
            }
        }
        if( inputs[1]->is_grad() )
        {
            if( bc.ope_type[1] != 0 ) {
                //Tensor tmp = this->grad * inputs[0]->output;
                //inputs[1]->grad += xt::sum( tmp );
                Tensor tmp = this->grad * inputs[0]->output;
                Tensor ga = broadcast_sum( tmp, bc.ope_type[1], bc.shape.size() );
                ga.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += ga;
            } else {
                inputs[1]->grad += this->grad * inputs[0]->output;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "mul" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "mul grad a", 0 );
        display_grad_shape1( "mul grad b", 1 );
    }
};

class DivOp : public BroadcastBase{
public:
    DivOp(){}
    
    bool forward()
    {
        print_message( "div(forward)" );
        bc = broadcast_check( inputs[0]->output, inputs[1]->output );
        output = inputs[0]->output / inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "div(backward)" );
        Tensor& x0 = inputs[0]->output;
        Tensor& x1 = inputs[1]->output;
        if( inputs[0]->is_grad() )
        {
            if( bc.ope_type[0] != 0 ) {
                //Tensor tmp = this->grad / x1;
                //inputs[0]->grad += xt::sum( tmp );
                Tensor tmp = this->grad / x1;
                Tensor ga = broadcast_sum( tmp, bc.ope_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad / x1;
            }
        }
        if( inputs[1]->is_grad() )
        {
             if( bc.ope_type[1] != 0 ) {
                //Tensor tmp = this->grad * ( -x0 / (x1*x1) );
                //inputs[1]->grad += xt::sum( tmp )
                Tensor tmp = this->grad * ( -x0 / (x1*x1) );
                Tensor gb = broadcast_sum( tmp, bc.ope_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += gb;
            } else {
                inputs[1]->grad += this->grad * ( -x0 / (x1*x1) );
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "div" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "div grad a", 0 );
        display_grad_shape1( "div grad b", 1 );
    }
};

class RsubOp : public MCTNode {
public:
    RsubOp(){}
    
    bool forward()
    {
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
        return true;
    }
};

class ExpOp : public MCTNode {
public:
    ExpOp(){}
    int axis;
    
    bool forward()
    {
        print_message( "exp(forward)" );
        output = xt::exp( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "exp(backward)" );
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad * output;
        return true;
    }
};

class LogOp : public MCTNode { 
public:
    LogOp(){}
    
    bool forward()
    {
        print_message( "log(forward)" );
        output = xt::log(inputs[0]->output);
        return true;
    }
    bool backward()
    {
        print_message( "log(backward)" );
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad / inputs[0]->output;
        return true;
    }
};

class Log1pOp : public MCTNode { 
public:
    Log1pOp(){}
    
    bool forward()
    {
        print_message( "log1p(forward)" );
        output = xt::log( inputs[0]->output + 1.0 );
        return true;
    }
    bool backward()
    {
        print_message( "log1p(backward)" );
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad / ( inputs[0]->output + 1.0 );
        return true;
    }
};

class PowOp : public MCTNode {
public:
    PowOp(){}
    
    bool forward()
    {
        print_message( "pow(forward)" );
        output = xt::pow( inputs[0]->output, inputs[1]->output );
        return true;
    }
    bool backward()
    {
        print_message( "pow(backward)" );
        Tensor& x = inputs[0]->output;
        fprec   c = (fprec)inputs[1]->output[0];
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad * c * xt::pow( x, c-1.0 );
        return true;
    }
    void display_shape()
    {
        display_shape1( "pow" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "pow grad", 0 );
    }
};

class MatMulBase : public BroadcastBase {
public:
    MatMulBase(){}
    
    unsigned int batch_size( BroadcastResult &bc )
    {
        int n = bc.shape.size();
        unsigned int num = 1;
        for(int k=0;k<n-2;k++)  num *= bc.shape[k];
        return num;
    }
    Tshape broadcast_shape( Tensor& a, BroadcastResult& bc )
    {
        Tshape as = a.shape();
        int    az = as.size();
        
        int n = bc.shape.size();
        vector<size_t> sv;
        for(int i=0;i<n-2;i++)  sv.push_back( bc.shape[i] );
        sv.push_back( as[az-2] );
        sv.push_back( as[az-1] );
        return (Tshape)sv;
    }
    Tshape batch_shape( Tensor& a, BroadcastResult& bc )
    {
        Tshape as = a.shape();
        int    az = as.size();
        
        unsigned int num = batch_size( bc );
        
        vector<size_t> sv;
        sv.push_back( num );
        sv.push_back( as[az-2] );
        sv.push_back( as[az-1] );
        return (Tshape)sv;
    }
    Tensor broadcast_dot( Tensor &a, Tensor &b, BroadcastResult& bc )
    {
        unsigned int num = batch_size( bc );
        
        Tshape sha1 = broadcast_shape( a, bc );
        Tensor a2 = xt::broadcast( a, sha1 );
        Tshape sha2 = batch_shape( a, bc );
        a2.reshape( sha2 );
        
        Tshape shb1 = broadcast_shape( b, bc );
        Tensor b2 = xt::broadcast( b, shb1 );
        Tshape shb2 = batch_shape( b, bc );
        b2.reshape( shb2 );
        
        Tshape as = a2.shape();
        Tshape bs = b2.shape();
        int    az = as.size();
        int    bz = bs.size();
        
        Tshape sh = { num,as[az-2],bs[bz-1] };
        Tensor tc( sh );
        for(unsigned int k=0;k<num;k++)
        {
            Tensor ta = xt::view( a2, k, xt::all(), xt::all() );
            Tensor tb = xt::view( b2, k, xt::all(), xt::all() );
            Tensor temp = DOT(ta,tb);
            xt::view(tc, k, xt::all(), xt::all() ) = temp;
        }
        return ( tc.reshape( bc.shape ) );
    }
    std::tuple<Tensor,Tensor> broadcast_dotgrad( Tensor &gc, Tensor &a, Tensor &b, BroadcastResult bc )
    {
        unsigned int num = batch_size( bc );
        
        Tshape sha1 = broadcast_shape( a, bc );
        Tensor a2 = xt::broadcast( a, sha1 );
        Tshape sha2 = batch_shape( a, bc );
        a2.reshape( sha2 );
        
        Tshape shb1 = broadcast_shape( b, bc );
        Tensor b2 = xt::broadcast( b, shb1 );
        Tshape shb2 = batch_shape( b, bc );
        b2.reshape( shb2 );
        
        Tshape as = a2.shape();
        Tshape bs = b2.shape();
        int    az = as.size();
        int    bz = bs.size();
        
        Tshape gs = gc.shape();
        int    gz = gs.size();
        Tensor g2 = gc;
        g2.reshape( { num, gs[gz-2], gs[gz-1] } );
        
        Tshape sha = { num, gs[gz-2], bs[bz-2] };
        Tensor tga( sha );
        for(unsigned int k=0;k<num;k++)
        {
            Tensor ta = xt::view( g2, k, xt::all(), xt::all() );
            Tensor tb = xt::view( b2, k, xt::all(), xt::all() );
            Tensor temp = DOT( ta, xt::transpose(tb) );
            xt::view(tga, k, xt::all(), xt::all() ) = temp;
        }
        tga.reshape( sha1 );
        
        Tshape shb = { num, as[az-1], gs[gz-1] };
        Tensor tgb( shb );
        for(unsigned int k=0;k<num;k++)
        {
            Tensor ta = xt::view( a2, k, xt::all(), xt::all() );
            Tensor tb = xt::view( g2, k, xt::all(), xt::all() );
            Tensor temp = DOT( xt::transpose(ta), tb );
            xt::view(tgb, k, xt::all(), xt::all() ) = temp;
        }
        tgb.reshape( shb1 );
        
        Tensor ga = broadcast_sum( tga, bc.ope_type[0], num+2 );
        Tensor gb = broadcast_sum( tgb, bc.ope_type[1], num+2 );
        
        ga.reshape( a.shape() );
        gb.reshape( b.shape() );
        return std::make_tuple( ga, gb );
    }
    
    bool forward()  { return true; }
    bool backward() { return true; }
    
    void display_shape() {}
    void display_grad_shape() {}
};

class DotOp:public MCTNode{
public:
    DotOp(){}
    
    bool forward()
    {
        if( inputs.size() != 2 )
        {
            cout<<"Error:DotOp"<<endl;
            return false;
        }
        print_message( "dot(forward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& b  = inputs[1]->output;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        
        if( az != 1 || bz != 1 )
        {
            cout<<"Error:DotOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        output = DOT( a, b );
        return true;
    }
    bool backward()
    {
        if( inputs.size() != 2 ) 
        {
            cout<<"Error:DotOp"<<endl;
            return false;
        }
        print_message( "dot(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& b  = inputs[1]->output;
        Tensor& ga = inputs[0]->grad;
        Tensor& gb = inputs[1]->grad;
        Tensor& gc = this->grad;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
       
        if( az != 1 || bz != 1 )
        {
            cout<<"Error:DotOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        ga += gc * b;
        gb += gc * a;
        return true;
    }
    void display_shape()
    {
        display_shape1( "dot" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "dot grad a", 0 );
        display_grad_shape1( "dot grad b", 1 );
    }
};

class MatMulOp : public MatMulBase {
public:
    MatMulOp(){}
    
    bool forward()
    {
        if( inputs.size() != 2 ) 
        {
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        print_message( "matmul(forward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& b  = inputs[1]->output;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        
        if( az < 1 || bz < 1 )
        {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( az < 3 && bz < 3 ) {
            
            //output = DOT(a,b);
            if( az == 1 ){
                Tensor a1 = a;
                a1.reshape( { 1, as[0] } );
                if( bz == 1 ) {
                    Tensor b1 = b;
                    b1.reshape( { bs[0], 1 } );
                    output = DOT( a1, b1 );
                } else {
                    output = DOT( a1, b );
                }
            } else {
                if( bz == 1 ) {
                    Tensor b1 = b;
                    b1.reshape( { bs[0], 1 } );
                    output = DOT( a, b1 );
                } else {
                    output = DOT( a, b );
                }
            }
            
        } else if( az > 2 || bz > 2 ) {
            
            //output = DOT(a,b);
            if( az == 1 ) {
                Tensor a1 = a;
                a1.reshape( { 1, as[0] } );
                bc = broadcast_check( a1, b, EB_MATMUL );
                output = broadcast_dot( a1, b, bc );
            } else {
                if( bz == 1 ) {
                    Tensor b1 = b;
                    b1.reshape( { bs[0], 1 } );
                    bc = broadcast_check( a, b1, EB_MATMUL );
                    output = broadcast_dot( a, b1, bc );
                } else {
                    bc = broadcast_check( a, b, EB_MATMUL );
                    output = broadcast_dot( a, b, bc );
                }
            }
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size() != 2 ) 
        {
            cout<<"Error:MatMulOp"<<endl;
            return false;
        }
        print_message( "matmul(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& b  = inputs[1]->output;
        Tensor& ga = inputs[0]->grad;
        Tensor& gb = inputs[1]->grad;
        Tensor& gc = this->grad;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        
        if( az < 1 || bz < 1 )
        {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( az < 3 && bz < 3 )
        {
            Tensor ga2,gb2;
            if( bz == 1 ) {
                Tensor b1 = b;
                b1.reshape( { bs[0], 1 } );
                ga2 = DOT( gc, xt::transpose( b1 ) );
            } else {
                ga2 = DOT( gc, xt::transpose( b ) );
            }
            if( az == 1 ) {
                Tensor a1 = a;
                a1.reshape( { 1, as[0] } );
                gb2 = DOT( xt::transpose( a1 ), gc );
            } else {
                gb2 = DOT( xt::transpose( a ), gc );
            }
            if( az == 1 )  ga2.reshape( as );
            if( bz == 1 )  gb2.reshape( bs );
            ga += ga2;
            gb += gb2;
            
        } else if( az > 2 || bz > 2 ) {
            
            Tensor ga2,gb2;
            if( az == 1 ) {
                Tensor a1 = a;
                a1.reshape( { 1, as[0] } );
                std::tie(ga2,gb2) = broadcast_dotgrad( gc, a1, b, bc );
            } else {
                if( bz == 1 ) {
                    Tensor b1 = b;
                    b1.reshape( { bs[0], 1 } );
                    std::tie(ga2,gb2) = broadcast_dotgrad( gc, a, b1, bc );
                } else {
                    std::tie(ga2,gb2) = broadcast_dotgrad( gc, a, b, bc );
                }
            }
            ga += ga2;
            gb += gb2;
            
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "matmul" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "matmul grad a", 0 );
        display_grad_shape1( "matmul grad b", 1 );
    }
};

class LinearOp : public MatMulBase {
public:
    LinearOp(){}
    
    bool forward()
    {
        if( inputs.size() != 3 )
        {
            cout<<"Error:LinearOp"<<endl;
            return false;
        }
        print_message( "linear(forward)" );
        Tensor& a  = inputs[0]->output;  // x
        Tensor& b  = inputs[1]->output;  // weight
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        
        if( az == 1 && bz == 2 ) 
        {
            Tensor a1 = a;
            a1.reshape( { 1, as[0] } );
            output = DOT( a1, xt::transpose( b ) );
        
        } else if( az == 2 && bz == 2 ) {
            
            output = DOT( a , xt::transpose( b ) );
            
        } else if( az > 2 && bz == 2 ) {
        
            Tensor bt = xt::transpose(b);
            bc = broadcast_check( a, bt, EB_MATMUL );
            output = broadcast_dot( a, bt, bc );
            
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( inputs[2] )
        {
            Tensor& d  = inputs[2]->output;  // bias
            Tshape  ds = d.shape();
            int     dz = ds.size();
            
            if( dz == 1 && ( ds[0] == bs[0] ) ) {
                output += inputs[2]->output;
            } else {
                cout<<"Error:MatMulOp"<<endl;
                cout<<"Error:D:"<<ds[0]<<" B:"<< bs[0]<<endl;
                return false;
            }
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size() != 3 ) 
        {
            cout<<"Error:LinearOp"<<endl;
            return false;
        }
        print_message( "linear(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& b  = inputs[1]->output;
        Tensor& ga = inputs[0]->grad;
        Tensor& gb = inputs[1]->grad;
        Tensor& gc = this->grad;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
            
        if( az == 1 && bz == 2 )
        {
            Tensor ga2 = DOT( gc, b );
            ga2.reshape( as );
            ga += ga2;
            
            Tensor a1 = a;
            a1.reshape( { 1, as[0] } );
            gb += DOT( xt::transpose( gc ), a1 );
                
        } else if( az == 2 && bz == 2 ) {
                
            ga += DOT( gc, b );
            gb += DOT( xt::transpose( gc ), a );
            
        } else if( az > 2 && bz== 2 ) {
            
            Tensor bt = xt::transpose( b );
            Tensor ga2,gb2;
            std::tie(ga2,gb2) = broadcast_dotgrad( gc, a, bt, bc );
            ga += ga2;
            gb += gb2;
            
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( inputs[2] )
        {
            Tensor& d  = inputs[2]->output;  // bias
            Tshape  ds = d.shape();
            int     dz = ds.size();
        
            if( dz == 1 && ( ds[0] == bs[0] ) )
            {
                Tensor& gd = inputs[2]->grad;
                gd += xt::sum( gc,{0} );
            } else {
                cout<<"Error:MatMulOp"<<endl;
                cout<<"Error:D:"<<ds[0]<<" B:"<< bs[0]<<endl;
                return false;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "linear" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "linear grad x", 0 );
        display_grad_shape1( "linear grad w", 1 );
        display_grad_shape1( "linear grad b", 2 );
    }
};

class AddMmOp : public MCTNode { 
public:
    AddMmOp() { dim = -1; } // d:noset
    int dim;
    
    bool forward()
    {
        if( inputs.size() != 5 )
        {
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
        print_message( "addmm(forward)" );
        Tensor& a  = inputs[1]->output;  // mat1
        Tensor& b  = inputs[2]->output;  // mat2
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        if( az != 2 || bz != 2 )
        {
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( as[1] != bs[0] )
        {
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A_shape:"<<as[0]<<","<<as[1]<<" B_shape:"<<bs[0]<<","<<bs[1]<<endl;
            return false;
        }
        fprec alpha = ( inputs[4] ) ? (fprec)inputs[4]->output[0] : 1.0;
        fprec beta  = ( inputs[3] ) ? (fprec)inputs[3]->output[0] : 1.0;
        if( inputs[0] )
        {
            Tensor& d  = inputs[0]->output;  // input
            Tshape  ds = d.shape();
            int     dz = ds.size();

            if( dz < 1 ) {
                dim = 0;
            } else if( dz == 1 ) {
                if( ds[0] == 1 )      dim = 0;
                if( ds[0] == bs[1] )  dim = 1;
            } else if( dz == 2 ) {
                if( ds[0] == as[0] && ds[1] == bs[1] )  dim = 2;
            }
            if( dim < 0 )
            {
                cout<<"Error:AddMmOp"<<endl;
                cout<<"Error:D:"<<dz<<endl;
                if( dz == 1 )  cout<<"Error:D_shape"<<ds[0]<<endl;
                if( dz == 2 )  cout<<"Error:D_shape"<<ds[0]<<","<<ds[1]<<endl;
                return false;
            }
            output = DOT(a,b) * alpha + d * beta;
        } else {
            output = DOT(a,b) * alpha;
        }
        return true;
    }
    bool backward()
    {
        if( inputs.size()!=5 )
        {
            cout<<"Error:AddMmOp"<<endl;
            return false;
        }
        print_message( "addmm(backward)" );
        Tensor& a  = inputs[1]->output; // mat1
        Tensor& b  = inputs[2]->output; // mat2
        Tensor& ga = inputs[1]->grad;
        Tensor& gb = inputs[2]->grad;
        Tensor& gc = this->grad;
        Tshape  as = a.shape();
        Tshape  bs = b.shape();
        int     az = as.size();
        int     bz = bs.size();
        fprec  alpha = ( inputs[4] ) ? (fprec)inputs[4]->output[0] : 1.0;
        fprec  beta  = ( inputs[3] ) ? (fprec)inputs[3]->output[0] : 1.0;
        
        if( az==2 && bz==2 ) 
        {
            ga += DOT( gc, xt::transpose( b ) ) * alpha;
            gb += DOT( xt::transpose( a ), gc ) * alpha;
            
        } else {
            
            cout<<"Error:AddMmOp"<<endl;
            cout<<"Error:A:"<<az<<" B:"<< bz<<endl;
            return false;
        }
        if( inputs[0] )
        {
            Tensor& gd = inputs[0]->grad;
            if( dim == 0 ) {
                gd += xt::sum( gc ) * beta;
            } else if( dim == 1 ) {
                gd += xt::sum( gc,{0} ) * beta;
            } else if( dim == 2 ) {
                gd += gc * beta;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "addmm" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "addmm grad b", 0 );
        display_grad_shape1( "addmm grad x", 1 );
        display_grad_shape1( "addmm grad w", 2 );
    }
};

class TransposeOp : public MCTNode {
public:
    TransposeOp(){}
    
    bool forward()
    {
        print_message( "transpose(forward)" );
        output = xt::transpose( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "transpose(backward)" );
        inputs[0]->grad += xt::transpose( this->grad );
        return true;
    }
    void display_shape()
    {
        display_shape1( "transpose" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "transpose grad", 0 );
    }
};

class MaxOp : public MCTNode {
public:
    MaxOp( int ax=1 ) { axis = ax; }
    int    axis;
    Tbool  cond;
    
    bool forward()
    {
        print_message( "max(forward)" );
        axis = (int)inputs[1]->output[0];
        output = xt::amax( inputs[0]->output, {axis} );
        return true;
    }
    bool backward()
    {
        print_message( "max(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& ga = inputs[0]->grad;
        Tensor  o  = output;
        Tensor& gc = this->grad;
        
        if( axis == 1 )
        {
            o.reshape( {-1,1} );
            ga.reshape( {-1,1} );
        }
        cond = xt::equal( a, o );
        cout<<"cond"<<cond<<endl;
        ga = gc * cond;
        //print_tensor( "max ga", ga );
        return true;
    }
    void display_shape()
    {
        display_shape1( "max" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "max grad", 0 );
    }
};

class MinOp : public MCTNode {
public:
    MinOp( int ax=1 ){ axis = ax; }
    int    axis;
    Tbool  cond;
    
    bool forward()
    {
        print_message( "min(forward)" );
        axis = (int)inputs[1]->output[0];
        output = xt::amin( inputs[0]->output, {axis} );
        return true;
    }
    bool backward() // same as MaxOp::backward()
    {
        print_message( "min(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& ga = inputs[0]->grad;
        Tensor  o  = output;
        Tensor& gc = this->grad;
        
        if( axis == 1 )
        {
            o.reshape( {-1,1} );
            ga.reshape( {-1,1} );
        }
        cond = xt::equal( a, o );
        ga = gc * cond;
        //print_tensor( "min ga", ga );
        return true;
    }
    void display_shape()
    {
        display_shape1( "min" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "min grad", 0 );
    }
};

class SigmoidOp : public MCTNode {
public:
    SigmoidOp(){}
    
    bool forward()
    {
        print_message( "sigmoid(forward)" );
        output = 1.0 / ( 1.0+xt::exp( -inputs[0]->output ) );
        return true;
    }
    bool backward()
    {
        print_message( "sigmoid(backward)" );
        inputs[0]->grad += this->grad * output * ( 1.0 - output );
        return true;
    }
    void display_shape()
    {
        display_shape1( "sigmoid" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "sigmoid grad", 0 );
    }
};

class ReluOp : public MCTNode {
public:
    ReluOp(){}
    
    bool forward()
    {
        print_message( "relu(forward)" );
        output = xt::maximum( inputs[0]->output, 0 );
        return true;
    }
    bool backward()
    {
        print_message( "relu(backward)" );
        inputs[0]->grad += this->grad * ( inputs[0]->output > 0 );
        return true;
    }
    void display_shape()
    {
        display_shape1( "relu" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "relu grad", 0 );
    }
};

class HardTanhOp : public MCTNode {
public:
    HardTanhOp( fprec v1=-1.0, fprec v2=1.0 )
    {
        min_val = v1;
        max_val = v2;
    }
    fprec  min_val;
    fprec  max_val;

    bool forward()
    {
        print_message( "hardtanh(forward)" );
        min_val = (fprec)inputs[1]->output[0];
        max_val = (fprec)inputs[2]->output[0];
        output = xt::maximum( inputs[0]->output, min_val );
        output = xt::minimum( output, max_val );
        return true;
    }
    bool backward()
    {
        print_message( "hardtanh(backward)" );
        Tensor& y  = inputs[0]->output;
        Tensor& gd = inputs[0]->grad;
        gd += this->grad * ( y > min_val && y < max_val );
        return true;
    }
    void display_shape()
    {
        display_shape1( "hardtanh" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "hardtanh grad", 0 );
    }
};

class EluOp : public MCTNode {
public:
    EluOp(fprec a=1.0) { alpha = a; }
    fprec  alpha;
    Tbool  mask;
    
    bool forward()
    {
        print_message( "elu(forward)" );
        Tensor& y = inputs[0]->output;
        alpha = (fprec)inputs[1]->output[0];
        output = xt::maximum( y, 0.0 );
        mask = ( y < 0.0 );
        auto m = xt::masked_view( output, mask );
        m = xt::minimum( alpha * ( xt::exp( y ) - 1.0 ), 0.0 );
        return true;
    }
    bool backward()
    {
        print_message( "elu(backward)" );
        Tensor& y  = inputs[0]->output;
        Tensor& gd = inputs[0]->grad;
        gd = this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = alpha * xt::exp( y );
        return true;
    }
    void display_shape()
    {
        display_shape1( "elu" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "elu grad", 0 );
    }
};

class LeakyReluOp : public MCTNode {
public:
    LeakyReluOp( fprec s=0.01 ) { slope=s; }
    fprec  slope;
    Tbool  mask;
    
    bool forward()
    {
        print_message( "leakyrelu(forward)" );
        Tensor& y = inputs[0]->output;
        slope = (fprec)inputs[1]->output[0];
        mask  = ( y < 0.0 );
        output = y;
        auto m = xt::masked_view( output, mask );
        m = y * slope;
        return true;
    }
    bool backward()
    {
        print_message( "leakyrelu(backward)" );
        Tensor& y  = inputs[0]->output;
        Tensor& gd = inputs[0]->grad;
        gd += this->grad;
        mask = ( y < 0.0 );
        auto m = xt::masked_view( gd, mask );
        m = gd * slope;
        return true;
    }
    void display_shape()
    {
        display_shape1( "leakyrelu" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "leakyrelu grad", 0 );
    }
};

class SoftplusOp : public MCTNode {
public:
    SoftplusOp(){}
    Tbool  mask;
    
    bool forward()
    {
        fprec beta = (fprec)inputs[1]->output[0];
        if( beta == 0.0 )
        {
            cout<<"Error:Beta is 0.0"<<endl;
            return false;
        }
        print_message( "softplus(forward)" );
        fprec threshold = (fprec)inputs[2]->output[0];
        output = inputs[0]->output;
        mask = ( output * beta < threshold );
        auto m = xt::masked_view( output, mask );
        m = xt::log( 1.0 + xt::exp( beta * output ) ) / beta;
        return true;
    }
    bool backward()
    {
        print_message( "softplus(backward)" );
        Tensor& a  = inputs[0]->output;
        Tensor& ga = inputs[0]->grad;
        fprec beta      = (fprec)inputs[1]->output[0];
        fprec threshold = (fprec)inputs[2]->output[0];
        ga = this->grad;
        mask = ( a * beta < threshold );
        auto m = xt::masked_view( ga, mask );
        m = 1.0 / ( 1.0 + 1.0/xt::exp( beta * a ) );
        return true;
    }
    void display_shape()
    {
        display_shape1( "softplus" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "softplus grad", 0 );
    }
};

class TanhOp : public MCTNode {
public:
    TanhOp(){}
    
    bool forward()
    {
        print_message( "tanh(forward)" );
        output = xt::tanh( inputs[0]->output );
        return true;
    }
    bool backward()
    {
        print_message( "tanh(backward)" );
        inputs[0]->grad += this->grad * ( 1.0 - output * output );
        return true;
    }
    void display_shape()
    {
        display_shape1( "tanh" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "tanh grad", 0 );
    }
};

class SoftmaxBase : public MCTNode {
public:
    SoftmaxBase(){}
    
    bool forward()  { return true; }
    bool backward() { return true; }
    
    Tensor _row2col( Tensor& b, Tshape as )
    {
        Tshape bs = b.shape();
        int    az = as.size();
        int    bz = bs.size();
        if( az == bz )  return b;
        
        if( az == 2 && bz == 1 ) {
            return xt::view( b, xt::all(), xt::newaxis() );
            //return xt::expand_dims( b, {1} );
            //return b.reshape({-1,1} );
        }
        return b; // error
    }
    virtual Tensor _softmax( Tensor& a, int ax )
    {
        //print_message( "_softmax" );
        Tshape as = a.shape();
        Tensor ga = this->grad;
       
        Tensor sm = xt::amax( a, {ax} );
        sm = _row2col( sm, as );
        Tensor sa = a - sm;
        Tensor se = xt::exp( sa );
        Tensor sd = xt::sum( se, {ax} );
        sd = _row2col( sd, as );
        Tensor y = se / sd;
        return y;
    }
    virtual Tensor _log_softmax( Tensor& a, int ax )
    {
        //print_message( "_log_softmax" );
        Tshape as = a.shape();
        Tensor sm = xt::amax( a, {ax} );
        sm = _row2col( sm, as );
        Tensor sa = a - sm;
        Tensor se = xt::exp( sa );
        Tensor sd = xt::sum( se, {ax} );
        Tensor sl = xt::log( sd );
        sl = _row2col( sl, as );
        return ( a - (sm + sl) );
    }
};

class SoftmaxOp : public SoftmaxBase {
public:
    SoftmaxOp(int ax=1) { axis = ax; }
    int axis;
    
    bool forward()
    {
        print_message( "softmax(forward)" );
        output = _softmax( inputs[0]->output, axis ); 
        return true;
    }
    bool backward()
    {
        print_message( "softmax(backward)" );
        Tshape as = output.shape();
        Tensor ga = output * this->grad;
        Tensor sg = xt::sum( ga, {axis} );
        sg = _row2col( sg, as );
        inputs[0]->grad += ( ga - output * sg );
        return true;
    }
    void display_shape()
    {
        display_shape1( "softmax" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "softmax grad", 0 );
    }
};

class LogSoftmaxOp : public SoftmaxBase {
public:
    LogSoftmaxOp( int ax=1 ) { axis=ax; }
    int axis;
    
    bool forward()
    {
        print_message( "log_softmax(forward)" );
        output = _log_softmax( inputs[0]->output, axis );
        return true;
    }
    bool backward()
    {
        print_message( "log_softmax(backward)" );
        Tensor& ga = this->grad;
        Tshape  gs = ga.shape();
        Tensor  sg = xt::sum( ga, {axis} );
        sg = _row2col( sg, gs );
        Tensor  se = xt::exp( output );
        inputs[0]->grad += ( ga - se * sg );
        return true;
    }
    void display_shape()
    {
        display_shape1( "logsoftmax" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "logsoftmax grad", 0 );
    }
};
 
class FullLikeOp : public MCTNode {
public:
    FullLikeOp( fprec v=0.0 ) { value = v; }
    fprec value;
    
    bool forward()
    {
        print_message( "full_like(forward)" );
        if( inputs[0] )
        {
            Tensor& a  = inputs[0]->output;
            Tshape  as = a.shape();
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
        return true;
    }
    void display_shape()
    {
        display_shape1( "fulllike" );
    }
};

class ZerosLikeOp : public FullLikeOp {
public:
    ZerosLikeOp():FullLikeOp( fprec v=0.0 ){}

class OnesLikeOp : public FullLikeOp {
public:
    ZerosLikeOp():FullLikeOp( fprec v=1.0 ){}



class ZerosOp : public MCTNode {
public:
    ZerosOp(){}
    
    bool forward()
    {
        print_message( "zeros(forward)" );
        if( inputs[0] )
        {
            vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                vector<size_t> sh;
                for(unsigned int k=0;k<tlist->size();k++)
                {
                    Tensor t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                output = xt::zeros<fprec>( sh );
                return true;
                
            } else {
                
                Tensor& a  = inputs[0]->output;
                Tshape  as = a.shape();
                vector<size_t> sh;
                for(unsigned int k=0;k<as.size();k++)  sh.push_back( as[k]);
                output = xt::zeros<fprec>( sh );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
    void display_shape()
    {
        display_shape1( "zeros" );
    }
};

class OnesOp : public MCTNode {
public:
    OnesOp(){}
    
    bool forward()
    {
        print_message( "ones(forward)" );
        if( inputs[0] )
        {
            vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                vector<size_t> sh;
                for(unsigned int k=0;k<tlist->size();k++)
                {
                    Tensor t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                output = xt::ones<fprec>( sh );
                return true;
                
            } else {
                
                Tensor& a  = inputs[0]->output;
                Tshape  as = a.shape();
                vector<size_t> sh;
                for(unsigned int k=0;k<as.size();k++)  sh.push_back( as[k] );
                output = xt::ones<fprec>( sh );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
    void display_shape()
    {
        display_shape1( "ones" );
    }
};

class RandnOp : public MCTNode {
public:
    RandnOp(){}
    
    bool forward()
    {
        print_message( "randn(forward)" );
        if( inputs[0] )
        {
            vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                vector<size_t> sh;
                for(unsigned int k=0;k<tlist->size();k++)
                {
                    Tensor t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                output = xt::random::randn<fprec>( sh );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
    void display_shape()
    {
        display_shape1( "randn" );
    }
};

class NormalOp : public MCTNode {
public:
    NormalOp(){}
  
    bool forward()
    {
        if( !inputs[0] )  return false;
        if( !inputs[1] )  return false;
        print_message( "normal(forward)" );
        if( inputs[0]->is_grad() ) 
        {
            if( inputs[1]->is_grad() )  // inputs[0,1] are tensors
            {
                Tensor& mean = inputs[0]->output;
                Tensor& std  = inputs[1]->output;
                Tshape  ms   = mean.shape();
            
                Tensor tmp = xt::random::randn<fprec>( ms );
                output = tmp * std + mean;
                return true;
            }
        } else {
            if( !inputs[1]->is_grad() )  // inputs[0,1] are scalar
            {
                if( inputs[2] )  // inputs[2] is tensor shape
                {
                    vector<Tensor>* tlist = inputs[2]->get_tlist();
                    if( tlist )
                    {
                        fprec mean = (fprec)inputs[0]->output[0];
                        fprec std  = (fprec)inputs[1]->output[0];
                        vector<size_t> sh;
                        for(unsigned int k=0;k<tlist->size();k++)
                        {
                            Tensor t1 = tlist->at(k);
                            sh.push_back( (int)t1[0] );
                        }
                        output = xt::random::randn<fprec>( sh, mean, std );
                        return true;
                    }
                }
            }
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
    void display_shape()
    {
        display_shape1( "normal" );
    }
};

class BatchNormOp : public MCTNode {
public:
    BatchNormOp() {}
    
    bool forward()
    {
        print_message( "batchnorm(forward)" );
        Tensor x  = inputs[0]->output;
        Tshape xs = x.shape();
        int n_dim = xs.size();
        
        if( n_dim != 2 && n_dim != 4 )
        {
            cout<<"Error:BatchNormOp input dimension(not 2 or 4)"<<endl;
            cout<<"ndim="<<n_dim<<endl;
            return false;
        }
        
        unsigned int N,C,H,W;
        if( n_dim == 4 )
        {
            N = xs[0]; C = xs[1]; H = xs[2]; W = xs[3];
            vector<int> perm1{ 0, 2, 3, 1 };
            x = xt::transpose( x, perm1 );
            x.reshape( { -1, (int)C } );
        }
        
        Tensor& gamma = inputs[1]->output;  // gamma (weight)
        Tensor& beta  = inputs[2]->output;  // beta  (bias)
        Tensor& running_mean = inputs[3]->output;
        Tensor& running_var  = inputs[4]->output;
        fprec   momentum = (fprec)inputs[6]->output[0];
        fprec   eps      = (fprec)inputs[7]->output[0]; 
        
        // set ntype for no backward
        inputs[3]->set_ntype( VAR_RUNNING );  // running_mean
        inputs[4]->set_ntype( VAR_RUNNING );  // running_var
        
        Tensor xn;
        if( train_mode )
        {
            Tensor mean = xt::mean( x, {0} );
            Tensor var  = xt::variance( x, {0} );
            xn = ( x - mean ) / xt::sqrt( var + eps );
            int    gz = gamma.size();
            Tshape xs = x.shape();
            fprec  m  = (fprec)xs[1] / (fprec)gz;
            fprec  adjust = ( m > 2.0 ) ? (m-1.0): 1.0;  
            running_mean = running_mean * (1-momentum) + momentum * mean;
            running_var  = running_var  * (1-momentum) + momentum * var * adjust;
        } else {
            xn = ( x - running_mean ) / ( xt::sqrt( running_var + eps ) );
        }
    
        if( n_dim == 4 ) {
            Tensor u = gamma * xn + beta;
            u = u.reshape( { N, H, W, C } );
            vector<int> perm2{ 0, 3, 1, 2 };
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
        Tshape   gs = gc.shape();
        int n_dim   = gs.size();
        int n_batch = gs[0];
        
        unsigned int N,C,H,W;
        vector<int> perm1{ 0, 2, 3, 1 };
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
        
        Tensor& gamma = inputs[1]->output;  // gamma (weight)
      //Tensor& beta  = inputs[2]->output;  // beat  (bias)
        Tensor& running_mean = inputs[3]->output;
        Tensor& running_var  = inputs[4]->output;
        fprec   eps   = (fprec)inputs[7]->output[0]; 
        
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
            
        Tensor& gx = inputs[0]->grad; // x
        Tensor& gm = inputs[1]->grad; // gamma
        Tensor& gb = inputs[2]->grad; // beta
            
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
            vector<int> perm2{ 0, 3, 1, 2 };
            gx = xt::transpose( gx, perm2 );
        }
    
        return true;
    }
    
    void display_shape()
    {
        display_shape1( "batchnorm" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "batchnorm grad x", 0 );
        display_grad_shape1( "batchnorm grad gamma", 1 );
        display_grad_shape1( "batchnorm grad beta", 2 );
    }
};

class DropoutOp : public MCTNode {
public:
    DropoutOp( int kd=1 ) { kind = kd; }
    Tensor dropout;
    int kind;  // dropout-type ( 0: normal 1: inverted[pytorch] )
    
    bool forward()
    {
        print_message( "dropout(forward)" );
        Tensor&   x = inputs[0]->output;
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
            //print_tensor("dropout",dropout);
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
        return true;
    }
    void display_shape()
    {
        display_shape1( "dropout" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "dropout grad", 0 );
    }
};

class MseLossOp : public MCTNode {
public:
    MseLossOp(){}
    
    bool forward()
    {
        print_message( "mseloss(forward)" );
        Tensor& a = inputs[0]->output;
        Tensor& b = inputs[1]->output;
        int  type = (int)inputs[2]->output[0];
        Tensor diff = a - b;
        output = xt::sum( xt::pow( diff, 2.0 ) );
        if( type == 1 )  output = output/(fprec)a.size();
        //cout<<"mseloss"<<output<<endl;
        return true;
    }
    bool backward()
    {
        print_message( "mseloss(backward)" );
        Tensor& a = inputs[0]->output;
        Tensor& b = inputs[1]->output;
        int    type = (int)inputs[2]->output[0];
        Tensor diff = a - b;
        Tensor&  ga = inputs[0]->grad;
        Tensor&  gb = inputs[1]->grad;
        Tensor&  gc = this->grad;
        ga = gc * diff * 2.0;
        if( type == 1 )  ga = ga / (fprec)a.size();
        gb = -ga;
        //print_tensor( "mseloss_grad", ga );
        return true;
    }
    fprec get_loss() { return output[0]; }
};

class CrossEntropyLossOp : public SoftmaxBase { // ( == log_softmax )
public:
    CrossEntropyLossOp( int ax=1 ) { axis = ax; }
    int axis;
    
    bool forward()
    {
        print_message( "cross_entropy_loss(forward)" );
        Tensor& a  = inputs[0]->output;
        Tshape  as = a.shape();
        
        Tensor sz = _log_softmax( a, axis );
        
        fprec  h = ( inputs[4] ) ? (fprec)inputs[4]->output[0] : -100.0;
        
        Tensor t = xt::zeros<fprec>( {as[0]} );
        for(unsigned int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            t[i] = sz( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / (fprec)as[0];
        //print_tensor( "crossloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "cross_entropy_loss(backward)" );
        Tensor& a  = inputs[0]->output;
        Tshape  as = a.shape();
        Tensor& ga = this->grad;
        fprec   sc = 1.0 / (fprec)as[0];
        
        Tensor y = _softmax( a, axis );
        
        Tensor one = xt::zeros<fprec>( as );
        for(unsigned int i=0;i<as[0];i++){
            int j = (int)inputs[1]->output[i];
            one(i,j) = 1.0;
        }
        
        inputs[0]->grad = ( y - one ) * ga * sc;
        //print_tensor( "crossloss_grad", inputs[0]->grad );
      
        return true;
    }
    Tensor get_classes()
    {
        Tensor sm = _softmax( inputs[0]->output, axis );
        Tshape sh = sm.shape();
        Tshape shape = { sh[0] };
        
        Tensor lbs( shape );
        for(unsigned int i=0;i<sh[0];i++)
        {
            fprec smax = sm(i,0);
            int jm = 0;
            for(unsigned int j=1;j<sh[1];j++)
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

class BCELossOp : public MCTNode {
public:
    BCELossOp(){};
    
    bool forward()
    {
        print_message( "bceloss(forward)" );
        Tensor& y = inputs[0]->output;
        Tensor& t = inputs[1]->output;
        
        fprec eps = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0e-7;
        fprec d = fprec( y.size() );
        if( inputs[3] ) 
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        Tensor q = t * xt::log(y+eps) + (1-t) * xt::log(1-y+eps);
        output = -xt::sum( q )/ d;
        //print_tensor( "bceloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "bceloss(backward)" );
        Tensor& y = inputs[0]->output;
        Tensor& t = inputs[1]->output;
        
        fprec eps = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0e-7;
        fprec d = fprec( y.size() );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        Tensor& gy = inputs[0]->grad;
        Tensor& gt = inputs[1]->grad;
        gy += this->grad * ( -t/(y+eps) + (1-t)/(1-y+eps) ) / d;
        gt += this->grad * ( -xt::log(y+eps) + xt::log(1-y+eps) ) / d;
        //print_tensor( "bceloss grad", gy );
        return true;
    }
};

class NLLLossOp : public MCTNode {
public:
    NLLLossOp(){}
    
    bool forward() 
    {
        print_message( "nullload(forward)" );
        Tensor& a  = inputs[0]->output;
        Tshape  as = a.shape();
        fprec   h  = (fprec)inputs[4]->output[0];
        
        fprec d = fprec( as[0] );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        Tensor t = xt::zeros<fprec>( {as[0]} );
        for(unsigned int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            t[i] = a( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / d;
        print_tensor( "nllloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "nullload(backward)" );
        Tensor& a  = inputs[0]->output;
        Tshape  as = a.shape();
        
        fprec d = fprec( as[0] );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        fprec sc = 1.0 / d;
        
        Tensor& gc = this->grad;
        print_tensor( "gc ", gc );
        
        Tensor one = xt::zeros<fprec>( as );
        for(unsigned int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            one(i,j) = -gc[0];
        }
        inputs[0]->grad = one * sc;
        print_tensor( "nllloss grad", inputs[0]->grad );
        return true;
    }
};

class BroadcastTensorsOp : public MCTNode{
public:
    BroadcastTensorsOp() 
    {
        broadcast_shape = {0}; 
        result = SHAPE_ACCEPT;
    }
    vector<Tensor>  tlist;  // tensor list
    vector<Tensor>  glist;  // grad tensor list
    
    vector<Tensor>* get_tlist() { return &tlist; };
    vector<Tensor>* get_glist() { return &glist; };
    Tshape broadcast_shape;
    enum Eshape result;
    
private:
    // return value   0: no broadcast
    //               >0; broadcast by shape
    //               <0: broadcast error
    enum Eshape check( vector<Tensor> &a, unsigned int num, Tshape &shape )
    {
        shape = {0};
        if( num <  1 )  return SHAPE_REJECT;
        if( num == 1 )
        {
            shape = a[0].shape();
            return SHAPE_ERROR;  // Tensor only one
        }
        
        vector<Tshape> as(num);
        vector<int>    az(num);
        int n_dim = -1;
        for(unsigned int i=0;i<num;i++)  
        {
            as[i] = a[i].shape();
            az[i] = as[i].size();
            if( n_dim < az[i] )  n_dim = az[i];
        }
        if( n_dim < 1 )  return SHAPE_ACCEPT; // all numbers
        
        unsigned int equal = 1;
        unsigned int err   = 0;
        vector<size_t>  na=extend_shape( as[0], n_dim);

#ifdef _DEBUG
        cout<<"--------------------------"<<endl;
        cout<<"broadcast n_dim="<<n_dim<<endl;
        print_ints( "broadcast tensor-1 ", na, n_dim );
#endif
        
        for(unsigned int k=1;k<num;k++)
        {
            vector<size_t>  nb=extend_shape( as[k], n_dim );
            
            // check broadcast
            int n_equal = 0;
            int n_broadcast = 0;
            int n_miss = 0;
            for(int i=0;i<n_dim;i++){
                if( na[i] == nb[i] ){
                    n_equal += 1;
                } else if( na[i] == 1 ){
                    n_broadcast += 1;
                } else if( nb[i] == 1 ){
                    n_broadcast += 1;
                } else {
                    //cout<<"broadcast mismatch dimension no."<<i<<endl;
                    n_miss += 1;
                }
            }
#ifdef _DEBUG
            string ss = "broadcast tensor-" + std::to_string(k+1) + " ";
            print_ints( ss, nb, n_dim );
            cout<<"broadcast ("<<k<<")  equal="<<ne<<" eq1="<<n1<<" other="<<n2<<endl;
#endif
            
            Eshape status =SHAPE_ACCEPT;
            if( n_equal == n_dim ) {  
                status = SHAPE_ACCEPT;
                equal++;
            } else if( (n_equal+n_broadcast) == n_dim ) {
                status = SHAPE_BROADCAST;
            } else {
                cout<<"broadcast error. ("<<k<<")"<<endl; 
                err += 1;
            }
            if( status == SHAPE_BROADCAST )
            {
                for(int i=0;i<n_dim;i++)
                {
                    na[i] = std::max(na[i], nb[i]);
                }
            }
        }
        
        enum Eshape result = SHAPE_BROADCAST;
        if( err == 0 )
        {
            if( equal == num ) {
                result = SHAPE_ACCEPT;
            } else {
                // set shape size by vector
                //   ex. shape = { na[0], na[1], na[2] } 
                //   vector<size_t> v= { na[0], na[1], na[2] }
                vector<size_t> v;
                for(int i=0;i<n_dim;i++)  v.push_back( na[i] );
                shape = v;
                result = SHAPE_BROADCAST;
                print_ints( "broadcast shape ", na, n_dim );
            }
        } else {
            for(unsigned int i=0;i<num;i++)
            {
                string ss = "tensor(" + std::to_string(i) + ")  shape=";
                print_shape( ss.c_str(), a[i] );
            }
            result = SHAPE_REJECT;
        }
        return result;
    }


    enum Eshape restore_broadcast( Tensor &ga, Tshape as, Tshape os )
    {
        int  az = as.size();
        int  oz = os.size();
        cout<<"az "<<az<<", oz"<<oz<<endl;
        if( oz == 0 )
        {
            ga = (fprec)ga[0];
            return SHAPE_BROADCAST;
        }
        if( az == oz )
        {
            int equal = 0;
            for(int i=0;i<oz;i++)
            {
                if( as[i] == os[i] )  equal++;
            }
            cout<<"equal"<<equal<<endl;
            if( oz == equal )  return SHAPE_ACCEPT;  // no change
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
            return SHAPE_ERROR;
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
        return SHAPE_BROADCAST;
    }
    
public:
    bool forward()
    {
        print_message( "broadcast_tensors(forward)" );
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            if( tlist1 )
            {
                result = check( *tlist1, tlist1->size(), broadcast_shape );
                if( result != 0 )  cout<<"broadcast check="<<result<<endl;
                if( result < 0 )   return false;
                
                tlist.clear();
                glist.clear();
                for(unsigned int i=0;i<tlist1->size();i++)
                {
                    if( result == 0 ) {
                        tlist.push_back( tlist1->at(i) );
                    } else { // result > 0  
                        Tensor &a1 = tlist1->at(i);
                        Tensor  a2 = xt::broadcast( a1, broadcast_shape );
                        tlist.push_back( a2 );
                    }
                }
            }
        }
        return true;
    }
    bool backward()
    {
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(unsigned int i=0;i<glist.size();i++)
                {
                    if( result == 0 ) {
                        glist1->push_back( glist.at(i) );
                    } else { // result > 0 
                        Tensor& a1 = tlist1->at(i);
                        Tensor& g1 = glist.at(i);
                        Tensor  g2 = g1;
                        enum Eshape status = restore_broadcast( g2, g1.shape(), a1.shape() );
                        if( status < 0 )  return false;
                        glist1->push_back( g2 );
                    }
                }
            }
            glist.clear();
        }
        print_message( "broadcast_tensors(backward)" );
        return true;
    }
    void display_shape()
    {
        display_shape1( "broadcast_tensors" );
    }
    void display_grad_shape()
    {
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(unsigned int i=0;i<tlist1->size();i++)
                {
                    Tensor& a1 = tlist1->at(i);
                    Tensor& g1 = glist1->at(i);
                    string s = "broadcast_tensors_" + std::to_string(i) + " grad";
                    display_grad_shape_size( s, id, a1, g1 );
                }
            }
        }
    }
};

// create tensor list
class ListConstructOp : public MCTNode {
public:
    ListConstructOp() {}
    ListConstructOp( string na ) { name = na; }
    
    vector<Tensor>  tlist;  // tensor list
    vector<Tensor>  glist;  // grad tensor list
    vector<Tensor>* get_tlist() { return &tlist; };
    vector<Tensor>* get_glist() { return &glist; };
    
    bool forward()
    {
        print_message( "list_contruct(forward)" );
        if( inputs.size() < 1 )
        {
            cout<<"Error:ListConstructOp"<<endl;
            return false;
        }
        
        tlist.clear();
        glist.clear();
        for(unsigned int i=0;i<inputs.size();i++)
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
        print_message( "list_contruct(backward)" );
        if( glist.size() > 0 )
        {
            for(unsigned int i=0;i<inputs.size();i++)
            {
                if( inputs[i] )
                {
                    if( inputs[i]->is_grad() )
                    {
                        inputs[i]->grad = glist[i];
                    }
                }
            }
        }
        return true;
    }
};

class ListUnpackOp : public MCTNode {
public:
    ListUnpackOp( int id=0 ) { output_id = id; };
    ListUnpackOp( string na, int id=0 ) 
    { 
        name = na;
        output_id = id;
    }
    int  output_id;
    
    bool forward()
    {
        print_message( "list_unpack(forward)" );
        
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            if( tlist1 )
            {
                output = tlist1->at( output_id );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        print_message( "list_unpack(backward)" );
        vector<Tensor>* glist1 = inputs[0]->get_glist();
        if( glist1 )
        {
            int sz = glist1->size();
            if( sz <= output_id )
            {
                for(int i=sz;i<output_id;i++)  // temporary add
                {
                    glist1->push_back( (Tensor)0 );
                }
                glist1->push_back( this->grad );
            } else {
                glist1->at(output_id)= this->grad;
            }
        }
        return true;
    }
    void display_shape()
    {
        display_shape1( "ListUnpack" );
    }
    void display_grad_shape()
    {
        vector<Tensor>* tlist1 = inputs[0]->get_tlist();
        vector<Tensor>* glist1 = inputs[0]->get_glist();
        if( tlist1 && glist1 )
        {
            int sz = glist1->size();
            if( output_id < sz )
            {
                Tensor& a1 = tlist1->at( output_id );
                Tensor& g1 = glist1->at( output_id );
                string s = "ListUnpack_" + std::to_string( output_id ) + " grad";
                display_grad_shape_size( s, id, a1, g1 );
            }
        }
    }
};

class TupleConstructOp : public MCTNode {
public:
    TupleConstructOp() {}
    TupleConstructOp( string na ) { name = na; }
    
    vector<MCTNode*>  ndlist;  // MCTnode pointer list
    vector<MCTNode*>* get_ndlist() { return &ndlist; };
    
    bool forward()
    {
        print_message( "tuplecontruct(forward)" );
        
        ndlist.clear();  
        for(unsigned int i=0;i<inputs.size();i++)
        {
            if( inputs[i] ) {
                ndlist.push_back( inputs[i] );
            } else {
                ndlist.push_back( NULL );
            }
        }
        return true;
    }
    bool backward()
    {
        return true;
    }
};

class TupleUnpackOp : public MCTNode {
public:
    TupleUnpackOp( int id=0 ) { output_id = id; }
    TupleUnpackOp( string na, int id=0  ) 
    { 
        name = na; 
        output_id = id;
    }
    int output_id;
    
    bool forward()
    {
        print_message( "tupleunpack(forward)" );
        
        if( inputs[0] )
        {
            vector<MCTNode*>* ndlist1 = inputs[0]->get_ndlist();
            if( ndlist1 )
            {
                MCTNode *node = ndlist1->at( output_id );
                if( node )  output = node->output;
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        print_message( "tupleunpack(backward)" );
        vector<MCTNode*>* ndlist1 = inputs[0]->get_ndlist();
        if( ndlist1 )
        {
            MCTNode* node = ndlist1->at( output_id );
            if( node )  node->grad += this->grad;
        }
        return true;
    }
};

class SizeOp : public MCTNode {
public:
    SizeOp(){}
    
    bool forward()
    {
        print_message( "size(forward)" );
        Tensor& a  = inputs[0]->output;
        Tshape  as = a.shape();
        int     no = (int)inputs[1]->output[0];
        output = (fprec)as[no];
        return true;
    }
    bool backward()
    {
        return true;
    }
};

class ExpandOp : public MCTNode {
public:
    ExpandOp(){}
    
    bool forward()
    {
        print_message( "expand(forward)" );
        
        if( inputs[1] )  // shape
        {
            vector<Tensor>* tlist = inputs[1]->get_tlist();
            if( tlist )
            {
                vector<size_t> sh;
                for(unsigned int k=0;k<tlist->size();k++)
                {
                    Tensor t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                //Tensor& a = inputs[0]->output;
                output = xt::broadcast( inputs[0]->output, sh );
                return true;
            }
            
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
    void display_shape()
    {
        display_shape1( "expand" );
    }
};

class NumToTensorOp : public MCTNode {
public:
    NumToTensorOp(){}
    
    bool forward()
    {
        print_message( "numtotensor(forward)" );
        output = inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "numtotensor(backward)" );
        inputs[0]->grad += this->grad;
        return true;
    }
};

class IntOp : public MCTNode {
public:
    IntOp(){}
    
    bool forward()
    {
        print_message( "int(forward)" );
        output = inputs[0]->output;
        return true;
    }
    bool backward()
    {
        print_message( "int(backward)" );
        inputs[0]->grad += this->grad;
        return true;
    }
};

class ViewOp : public MCTNode {
public:
    ViewOp(){}
    Tshape  org_shape;
    
    bool forward()
    {
        print_message( "view(forward)" );
        Tensor& a = inputs[0]->output;
        org_shape = a.shape();
        
        if( inputs[1] )
        {
            vector<Tensor>* tlist1 = inputs[1]->get_tlist();
            if( tlist1 )
            {
                vector<int> sh;
                for(unsigned int i=0;i<tlist1->size();i++)
                {
                    Tensor& t = tlist1->at(i);
                    sh.push_back( (int)t[0] );
                }
                output = a.reshape( sh );
            }
            return true;
        }
        return false;
    }
    bool backward()
    {
        print_message( "view(backward)" );
        //Tensor& gc = this->grad;
        //inputs[0]->grad = gc.reshape( org_shape );
        Tensor &ga = inputs[0]->grad;
        ga = this->grad;
        ga.reshape( org_shape );
        return true;
    }
    void display_shape()
    {
        display_shape1( "view" );
    }
    void display_grad_shape()
    {
        display_grad_shape1( "view grad", 0 );
    }
};

class ToOp : public MCTNode {
public:
    ToOp() {}
    
    bool forward()
    {
        print_message( "to(forward)" );
        if( inputs.size() < 2 )
        {
            cout<<"Error:ToOp"<<endl;
            return false;
        }
        if( inputs[0] )
        {
            output = inputs[0]->output;
            return true;
        }
        return false;
    }
    bool backward()
    {
        return true;
    }
};

class DetachOp : public MCTNode {
public:
    DetachOp( int id=0 ) { out_id = id; }
    int out_id;
    
    bool forward()
    {
        print_message( "detach(forward)" );
        
        if( inputs[0] )
        {
            vector<Tensor>* tlist1 = inputs[0]->get_tlist();
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
        return true;
    }
};

class EinsumOp : public MCTNode {
public:
    EinsumOp(){}
    
    bool forward(){
        return true;
    }
};

