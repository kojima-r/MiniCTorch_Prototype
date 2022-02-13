//
//   minictorch.hpp
//
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
    MCTNode()
    {
        //frontcnt = 0;
        //backcnt  = 0;
        id = -1;    // for check
        grad = 0;   // 220120 add
        ntype = 0;  // 220120 add
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
    //int    frontcnt;
    //int    backcnt;
    int    id;    // for check
    int    ntype; // 0: node, 1:constant, 2: attribute, 3: input_var 4:running_*(batchnorm) // 220120 add
    
    void set_inputs( MCTNode *node )
    {
        inputs.push_back( node );
        //if( node )  node->backcnt++;
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
    void set_ntype( int t ) { ntype=t; };
    bool is_grad()
    {
        switch( ntype ) {
        case 1:  // constant
        case 4:  // running_*
            return false;
        }
        return true;
    };
    
    void eval_shape( Tensor::shape_type as, vector<unsigned int>&na, int n_dim )  // 220202 rename
    {
        unsigned int  az = as.size();
        for(int i=0;i<n_dim;i++)  na[i] = 1;
        if( az > 0 ) 
        {
            int inc = n_dim - az;
            for(int i=0;i<az;i++)  na[inc+i] = as[i];
        }
    }
    
    // shape check utility function
    Tensor::shape_type  out_shape;  // 220130 add
    void set_shape( Tensor::shape_type s ) { out_shape=s; };
    int  check_shape_size( string ss, int id, Tensor::shape_type sh1, Tensor::shape_type sh2, int size2 )
    {
        int sz1 = sh1.size();
        if( sz1 < 1 )  return 0;
        int sz2 = sh2.size();
        if( sz1 == 1 && sz2 == 0 ) 
        {
            cout<<"# "<<ss<<" shape "<<"id="<<id<<"  (1) - (scalar) ->0"<<endl;
            return 1;
        }
        if( sz1 != sz2 )
        {
            int nsz = 1;
            for(int i=0;i<sz1;i++)  nsz *= sh1[i];
            cout<<"# "<<ss<<" shape size  "<<"id="<<id<<" "<<nsz<<","<<size2<<","<<sz1<<","<<sz2<<endl;
            if( nsz != size2 )
            {
                cout<<"# "<<ss<<" shape dim error "<<"id="<<id<<"  "<<sz1<<"-"<<sz2<<endl;
                return -1;
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
                return -2;
            }
        }
        return 1;
    }
    int check_grad_shape_size( string s, int id, Tensor& ga, Tensor& gb )  // 220202 mod
    {
        return check_shape_size( s, id, ga.shape(), gb.shape(), gb.size() );
    }
    int check_shape1( string s )
    {
        return check_shape_size( s, id, out_shape, output.shape(), output.size() );
    }
    int check_grad_shape1( string s, int k=0 )
    {
        if( inputs[k] )
        {
            check_grad_shape_size( s, id, inputs[k]->output, inputs[k]->grad );
        }
    }
    virtual int check_shape() {};
    virtual int check_grad_shape() {};
    
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
    void print_shape1( Tensor::shape_type &as )
    {
        cout<<"shape (";
        for(int i=0;i<as.size();i++){
            cout<<as[i]<<",";
        }
        cout<<")"<<endl;
    }
    void print_ints( string s, vector<unsigned int> &v, int nv )
    {
        cout<<s;
        for(int i=0;i<nv;i++)  cout<<v[i]<<",";
        cout<<endl;
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
    
    VariableTensor( Tensor tensor, int t=0 ) 
    {
        this->output = tensor;
        this->ntype = t;
        this->grad = xt::zeros_like( output );
    }
    VariableTensor( string name, Tensor tensor, int t=0 )
    {
        this->output = tensor;
        this->name = name;
        this->ntype = t;
        this->grad = xt::zeros_like( output );
    }
    bool forward()  { return true; }
    bool backward() { return true; }
    void set_output1( fprec o1 ) { output[0] = o1; }
    
    void update( fprec delta ) 
    {
        switch( ntype ) {
        case 2:  // attribute
            output = output - delta * grad;
            break;
        }
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
        
        //cout<<"sum axis"<<axis<<endl;
        //cout<<"sum gc"<<grad<<endl;
        if( axis >=0  ) { 
            auto e = xt::expand_dims( this->grad, axis );
            auto g = xt::repeat( e, sh[axis], axis ); // 220111 mod
            inputs[0]->grad = g;
        } else {
            /* 220113 mod
            auto e  = xt::expand_dims( this->grad, 0 );
            auto g1 = xt::repeat( e, sh[0], 0 ); 
            //cout<<"sum sh"<<sz<<","<<sh[0]<<","<<sh[1]<<endl;
            if( sz > 1 && sh[1] > 1 ) {
                auto g2 = xt::repeat( g1, sh[1], {1} );
                inputs[0]->grad = g2;
            } else {
                inputs[0]->grad = g1;
            }
            */
            //fprec g0 = this->grad[0];
            inputs[0]->grad = xt::full_like( inputs[0]->output, (fprec)this->grad[0] );
            //cout<<"g"<<inputs[0]->grad<<endl;
            
        }
        _backward_inputs();
        return true;
    }
};

class MeanOp:public MCTNode{
    public:
    MeanOp( int ax=-1 ){ axis = ax; };
    int axis;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "mean(forward)" );
        
        if( inputs[0] )
        {
            if( inputs[1] )
            {
                std::vector<Tensor>* tlist = inputs[1]->get_tlist();
                if( tlist )
                {
                    if( tlist->size() > 1 ) 
                    {
                        cout<<"sum argumenst 1 error: list length > 1 "<<endl;
                        return false;
                    }
                    auto& t0 = tlist->at(0);
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
            //cout<<"mean out"<<axis<<","<<output<<endl;
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
                std::vector<Tensor>* tlist = inputs[1]->get_tlist();
                if( tlist )
                {
                    if( tlist->size() > 1 ) 
                    {
                        cout<<"sum argumenst 1 error: list length > 1 "<<endl;
                        return false;
                    }
                    auto& t0 = tlist->at(0);
                    axis = (int)t0[0];
                } else {
                    axis = (int)inputs[1]->output[0];
                }
            } else {
                axis = -1;
            }
            Tensor& a = inputs[0]->output;
            auto   as = a.shape();
            if( axis >=0  ) {
                auto gd = this->grad / (fprec)as[axis];
                auto e = xt::expand_dims( gd, axis );
                auto g = xt::repeat( e, as[axis], axis );
                inputs[0]->grad = g;
            } else {
                fprec gd = this->grad[0] / a.size();
                inputs[0]->grad = xt::full_like( a, gd );
            }
        }
        _backward_inputs();
        return true;
    }
};

class StackOp:public MCTNode
{
    public:
    StackOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "stack(forward)" );
        if( inputs[0] )
        {
            int dim = (int)inputs[1]->output[0];
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                int  tr = tlist->size();
                auto a1 = tlist->at(0);
                auto a1_s = a1.shape();
                int  a1_z = a1_s.size();
                
                std::vector<size_t> sh;
                sh.push_back( tr );
                for(int i=0;i<a1_z;i++)  sh.push_back( a1_s[i] );
                
                Tensor out( sh );
                for(int k=0;k<tr;k++) 
                {
                    xt::xstrided_slice_vector sv;
                    for(int i=0;i<a1_z;i++)
                    {
                        if( i == dim )  sv.push_back( k );
                        else            sv.push_back( xt::all() );
                    }
                    xt::strided_view( out, sv ) = tlist->at(k);
                    //cout<<"stack in("<<k<<")  "<<tlist->at(k)<<endl;
                }
                output = out;
                //cout<<"stack o1"<<output<<endl;
                return true;
            } else {  // yet check
                output = inputs[0]->output;
                //cout<<"stack o2"<<output<<endl;
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
            auto& gc = this->grad;
            auto  gs = gc.shape();
            int   gr = gs.size();
                
            int dim = (int)inputs[1]->output[0];
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            std::vector<Tensor>* glist = inputs[0]->get_glist();
            if( tlist && glist )
            {
                int tr = tlist->size();
                for( int k=0;k<tr;k++)
                {
                    xt::xstrided_slice_vector sv;
                    for(int i=0;i<gr;i++)
                    {
                        if( i == dim )  sv.push_back( k );
                        else            sv.push_back( xt::all() );
                    }
                    auto g = xt::strided_view( gc, sv );
                    glist->push_back( g );
                    //cout<<"stack gc"<<k<<","<<gc<<endl;
                    //cout<<"stack g1"<<k<<","<<g<<endl;
                }
            } else {
                inputs[0]->grad = gc;
                //cout<<"stack g2"<<gc<<endl;
            }
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "stack" );
    }
    int check_grad_shape()
    {
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            std::vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(int i=0;i<tlist1->size();i++)
                {
                    auto& a1 = tlist1->at(i);
                    auto& g1 = glist1->at(i);
                    string s = "stack_" + std::to_string(i) + " grad";
                    check_grad_shape_size( s, id, a1, g1 );
                }
            }
        }
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


struct broadcast_result  // 220208 add
{
    int status;
    int sum_type[2];
    Tensor::shape_type shape;
};

class ArithBase : public MCTNode {
    public:
    ArithBase() {}
    broadcast_result  bc;
    
    bool forward()  { return true; };
    bool backward() { return true; };

    broadcast_result broadcast_check( Tensor& a, Tensor &b, int chk=0 )
    {
        auto as = a.shape();
        auto bs = b.shape();
        int  az = as.size();
        int  bz = bs.size();
        
        Tensor::shape_type shape = {1};
        
        int n_dim = ( az > bz ) ? az : bz;
        if( n_dim < 1 )  return { 0, 0, 0, shape };
        //if( n_dim < 1 )  return 0;
        //if( n_dim < 1 )  return std::make_tuple( 0, 0, 0 ); // all numbers
        
        std::vector<unsigned int>  na(n_dim);
        std::vector<unsigned int>  nb(n_dim);
        std::vector<unsigned int>  sh(n_dim);
        
        eval_shape( as, na, n_dim );
        eval_shape( bs, nb, n_dim );
        if( chk > 0 )
        {
            cout<<"n_dim "<<n_dim<<" <- "<<az<<","<<bz<<endl;
            print_ints( "broadcast tensor-a ", na, n_dim );
            print_ints( "broadcast tensor-b ", nb, n_dim );
        }
        
        // check broadcast
        int status = 0;
        int sum_a=0;
        int sum_b=0;
        {
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
                    cout<<"broadcast mismatch dimension no."<<i<<endl;
                    n2 += 1;
                }
            }
            if( chk > 0 )  cout<<"broadcast check:  equal="<<ne<<" eq1="<<n1<<" other="<<n2<<endl;
            
            if( ne == n_dim ) {  
                status = 0;
            } else if( (ne+n1) == n_dim ) {
                status = 1;
            } else {
                cout<<"broadcast error. " <<endl; 
                status = -1;
            }
            if( status == 1 )
            {
                // broadcast shape
                int q = 1;
                for(int i=0;i<n_dim;i++)
                {
                    sh[i] = ( na[i] > nb[i] ) ? na[i] : nb[i];
                    if( sh[i] != na[i] )  sum_a += q;
                    if( sh[i] != nb[i] )  sum_b += q;
                    q *= 2;
                }
                status = 0;
                if( sum_a > 0 )  status += 1;
                if( sum_b > 0 )  status += 2;
            }
        }
        if( status > 0 )
        {
            std::vector<size_t> sv;
            for(int i=0;i<n_dim;i++)  sv.push_back( sh[i] );
            shape = sv;
            if( chk > 0 )  print_shape1( shape );
        }
        if( chk > 0 )  cout<<"broadcast check:  status="<<status<<"  sum_a="<<sum_a<<", sum_b="<<sum_b<<endl;
        return { status, sum_a, sum_b, shape };
    }
    Tensor broadcast_sum( Tensor& g, int sum_type, int n_dim, int chk=0 )
    {
        int q = 1;
        for(int i=0;i<n_dim;i++)  q *= 2;
        q--;
        
        //cout<<"s="<<s<<",q="<<q<<endl;
        if( sum_type <= 0 )  return g;
        if( sum_type == q )  return xt::sum( g );
        
        q = 1;
        std::vector<size_t> sv;
        for(int i=0;i<n_dim;i++)
        {
            if( sum_type & q )
            {
                //cout<<"sum-"<<i<<endl;
                sv.push_back( i );
            }
            q *= 2;
        }
        if( sv.size() > 0 )
        {
            if( chk > 0 )
            {
                cout<<"sum shape ";
                for(int i=0;i<sv.size();i++) cout<<sv.at(i)<<",";
                cout<<endl;
            }
            return xt::sum( g, sv );
        }
        return g;
    }
};

    
class AddOp : public ArithBase { // MCTNode{
    public:
    AddOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "add(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        bc = broadcast_check( inputs[0]->output, inputs[1]->output ); // 220208 add
        output = inputs[0]->output + inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "add(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() ) 
        {
            /*int oz0 = inputs[0]->output.size();  // 220208 mod
            if( oz0 > 1 ) {
                inputs[0]->grad += this->grad;
            } else {
                inputs[0]->grad += xt::sum( this->grad );
            }*/
            if( bc.sum_type[0] > 0 ) {
                Tensor ga = broadcast_sum( this->grad, bc.sum_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad;
            }
        }
        if( inputs[1]->is_grad() )
        {
            /*int oz1 = inputs[1]->output.size();  // 220208 mod
            if( oz1 > 1 ) {
                inputs[1]->grad += this->grad * s;
            } else {
                //auto tmp = this->grad * s;
                inputs[1]->grad += xt::sum( this->grad *s );
            }*/
            if( bc.sum_type[1] > 0 ) {
                Tensor gb = broadcast_sum( this->grad, bc.sum_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += gb * s;
            } else {
                inputs[1]->grad += this->grad * s;
            }
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "add" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "add grad a", 0 );
        check_grad_shape1( "add grad b", 1 );
    }
};

class SubOp : public ArithBase { // MCTNode{
    public:
    SubOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "sub(forward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        bc = broadcast_check( inputs[0]->output, inputs[1]->output ); // 220208 add
        output = inputs[0]->output - inputs[1]->output * s;
        return true;
    }
    bool backward()
    {
        print_message( "sub(backward)" );
        fprec s = ( inputs[2] ) ? (fprec)inputs[2]->output[0] : 1.0;
        if( inputs[0]->is_grad() )
        {
            /*int oz0 = inputs[0]->output.size(); // 220208 mod
            if( oz0 > 1 ) {
                inputs[0]->grad += this->grad;
            } else {
                inputs[0]->grad += xt::sum( this->grad );
            }*/
            if( bc.sum_type[0] > 0 ) {
                Tensor ga = broadcast_sum( this->grad, bc.sum_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad;
            }
        }
        if( inputs[1]->is_grad() )
        {
            /*int oz1 = inputs[1]->output.size();  // 220208 mod
            if( oz1 > 1 ) {
                inputs[1]->grad -= this->grad * s;
            } else {
                auto tmp = this->grad * s;
                inputs[1]->grad -= xt::sum( tmp );
            }*/
            if( bc.sum_type[1] > 0 ) {
                Tensor gb = broadcast_sum( this->grad, bc.sum_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad -= gb * s;
            } else {
                inputs[1]->grad -= this->grad * s;
            }
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "sub" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "sub grad a", 0 );
        check_grad_shape1( "sub grad b", 1 );
    }
};

class MulOp : public ArithBase { // MCTNode{
    public:
    MulOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "mul(forward)" );
        bc = broadcast_check( inputs[0]->output, inputs[1]->output ); // 220208 add
        this->output = inputs[0]->output * inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "mul(backward)" );
        if( inputs[0]->is_grad() ) 
        {
            /*int oz0 = inputs[0]->output.size();  // 220208 mod
            if( oz0 > 1 ) {
                inputs[0]->grad += this->grad * inputs[1]->output;
            } else {
                auto tmp = this->grad * inputs[1]->output;
                inputs[0]->grad += xt::sum( tmp );
            }*/
            if( bc.sum_type[0] > 0 ) {
                Tensor tmp = this->grad * inputs[1]->output;
                Tensor ga = broadcast_sum( tmp, bc.sum_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad * inputs[1]->output;
            }
        }
        if( inputs[1]->is_grad() )
        {
            /*int oz1 = inputs[1]->output.size();  // 220208 mod
            if( oz1 > 1 ) {
                inputs[1]->grad += this->grad * inputs[0]->output;
            } else {
                auto tmp = this->grad * inputs[0]->output;
                inputs[1]->grad += xt::sum( tmp );
            }*/
            if( bc.sum_type[1] > 0 ) {
                Tensor tmp = this->grad * inputs[0]->output;
                Tensor ga = broadcast_sum( tmp, bc.sum_type[1], bc.shape.size() );
                ga.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += ga;
            } else {
                inputs[1]->grad += this->grad * inputs[0]->output;
            }
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "mul" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "mul grad a", 0 );
        check_grad_shape1( "mul grad b", 1 );
    }
};

class DivOp : public ArithBase{ // MCTNode{
    public:
    DivOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "div(forward)" );
        bc = broadcast_check( inputs[0]->output, inputs[1]->output ); // 220208 add
        output = inputs[0]->output / inputs[1]->output;
        return true;
    }
    bool backward()
    {
        print_message( "div(backward)" );
        auto& x0 = inputs[0]->output;
        auto& x1 = inputs[1]->output;
        if( inputs[0]->is_grad() )
        {
            /*int oz0 = x0.size();  // 220208 mod
            if( oz0 > 1 ) {
                inputs[0]->grad += this->grad / x1;
            } else {
                auto tmp = this->grad / x1;
                inputs[0]->grad += xt::sum( tmp );
            }*/
            if( bc.sum_type[0] > 0 ) {
                Tensor tmp = this->grad / x1;
                Tensor ga = broadcast_sum( tmp, bc.sum_type[0], bc.shape.size() );
                ga.reshape( inputs[0]->output.shape() );
                inputs[0]->grad += ga;
            } else {
                inputs[0]->grad += this->grad / x1;
            }
        }
        if( inputs[1]->is_grad() )
        {
            /*int oz1 = x1.size();  // 220208 mod
            if( oz1 > 1 ) {
                inputs[1]->grad += this->grad * ( -x0 / (x1*x1) );
            } else {
                auto tmp = this->grad * ( -x0 / (x1*x1) );
                inputs[1]->grad += xt::sum( tmp );
            }*/
             if( bc.sum_type[1] > 0 ) {
                Tensor tmp = this->grad * ( -x0 / (x1*x1) );
                Tensor gb = broadcast_sum( tmp, bc.sum_type[1], bc.shape.size() );
                gb.reshape( inputs[1]->output.shape() );
                inputs[1]->grad += gb;
            } else {
                inputs[1]->grad += this->grad * ( -x0 / (x1*x1) );
            }
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "div" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "div grad a", 0 );
        check_grad_shape1( "div grad b", 1 );
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
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad * output;
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
        if( inputs[0]->is_grad() ) 
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
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad / ( inputs[0]->output + 1.0 );
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
        output = xt::pow( inputs[0]->output, inputs[1]->output );
        return true;
    }
    bool backward()
    {
        print_message( "pow(backward)" );
        auto& x = inputs[0]->output;
        fprec c = (fprec)inputs[1]->output[0];
        if( inputs[0]->is_grad() ) 
            inputs[0]->grad += this->grad * c * xt::pow( x, c-1.0 );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "pow" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "pow grad", 0 );
    }
};

class MatMulBase:public MCTNode
{
    public:
    MatMulBase(){}
    
    bool forward()  { return true; }
    bool backward() { return true; }
    
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
        int  b = 1.;
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
        Tensor::shape_type new_s( ar );
        for(int i=0;i<ar-2;i++)  new_s[i] = as[i];
        new_s[ar-2] = bs[ar-2];
        new_s[ar-1] = bs[ar-1];
        return new_s;
    }
    
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
            //  a.rank: 2,  b.rank: 2 
            output = DOT(a,b);
            //output =xt::linalg::dot(a,b);
        } else if( ar > 2 && br==2 ) {
            // batched matmul:
            //  a.rank: >2,  b.rank: 2 
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
            //  a.rank: 2,  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            gb += DOT( xt::transpose(a), gc );
            
        } else if( ar==1 && br==2 ){
            //  a.rank: 1,  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            
        } else if( ar > 2 && br==2 ){
            // batched matmul:
            //  a.rank: >2,  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            //print_tensor( "ga", ga );
            //print_tensor( "gc", gc );
            gb += _batch_grad( a, gc, as );
            
        } else {
            cout<<"Error:MatMulOp"<<endl;
            cout<<"Error:A:"<<ar<<" B:"<< br<<endl;
            return false;
        }
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "matmul" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "matmul grad a", 0 );
        check_grad_shape1( "matmul grad b", 1 );
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
        //cout<<"linear a"<<a<<endl;
        
        if(( ar==1 || ar==2 ) && br==2 ) { 
            //  a.rank: 2,  b.rank: 2 
            output = DOT(a, xt::transpose(b) ) + d;
            
        } else if( ar > 2 && br==2 ){  // yet 
            //  a.rank: >2,  b.rank: 2 
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
            //  a.rank: 2,  b.rank: 2 
            ga += DOT( gc, b );
            gb += DOT( xt::transpose(gc), a );
            gd += xt::sum( gc,{0} );
            
        } else if( ar==1 && br==2 ){ // yet
            //  a.rank: 1,  b.rank: 2
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} );
            
        } else if( ar > 2 && br==2 ){
            //  a.rank: >2,  b.rank: 2 
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
    int check_shape()
    {
        return check_shape1( "linear" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "linear grad x", 0 );
        check_grad_shape1( "linear grad w", 1 );
        check_grad_shape1( "linear grad b", 2 );
    }
    
    /* 220120 del
    void update( fprec delta )
    {
        auto& b  = inputs[1]->output;  // weight
        auto& d  = inputs[2]->output;  // bias
        auto& gb = inputs[1]->grad;
        auto& gd = inputs[2]->grad;
        
        b = b - delta * gb;
        d = d - delta * gd;
    }*/
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
            //  a.rank: 2,  b.rank: 2 
            output = DOT(a,b) + d;
            
        } else if( ar > 2 && br==2 ){
            //  a.rank: >2,  b.rank: 2 
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
            //  a.rank: 2,  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            gb += DOT( xt::transpose(a), gc );
            gd += xt::sum( gc,{0} );
            
        } else if( ar==1 && br==2 ) {
            //  a.rank: 1,  b.rank: 2 
            ga += DOT( gc, xt::transpose(b) );
            Tensor a2 = a.reshape({-1,1});
            Tensor g2 = gc.reshape({1,-1});
            gb += DOT( a2, g2 );
            gd += xt::sum( gc,{0} );
            
        } else if( ar > 2 && br==2 ) {
            //  a.rank: >2,  b.rank: 2 
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
    int check_shape()
    {
        return check_shape1( "addmm" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "addmm grad b", 0 );
        check_grad_shape1( "addmm grad x", 1 );
        check_grad_shape1( "addmm grad w", 2 );
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
    int check_shape()
    {
        return check_shape1( "transpose" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "transpose grad", 0 );
    }
};

class MaxOp:public MCTNode{
    public:
    MaxOp( int ax=1 ) { axis = ax; }
    int axis;
    xt::xarray<bool> cond;
    
    bool forward()
    {
        _forward_inputs();
        print_message( "max(forward)" );
        axis = (int)inputs[1]->output[0];
        output = xt::amax( inputs[0]->output, {axis} );
        return true;
    }
    bool backward()
    {
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
    int check_shape()
    {
        return check_shape1( "max" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "max grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "min" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "min grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "sigmoid" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "sigmoid grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "relu" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "relu grad", 0 );
    }
};

class HardTanhOp:public MCTNode{
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
        _forward_inputs();
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
        auto& y  = inputs[0]->output;
        auto& gd = inputs[0]->grad;
        gd += this->grad * ( y > min_val && y < max_val );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "hardtanh" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "hardtanh grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "elu" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "elu grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "leakyrelu" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "leakyrelu grad", 0 );
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
        mask = ( output * beta < threshold );
        auto m = xt::masked_view( output, mask );
        m = xt::log( 1.0 + xt::exp( beta * output ) ) / beta;
        return true;
    }
    bool backward()
    {
        print_message( "softplus(backward)" );
        auto& a  = inputs[0]->output;
        fprec beta      = (fprec)inputs[1]->output[0];
        fprec threshold = (fprec)inputs[2]->output[0];
        auto& ga = inputs[0]->grad;
        ga = this->grad;
        mask = ( a * beta < threshold );
        auto m = xt::masked_view( ga, mask );
        m = 1.0/( 1.0 + 1.0/xt::exp( beta * a ) );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "softplus" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "softplus grad", 0 );
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
            //return xt::expand_dims( b, {1} );
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
        Tensor y = se / sd;
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
        Tensor ga = output * this->grad;
        Tensor sg = xt::sum( ga, {axis} );
        sg = _row2col( sg, as );
        inputs[0]->grad += ( ga - output * sg );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "softmax" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "softmax grad", 0 );
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
        auto    gs = ga.shape();
        Tensor  sg = xt::sum( ga, {axis} );
        sg = _row2col( sg, gs );
        Tensor  se = xt::exp( output );
        inputs[0]->grad += ( ga - se * sg );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "logsoftmax" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "logsoftmax grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "tanh" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "tanh grad", 0 );
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
    int check_shape()
    {
        return check_shape1( "fulllike" );
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
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist ) // 220107 mod
            {
                std::vector<size_t> sh;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                output = xt::zeros<fprec>( sh );
                return true;
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                std::vector<size_t> sh;
                for(int k=0;k<as.size();k++)
                {
                    sh.push_back( as[k]);
                }
                output = xt::zeros<fprec>( sh );
                return true;
            }
        }
        return false;
    }
    bool backward()
    {
        //print_message( "zeros(backward)" );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "zeros" );
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
            std::vector<Tensor>* tlist = inputs[0]->get_tlist();
            if( tlist )
            {
                std::vector<size_t> sh;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                output = xt::ones<fprec>( sh );
                return true;
            } else {
                auto& a  = inputs[0]->output;
                auto  as = a.shape();
                std::vector<size_t> sh;
                for(int k=0;k<as.size();k++)
                {
                    sh.push_back( as[k] );
                }
                output = xt::ones<fprec>( sh );
                return true;
            }
        }
        return false;
    }
    bool backward(){
        //print_message( "ones(backward)" );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "ones" );
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
                std::vector<size_t> sh;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
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
        //print_message( "randn(backward)" );
        _backward_inputs();
        return true;
    }int check_shape()
    {
        return check_shape1( "randn" );
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
                auto ms = mean.shape();
                /*
                int  dim = ms.size();
                int  mz  = mean.size();
                if( dim > 1 ) 
                {
                    mean.reshape( {-1} );
                    std.reshape( {-1} );
                }
                Tensor out = xt::zeros<fprec>( {mz} );
                for(int i=0;i<mz;i++)
                {
                    Tensor o = xt::random::randn<fprec>( {1}, (fprec)mean[i], (fprec)std[i] );
                    out(i) = (fprec)o(0);
                }
                if( dim > 1 )
                {
                    out.reshape( ms );
                }
                output = out;
                // 220120 mod until this lines
                */
                
                // ---220208 mod
                Tensor tmp = xt::random::randn<fprec>( ms );
                output = tmp * std + mean;
                // --- 220208 mod 
                //print_tensor( "randn1", output );
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
                        std::vector<size_t> sh; // 220110 mod
                        for(int k=0;k<tlist->size();k++)
                        {
                            auto t1 = tlist->at(k);
                            sh.push_back( (int)t1[0] );
                        }
                        output = xt::random::randn<fprec>( sh, mean, std );
                        //print_tensor( "randn2", output );
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
    int check_shape()
    {
        return check_shape1( "normal" );
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
        
        // 220120 mod
        inputs[3]->set_ntype( 4 );  // running_mean
        inputs[4]->set_ntype( 4 );  // running_var
        
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
    
    int check_shape()
    {
        return check_shape1( "batchnorm" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "batchnorm grad x", 0 );
        check_grad_shape1( "batchnorm grad gamma", 1 );
        check_grad_shape1( "batchnorm grad beta", 2 );
    }
    /* 220120 del
    void update( fprec delta )
    {
        auto& g  = inputs[1]->output;  // gamma
        auto& b  = inputs[2]->output;  // beta
        auto& gm = inputs[1]->grad;
        auto& gb = inputs[2]->grad;
        
        g = g - delta * gm;
        b = b - delta * gb;
    }*/
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
    int check_shape()
    {
        return check_shape1( "dropout" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "dropout grad", 0 );
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

class NLLLossOp:public MCTNode{
    public:
    NLLLossOp(){}
    
    bool forward() 
    {
        _forward_inputs();
        print_message( "nullload(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        fprec h  = (fprec)inputs[4]->output[0];
        auto& b  = inputs[1]->output;
        
        fprec d = fprec( as[0] );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        
        xt::xarray<float> t = xt::zeros<float>( {as[0]} );
        for(int i=0;i<as[0];i++)
        {
            int j = (int)inputs[1]->output[i];
            t[i] = a( i, j );
            if( t[i] < h )  t[i] = h;
        }
        output = -xt::sum(t) / d;  // 220120 mod (fprec)as[0]->d
        print_tensor( "nllloss", output );
        return true;
    }
    bool backward()
    {
        print_message( "nullload(backward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
        
        fprec d = fprec( as[0] );
        if( inputs[3] )
        {
            int reduction = (int)inputs[3]->output[0];
            if( reduction == 2 )  d = 1.0; // sum
        }
        fprec sc = 1.0 / d; // 220120 mod (fprec)as[0]->d;
        
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
    
    /*
    void eval_shape( Tensor::shape_type as, vector<unsigned int>&na, int n_dim )  // 220202 rename
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
    }*/
    
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
        eval_shape( as[0], na, n_dim );
        if( chk > 0 )
        {
            cout<<"--------------------------"<<endl;
            cout<<"broadcast n_dim="<<n_dim<<endl;
            print_ints( "broadcast tensor-1 ", na, n_dim );
        }
        
        for(int k=1;k<num;k++)
        {
            eval_shape( as[k], nb, n_dim );
            
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
    int check_shape()
    {
        return check_shape1( "broadcast_tensors" );
    }
    int check_grad_shape()
    {
        if( inputs[0] )
        {
            std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
            std::vector<Tensor>* glist1 = inputs[0]->get_glist();
            if( tlist1 && glist1 )
            {
                for(int i=0;i<tlist1->size();i++)
                {
                    auto& a1 = tlist1->at(i);
                    auto& g1 = glist1->at(i);
                    string s = "broadcast_tensors_" + std::to_string(i) + " grad";
                    check_grad_shape_size( s, id, a1, g1 );
                }
            }
        }
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
        if( inputs.size() < 1 ) // 220106 mod 2->1
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
    int check_shape()
    {
        return check_shape1( "ListUnpack" );
    }
    int check_grad_shape()
    {
        std::vector<Tensor>* tlist1 = inputs[0]->get_tlist();
        std::vector<Tensor>* glist1 = inputs[0]->get_glist();
        if( tlist1 && glist1 )
        {
            int sz = glist1->size();
            if( out_id < sz )
            {
                auto& a1 = tlist1->at(out_id);
                auto& g1 = glist1->at(out_id);
                string s = "ListUnpack_" + std::to_string(out_id) + " grad";
                check_grad_shape_size( s, id, a1, g1 );
            }
        }
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

class SizeOp:public MCTNode{
    public:
    SizeOp(){}
    
    bool forward()
    {
        _forward_inputs();
        print_message( "size(forward)" );
        auto& a  = inputs[0]->output;
        auto  as = a.shape();
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
        
        if( inputs[1] )  // shape
        {
            std::vector<Tensor>* tlist = inputs[1]->get_tlist();
            if( tlist )
            {
                std::vector<size_t> sh;
                for(int k=0;k<tlist->size();k++)
                {
                    auto t1 = tlist->at(k);
                    sh.push_back( (int)t1[0] );
                }
                auto& a = inputs[0]->output;
                output = xt::broadcast( a, sh );
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
    int check_shape()
    {
        return check_shape1( "expand" );
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
                std:vector<int> sh;
                for(int i=0;i<tlist1->size();i++)
                {
                    auto& t = tlist1->at(i);
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
        Tensor& gc = this->grad;
        inputs[0]->grad = gc.reshape( org_shape );
        _backward_inputs();
        return true;
    }
    int check_shape()
    {
        return check_shape1( "view" );
    }
    int check_grad_shape()
    {
        check_grad_shape1( "view grad", 0 );
    }
};

class ToOp:public MCTNode{
    public:
    ToOp() {}
    
    bool forward()
    {
        _forward_inputs();
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

