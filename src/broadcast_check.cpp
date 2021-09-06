
#include <stdio.h>
#include <iostream>
#include <fstream>
#include"xtensor/xarray.hpp"
#include"xtensor/xio.hpp"
#include"xtensor/xrandom.hpp"
#include"xtensor/xbroadcast.hpp"
#include <xtensor/xlayout.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xview.hpp>

//#include"cblas.h"

using namespace std;

#define fprec float

typedef xt::xarray<fprec> Tensor;



int* check_shape( Tensor::shape_type as, int n_dim )
{
    int  az = as.size();
    int* na = new int[ n_dim ];
    for(int i=0;i<n_dim;i++)  na[i] = 1;
    int inc = n_dim - az;
    for(int i=0;i<az;i++)  na[inc+i] = as[i];
    return na;
}

void print_ints( string s, int *v, int nv )
{
    cout<<s;
    for(int i=0;i<nv;i++)  cout<<v[i]<<",";
    cout<<endl;
}

void print_shape1( string s, Tensor a )
{
    auto as=a.shape();
    cout<<s<<"(";
    for(int i=0;i<as.size();i++){
        cout<<as[i]<<",";
    }
    cout<<")"<<endl;
}

void print_shape2( Tensor::shape_type &as )
{
    cout<<"shape (";
    for(int i=0;i<as.size();i++){
        cout<<as[i]<<",";
    }
    cout<<")"<<endl;
}

int check_broadcast( Tensor a, Tensor b, Tensor::shape_type &shape )
{
    /*auto print_shape2=[]( Tensor::shape_type &as ) 
    {
        cout<<"shape (";
        for(int i=0;i<as.size();i++){
            cout<<as[i]<<",";
        }
        cout<<")"<<endl;
    };*/
    
    //Tensor::shape_type shape = {0};
    shape = {0};
    
    auto as = a.shape();
    auto bs = b.shape();
    int  az = as.size();
    int  bz = bs.size();
    
    cout<<"--- brodcat check -----------"<<endl;
    cout<<"a"<<a<<endl;
    cout<<"b"<<b<<endl;
    print_shape1("a shape ",a);
    print_shape1("b shape ",b);
    
    // scalar check
    int ns = 0;
    if( az == 0 )  ns += 1;
    if( bz == 0 )  ns += 1;
    if( ns > 0 ) {
        cout<<"scalar check = "<<ns<<endl;
        cout<<"broadcast check status="<<2<<endl;
        return 2;
    }
    
    // broadcast array dimension
    int n_dim = ( az > bz ) ? az : bz;
    cout<<"n_dim="<<n_dim<<endl;
    int* na = check_shape( as, n_dim );
    int* nb = check_shape( bs, n_dim );
    
    // check broadcast
    int ne = 0;
    int n1 = 0;
    int n2 = 0;
    for(int i=0;i<n_dim;i++){
        if( na[i] == 1 ){
            n1 += 1;
        } else if( nb[i] == 1 ){
            n1 += 1;
        } else if( na[i] == nb[i] ){
            ne += 1;
        } else {
            //cout<<"broadcast mismatch dimension no."<<i<<endl;
            n2 += 1;
        }
    }
    print_ints("na ",na,n_dim);
    print_ints("nb ",nb,n_dim);
    cout<<"equal="<<ne<<" eq1="<<n1<<" other="<<n2<<endl;
    
    int status = 0;
    if( ne == n_dim ) {  
        status = 1;
    } else if( (ne+n1) == n_dim ) {
        status = 1;
    } else if( ((n1+n2) == n_dim) & (n2==1) ) {
        status = 1;
    } else {
        cout<<"broadcast error. shapes=";  // yet
        print_shape1("a", a);
        print_shape1("b", b);
    }
    
    if( status == 1 )
    {
        unsigned int* nc = new unsigned int [n_dim];
        
        for(int i=0;i<n_dim;i++)
        {
            nc[i] = (na[i]>=nb[i]) ? na[i] : nb[i];
        }
        //Tensor::shape_type shape;
        if( n_dim == 2 ) {
            shape = { nc[0], nc[1] };
            cout<<"shape "<<nc[0]<<","<<nc[1]<<endl;
        } else if( n_dim == 3 ) {
            shape = { nc[0], nc[1], nc[2] };
            cout<<"shape "<<nc[0]<<","<<nc[1]<<","<<nc[2]<<endl;
        } else if( n_dim == 4 ) {
            shape = { nc[0], nc[1], nc[2], nc[3] };
            cout<<"shape "<<nc[0]<<","<<nc[1]<<","<<nc[2]<<","<<nc[3]<<endl;
        } else {
            cout<<"dimension over 4"<<endl;
            return 0;
        }
       
       /*
        auto a2 = xt::broadcast( a, shape );
        auto b2 = xt::broadcast( b, shape );
        cout<<"a-new"<<a2<<endl;
        cout<<"b-new"<<b2<<endl;
        
        auto c = a+b;
        cout<<"a+b"<<c<<endl;
        */
        delete [] nc;
    }
    
    delete [] na;
    delete [] nb;
      
    print_shape2( shape );
    cout<<"broadcast check status="<<status<<endl;
    return status;
};

void try_broadcast( int type, Tensor &a, Tensor &b, Tensor::shape_type &shape ) 
{
    cout<<"try type="<<type<<endl;
    if( type == 1 ) {
        auto a2 = xt::broadcast(a,shape);
        auto b2 = xt::broadcast(b,shape);
        auto z = a2 + b2;
        cout<<"a+b "<<z<<endl;
    } else if( type == 2 ) {
        auto z = a+b;
        cout<<"a+b "<<z<<endl;
    } else {
    }
}

void _broadcast()
{
    Tensor a  = xt::zeros<fprec>({3, 3});
    Tensor b  = xt::arange(3);
    Tensor c  = xt::view(b,xt::all(), xt::newaxis() );
    Tensor d  = xt::arange(4);
    Tensor e  = (fprec)1.0;
    
    cout<<"a"<<a<<endl;
    cout<<"b"<<b<<endl;
    cout<<"c"<<c<<endl;
    
    Tensor::shape_type shape = {0};
    
    int g1 = check_broadcast(a,b,shape);
    //cout<<"--- g1: "<<g1<<","<<shape<<endl;
    try_broadcast( g1, a, b, shape );
    
    int g2 = check_broadcast(a,c,shape);
    try_broadcast( g2, a, c, shape );
    //cout<<"--- g2: "<<g2<<endl;
    
    int g3 =check_broadcast(d,c,shape);
    try_broadcast( g3, d, c, shape );
    //cout<<"--- g3: "<<g3<<endl;
    
    Tensor a5 = xt::zeros<fprec>( {4, 3} );
    cout<<"a5"<<a5<<endl;
    
    Tensor b5 = xt::arange(6).reshape( {2, 3} );
    cout<<"b5"<<b5<<endl;
    
    int g4 =check_broadcast(a5,b5,shape);
    try_broadcast( g4, a5, b5, shape );
    //cout<<"--- g4: "<<g4<<endl;
    
    int g5 =check_broadcast(a,e,shape);
    try_broadcast( g5, a, e, shape );
    
    auto e2 = xt::broadcast(e,{3,3});
    cout<<"e2"<<e2<<endl;
    
    vector<Tensor> tlist;
    tlist.push_back(a);
    tlist.push_back(b);
    tlist.push_back(c);
    
    int k=1;
    for(auto& itr:tlist){
        cout<<"tlist"<<k<<itr<<endl;
        k++;
    }
}

void main12()
{
    Tensor a  = {{ 1., 2., 3., 4., 5.}, 
                 { 6., 7., 8., 9.,10.},
                 {11.,12.,13.,14.,15.},
                 {21.,22.,23.,24.,25.},
                 {31.,32.,33.,34.,35.}};
    Tensor b  = xt::arange(5);
    xt::random::shuffle(b);
    cout<<"a"<<a<<endl;
    cout<<"b"<<b<<endl;
    //cout<<"c"<<c<<endl;
    
    Tensor a1 = xt::zeros<fprec>( {5,5} );
    //Tensor::shape_type a1_s = { 5,5 };
    //Tensor a1(a1_s);
    //auto a1_s = a1.shape();
    //cout<<"a1_s "<<a1_s[0]<<","<<a1_s[1]<<endl;
    //cout<<"a1"<<a1.data<<endl;
    
    for(int i=0;i<5;i++)  
    {
        //auto aaa = xt::row(a,b(i));
        //cout<<"aaa"<<aaa<<endl;
        //auto a_s = aaa.shape();
        //cout<<"a_s "<<a_s[0]<<","<<a_s[1]<<endl;
        //auto a1_s = a1.shape();
        //cout<<"a1_s "<<a1_s[0]<<","<<a1_s[1]<<endl;
        //xt::row(a1,i) = xt::row(a,b1(i));  // broadcaast error
        xt::row(a1,i) = xt::flatten( xt::row(a,b(i)) );
        //for(int j=0;j<5;j++)  a1(i,j)=a(b(i),j);
    }
    cout<<"a1"<<a1<<endl;
}

int main( int argc, char *argv[] )
{
   for (int i=1;i<argc;i+=2)
      printf("argv[%d] = %s %s\n", i, argv[i],argv[i+1]);
      
   _broadcast();
   
   return 1;
}