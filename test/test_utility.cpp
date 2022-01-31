#include <gtest/gtest.h>
#include "minictorch.hpp"


// ----- ListConstruc/ListUnpack/Broadcast_tensors -----

Tensor do_broadcast( Tensor &a, Tensor &b, int out=0 )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    ListConstructOp  list1;
    list1.set_inputs( &va );
    list1.set_inputs( &vb );
    
    BroadcastTensorsOp  bcast1;
    bcast1.set_inputs( &list1 );
    
    ListUnpackOp unpk0( 0 );
    unpk0.set_inputs( &bcast1 );
    ListUnpackOp unpk1( 1 );
    unpk1.set_inputs( &bcast1 );
    
    AddOp add1;
    add1.set_inputs( &unpk0 );
    add1.set_inputs( &unpk1 );
    add1.set_inputs( NULL );
    
    if( out > 0 ) 
    {
        cout<<"--------------------------------"<<endl;
        cout<<"a"<<a<<endl;
        cout<<"b"<<b<<endl;
    }
    
    list1.forward();
    if( bcast1.forward() )
    {
        unpk0.forward();
        unpk1.forward();
        add1.forward();
        
        if( out > 0 )
        {
            cout<<"a_mod"<<unpk0.output<<endl;
            cout<<"b_mod"<<unpk1.output<<endl;
            cout<<"a+b"<<add1.output<<endl;
        }
        
        return add1.output;
    }
    cout<<"broadcast error"<<endl;
    
    Tensor z = { -999., };
    return z;
}

TEST(MyTestCase, TestBroadcast1) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor q = {{ 0.,  1.,  2.},
                { 0.,  1.,  2.},
                { 0.,  1.,  2.}};
    
    auto z = do_broadcast( a, b );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<"broadcast"<<*itr1<<","<<*itr2<<endl;
        EXPECT_LE( *itr1++, *itr2++ );
    }
}

TEST(MyTestCase, TestBroadcast2) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor c = xt::view( b, xt::all(), xt::newaxis() );
    Tensor q = {{ 0.,  0.,  0.},
                { 1.,  1.,  1.},
                { 2.,  2.,  2.}};
    
    auto z = do_broadcast( a, c );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<"broadcast"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
}

TEST(MyTestCase, TestBroadcast3) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor c = xt::view( b, xt::all(), xt::newaxis() );
    Tensor d = xt::arange(4);
    Tensor q = {{ 0.,  1.,  2.,  3.},
                { 1.,  2.,  3.,  4.},
                { 2.,  3.,  4.,  5.}};
    
    auto z = do_broadcast( d, c );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<"broadcast"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
}

TEST(MyTestCase, TestBroadcast4) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor e = (fprec)1.0;
    Tensor q = {{ 1.,  1.,  1.},
                { 1.,  1.,  1.},
                { 1.,  1.,  1.}};
    
    auto z = do_broadcast( a, e );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<"broadcast"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
}

// broadcast error
TEST(MyTestCase, TestBroadcast5) 
{
    Tensor a = xt::zeros<fprec>({4, 3});
    Tensor b = xt::arange(6).reshape( {2, 3} );
    
    auto z = do_broadcast( a, b );
    //cout<<"z"<<z<<endl;
    
    auto itr1 = z.begin();
    while( itr1 != z.end() ){
        //cout<<"broadcast5 "<<*itr1<<endl;
        EXPECT_LE( *itr1++, -900. );
    }
}

// ----- randn, normal, zeros, ones, size -----

Tensor do_randn( int n1, int n2 )
{
    VariableTensor va( (fprec)n1 );
    VariableTensor vb( (fprec)n2 );
    
    ListConstructOp  list1;
    list1.set_inputs( &va );
    list1.set_inputs( &vb );
    list1.forward();
    
    VariableTensor vc( 6.0 );
    VariableTensor vd( 0.0 );
    
    RandnOp  randn1;
    randn1.set_inputs( &list1 );
    randn1.set_inputs( &vc );
    randn1.set_inputs( NULL );
    randn1.set_inputs( NULL );
    randn1.set_inputs( &vd );
    randn1.forward();
    
    return randn1.output;
}

TEST(MyTestCase, TestRandn) 
{
    auto z = do_randn( 2, 4 );
    //cout<<"randn"<<z<<endl;
    
    fprec s = 0;
    auto itr1 = z.begin();
    while( itr1 != z.end() ){
        cout<<"randn : "<<*itr1<<endl;
        EXPECT_GE( *itr1, -2.0 );
        EXPECT_LE( *itr1,  2.0 );
        s += *itr1;
        itr1++;
    }
}

//Tensor do_normal( int n1, int n2 )
Tensor do_normal( Tensor& e )
{
    VariableTensor ve( e );
    VariableTensor v0( 0.0 );
    VariableTensor v1( 1.0 );
    
    SizeOp size1;
    size1.set_inputs( &ve );
    size1.set_inputs( &v0 );
    size1.forward();
    
    SizeOp size2;
    size2.set_inputs( &ve );
    size2.set_inputs( &v1 );
    size2.forward();
    
    //cout<<"n1,n2"<<size1.output<<","<<size2.output<<endl;
    
    //VariableTensor va( (fprec)n1 );
    //VariableTensor vb( (fprec)n2 );
    
    ListConstructOp  list1;
    list1.set_inputs( &size1 );
    list1.set_inputs( &size2 );
    list1.forward();
    
    VariableTensor vc( 6.0 );
    VariableTensor vd( 0.0 );
    
    ZerosOp  zeros1;
    zeros1.set_inputs( &list1 );
    zeros1.set_inputs( &vc );
    zeros1.set_inputs( NULL );
    zeros1.set_inputs( NULL );
    zeros1.set_inputs( &vd );
    zeros1.forward();
    
    OnesOp  ones1;
    ones1.set_inputs( &list1 );
    ones1.set_inputs( &vc );
    ones1.set_inputs( NULL );
    ones1.set_inputs( NULL );
    ones1.set_inputs( &vd );
    ones1.forward();
    
    NormalOp  normal1;
    normal1.set_inputs( &zeros1 );
    normal1.set_inputs( &ones1 );
    normal1.set_inputs( NULL );
    normal1.forward();
    
    return normal1.output;
}

TEST(MyTestCase, TestNormal) 
{
    //auto z = do_normal( 2, 4 );
    Tensor a = xt::zeros<fprec>({2, 4});
    auto   z = do_normal( a );
    //cout<<"normal "<<z<<endl;
    
    fprec s = 0;
    auto itr1 = z.begin();
    while( itr1 != z.end() ){
        cout<<"normal : "<<*itr1<<endl;
        EXPECT_GE( *itr1, -2.0 );
        EXPECT_LE( *itr1,  2.0 );
        s += *itr1;
        itr1++;
    }
}

// ----- batch test -----

TEST(MyTestCase, TestBatch1) 
{
    Tensor a={{{1, 2, 3, 4},
               {4, 5, 6, 7},
               {7, 8, 9,10}},
              {{1, 2, 3, 4},
               {4, 5, 6,7},
               {7, 8, 9,9}}};
    Tensor b={{1, 2, 3},
              {4, 5, 6},
              {4, 5, 6},
              {7, 8, 9}};
              
    VariableTensor va(a);
    VariableTensor vb(b);
    MatMulOp op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    //cout<<"matmul1"<<op1.output<<endl;
    auto z = op1.output;
    
    op1.grad = xt::ones_like( op1.output );
    op1.backward();
    
    //cout<<"matmul_ga"<<va.grad<<endl;
    //cout<<"matmul_gb"<<vb.grad<<endl;
    auto ga = va.grad;
    auto gb = vb.grad;
    
    Tensor q ={{{  49.,   59.,   69.},
                {  97.,  119.,  141.},
                { 145.,  179.,  213.}},
               {{  49.,   59.,   69.},
                {  97.,  119.,  141.},
                { 138.,  171.,  204.}}};
    Tensor qa={{{  6.,  15.,  15.,  24.},
                {  6.,  15.,  15.,  24.},
                {  6.,  15.,  15.,  24.}},
               {{  6.,  15.,  15.,  24.},
                {  6.,  15.,  15.,  24.},
                {  6.,  15.,  15.,  24.}}};
    Tensor qb={{{ 12.,  12.,  12.},
                { 15.,  15.,  15.},
                { 18.,  18.,  18.},
                { 21.,  21.,  21.}},
               {{ 12.,  12.,  12.},
                { 15.,  15.,  15.},
                { 18.,  18.,  18.},
                { 20.,  20.,  20.}}};
                
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<"batch2-1"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
        
    auto itr11 = ga.begin();
    auto itr12 = qa.begin();
    while( itr11 != ga.end() ){
        //cout<<"batch2-2"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
        
    auto itr21 = gb.begin();
    auto itr22 = qb.begin();
    while( itr21 != gb.end() ){
        //cout<<"batch2-3"<<*itr21<<","<<*itr22<<endl;
        EXPECT_EQ( *itr21++, *itr22++ );
    }
}