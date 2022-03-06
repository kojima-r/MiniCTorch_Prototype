#include <gtest/gtest.h>
#include "minictorch.hpp"


inline void expect_eq1( string s, Tensor& z, Tensor& q )
{
    //cout<<"z"<<z<<endl;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<s<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
}

inline void expect_eqc( string s, Tensor& z, fprec c )
{
    //cout<<s<<z<<endl;
    
    auto itr1 = z.begin();
    while( itr1 != z.end() ){
        EXPECT_EQ( *itr1++, c );
    }
}

// ----- add -----

Tensor calc_add( Tensor& a, Tensor b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    AddOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );

    op1.forward();
    return op1.output;
}

Tensor diff_add( Tensor& a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    AddOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );

    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestAdd) 
{
    Tensor a = {{1., 2.}, {3., 4.}};
    Tensor b = {{5., 6.}, {7., 8.}};
    Tensor q = {{6., 8.}, {10., 12.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    AddOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
    
    expect_eq1( "add ", op1.output, q );
}

TEST(MyTestCase, TestAddGrad) 
{
    Tensor a = {{1., 2.}, {3., 4.}};
    Tensor b = {{5., 6.}, {7., 8.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    AddOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eqc( "add grad_a ", va.grad, 1.0 );
    expect_eqc( "add grad_b ", vb.grad, 1.0 );
}

// ----- sub -----

TEST(MyTestCase, TestSub) 
{
    Tensor a = {{5., 6.}, {3., 4.}};
    Tensor b = {{1., 2.}, {7., 8.}};
    Tensor q = {{4., 4.}, {-4.,-4.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    SubOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
    
    expect_eq1( "sub ", op1.output, q );
}

TEST(MyTestCase, TestSubGrad) 
{
    Tensor a = {{5., 6.}, {3., 4.}};
    Tensor b = {{1., 2.}, {7., 8.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    SubOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eqc( "sub grad_a ", va.grad,  1.0 );
    expect_eqc( "sub grad_a ", vb.grad, -1.0 );
}

// ----- Mul -----

TEST(MyTestCase, TestMul) 
{
    Tensor a = {{1., 2.}, {3., -4.}};
    Tensor b = {{1., 2.}, {5.,  6.}};
    Tensor q = {{1., 4.}, {15.,-24.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    MulOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
   
    expect_eq1( "mul ", op1.output, q );
}

TEST(MyTestCase, TestMulGrad) 
{
    Tensor a = {{1., 2.}, {3., -4.}};
    Tensor b = {{1., 2.}, {5.,  6.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    MulOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    auto& ga = va.grad;
    auto& gb = vb.grad;
    expect_eq1( "mul grad ", ga, b );
    expect_eq1( "mul grad ", gb, a );
}

// ----- Div -----

TEST(MyTestCase, TestDiv) 
{
    Tensor a = {{1., 4.}, {2.5, -3.}};
    Tensor b = {{1., 2.}, {5.0,  6.}};
    Tensor q = {{1., 2.}, {0.5,-0.5}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    DivOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
    
    expect_eq1( "div ", op1.output, q );
}

TEST(MyTestCase, TestDivGrad) 
{
    Tensor a = {{1., 4.}, {2.5, -3.}};
    Tensor b = {{1., 2.}, {5.0,  6.}};
    
    Tensor qa = xt::zeros_like( a );
    Tensor qb = xt::zeros_like( b );
    
    auto itr1  = a.begin();
    auto itr2  = b.begin();
    auto itr11 = qa.begin();
    auto itr12 = qb.begin();
    while( itr1 != a.end() )
    {
        *itr11 = 1.0 / (*itr2);
        *itr12 = -(*itr1) / ( (*itr2) * (*itr2) );
        //cout<<*itr1<<","<<*itr2<<"-"<<*itr11<<","<<*itr12<<endl;
        itr1++;  itr2++; 
        itr11++; itr12++;
    }
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    DivOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eq1( "div grad ", va.grad, qa );
    expect_eq1( "div grad ", vb.grad, qb );
}

// ----- Rsub -----

TEST(MyTestCase, TestRsub) 
{
    Tensor a = { 0.2, 0.5, 0.7 };
    fprec  b = 1.0;
    fprec  c = 1.0;
    Tensor q = { 0.8, 0.5, 0.3 };
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(c);
    
    RsubOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    op1.forward();
   
    expect_eq1( "rsub ", op1.output, q );
}

TEST(MyTestCase, TestRsubGrad) 
{
    Tensor a = { 0.2, 0.5, 0.7 };
    fprec  b = 1.0; // constant
    fprec  c = 1.0; // constant
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(c);
    
    RsubOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eqc( "rsub grad ", va.grad, -c );
}

// ----- Neg -----

TEST(MyTestCase, TestNeg ) 
{
    Tensor a = { 1.0, 3.0, -2.0, 7.0 };
    Tensor q = -a;
    
    VariableTensor va(a);
    
    NegOp  op1;
    op1.set_inputs( &va );
    op1.forward();
  
    expect_eq1( "neg ", op1.output, q );
}

TEST(MyTestCase, TestNegGrad) 
{
    Tensor a = { 1.0, 3.0, -2.0, 7.0 };
    
    VariableTensor va(a);
    
    NegOp  op1;
    op1.set_inputs( &va );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eqc( "neg grad ", va.grad, -1.0 );
}

// ----- Pow -----

TEST(MyTestCase, TestPow) 
{
    Tensor a = {{ 1.,-2., 3.},{ 2., -3., -4.}};
    fprec  b = 2.0;  // constant
    Tensor q = {{ 1., 4., 9.},{ 4.,  9., 16.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    PowOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
   
    expect_eq1( "pow ", op1.output, q );
}

TEST(MyTestCase, TestPowGrad) 
{
    Tensor a = {{ 1.,-2., 3.},{ 2., -3., -4.}};
    fprec  b = 2.0;  // constant
    
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = b * pow( *itr1, b-1.0 );
        //cout<<"pow"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    PowOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eq1( "mul ", va.grad, q );
}

// ----- Exp -----

TEST(MyTestCase, TestExp) 
{
    Tensor a = {{ 1.,-2., 3.},{ 2., -3., 0.5}};
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = exp( *itr1 );
        //cout<<"exp"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    ExpOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    
    expect_eq1( "exp", op1.output, q );
}

TEST(MyTestCase, TestExpGrad) 
{
    Tensor a = {{ 1.,-2., 3.},{ 2., -3., 0.5}};
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = exp( *itr1 );
        //cout<<"exp"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    ExpOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    
    op1.grad = xt::ones_like( a );
    op1.backward();
  
    expect_eq1( "exp grad", va.grad, q );
}

// ----- Log -----

TEST(MyTestCase, TestLog) 
{
    Tensor a = { 1., 2., 3., 0.5 };
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = log( *itr1 );
        //cout<<"log"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    LogOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    
    expect_eq1( "log", op1.output, q );
}

TEST(MyTestCase, TestLogGrad) 
{
    Tensor a = { 1., 2., 3., 0.5 };
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = 1.0 / *itr1;
        //cout<<"log"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    LogOp  op1;
    op1.set_inputs( &va );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eq1( "log grad ", va.grad, q );
}

// ----- Log1p -----

TEST(MyTestCase, TestLog1p) 
{
    Tensor a = { 1., 2., 3., 0.5 };
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = log( *itr1 + 1.0 );
        //cout<<"log"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    Log1pOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    
    expect_eq1( "log1p ", op1.output, q );
}

TEST(MyTestCase, TestLog1pGrad) 
{
    Tensor a = { 1., 2., 3., 0.5 };
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = 1.0/ ( *itr1 + 1.0 );
        //cout<<"log"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    Log1pOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    expect_eq1( "log1p grad", va.grad, q );
}

// ----- Sum -----

TEST(MyTestCase, TestSum) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = 18.;
    
    VariableTensor va(a);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( NULL );
    op1.forward();
    
    expect_eq1( "sum ", op1.output, q );
}

TEST(MyTestCase, TestSum0) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = { 4., 6., 8.};
    
    VariableTensor va(a);
    VariableTensor vb(0.0);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    expect_eq1( "sum0 ", op1.output, q );
}

TEST(MyTestCase, TestSum1) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = { 6., 12. };
    q.reshape( {2,1} );
    
    VariableTensor va(a);
    VariableTensor vb(1.0);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    expect_eq1( "sum1 ", op1.output, q );
}

TEST(MyTestCase, TestSumGrad) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = {18.};
    Tensor qa = {{ 1., 1., 1. },{ 1., 1., 1. }};
    
    VariableTensor va(a);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( NULL );
    op1.forward();
    
    op1.grad = xt::ones_like( q );
    op1.backward();
    
    expect_eq1( "sum grad ", va.grad, qa );
}

TEST(MyTestCase, TestSum0Grad) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = { 4., 6., 8. };
    Tensor qa = {{ 1., 1., 1. },{ 1., 1., 1. }};
    
    VariableTensor va(a);
    VariableTensor vb(0.0);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    op1.grad = xt::ones_like( q );
    op1.backward();
    
    expect_eq1( "sum0 grad", va.grad, qa );
}

TEST(MyTestCase, TestSum1Grad) 
{
    Tensor a  = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q  = { 6.,12.};
    Tensor qa = {{ 1., 1., 1. },{ 1., 1., 1. }};
    
    VariableTensor va(a);
    VariableTensor vb(1.0);
    
    SumOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    op1.grad = xt::ones_like( q );
    op1.backward();
    
    expect_eq1( "sum1 grad ", va.grad, qa );
}

// ----- mean -----

TEST(MyTestCase, TestMean) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = { 2., 3., 4. };
    
    VariableTensor va(a);
    VariableTensor vb(0.0);
    
    MeanOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    expect_eq1( "mean ", op1.output, q );
}

TEST(MyTestCase, TestMeanGrad) 
{
    Tensor a = {{ 1., 2., 3. },{ 3., 4., 5. }};
    Tensor q = { 2., 3., 4. };
    Tensor qa = {{ 0.5, 0.5, 0.5 },{ 0.5, 0.5, 0.5 }};
    
    VariableTensor va(a);
    VariableTensor vb(0.0);
    
    MeanOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    //op1.forward();
    
    op1.grad = xt::ones_like( q );
    op1.backward();
    
    expect_eq1( "mean grad", va.grad, qa );
}

// ----- randn, normal, zeros, ones, size -----

inline void expect_rand( string s, Tensor& z, fprec c )
{
    fprec sm = 0;
    auto itr1 = z.begin();
    while( itr1 != z.end() ){
        //cout<<s<<*itr1<<endl;
        EXPECT_GE( *itr1, -c );
        EXPECT_LE( *itr1,  c );
        sm += *itr1;
        itr1++;
    }
}

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

    expect_rand( "randn : ", z, 3.0 );
    
    auto mean = xt::mean( z );
    auto std  = xt::stddev( z );
    cout<<"mean"<<mean<<endl;
    cout<<"std "<<std<<endl;
}

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
    Tensor a = xt::zeros<fprec>({2, 4});
    auto   z = do_normal( a );
    //cout<<"normal "<<z<<endl;
    
    expect_rand( "normal : ", z, 3.0 );
    
    auto mean = xt::mean( z );
    auto std  = xt::stddev( z );
    cout<<"mean"<<mean<<endl;
    cout<<"std "<<std<<endl;
}

