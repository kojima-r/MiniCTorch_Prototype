#include <gtest/gtest.h>
#include "minictorch.hpp"


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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
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
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    
    auto& ga = va.grad;
    auto& gb = vb.grad;
    //cout<<"ga"<<ga<<endl;
    //cout<<"gb"<<gb<<endl;
    
    auto itr11 = ga.begin();
    auto itr12 = gb.begin();
    while( itr11 != ga.end() ){ 
        EXPECT_EQ( *itr11++, 1.0 );
        EXPECT_EQ( *itr12++, 1.0 );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
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
    
    auto& ga = va.grad;
    auto& gb = vb.grad;
    auto itr11 = ga.begin();
    auto itr12 = gb.begin();
    while( itr11 != ga.end() ){ 
        EXPECT_EQ( *itr11++,  1.0 );
        EXPECT_EQ( *itr12++, -1.0 );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
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
    auto itr1 = a.begin();
    auto itr2 = b.begin();
    auto itr11 = ga.begin();
    auto itr12 = gb.begin();
    while( itr11 != ga.end() ){ 
        EXPECT_EQ( *itr11++, *itr2++ );
        EXPECT_EQ( *itr12++, *itr1++ );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestDivGrad) 
{
    Tensor a = {{1., 4.}, {2.5, -3.}};
    Tensor b = {{1., 2.}, {5.0,  6.}};
    
    Tensor qa = xt::zeros_like( a );
    Tensor qb = xt::zeros_like( b );
    
    auto itr1 = a.begin();
    auto itr2 = b.begin();
    auto itr11 = qa.begin();
    auto itr12 = qb.begin();
    while( itr1 != a.end() )
    {
        *itr11 = 1.0 / (*itr2);
        *itr12 = -(*itr1) / ( (*itr2) * (*itr2) );
        //cout<<*itr1<<","<<*itr2<<"-"<<*itr11<<","<<*itr12<<endl;
        itr1++; itr2++; 
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
    auto& ga = va.grad;
    auto& gb = vb.grad;
    
    auto itr21 = qa.begin();
    auto itr22 = qb.begin();
    auto itr31 = ga.begin();
    auto itr32 = gb.begin();
    while( itr21 != qa.end() ){
        //cout<< *itr21<<","<<*itr31<<endl;
        //cout<< *itr22<<","<<*itr32<<endl;
        EXPECT_EQ( *itr21++, *itr31++ );
        EXPECT_EQ( *itr22++, *itr32++ );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
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
    auto& g = va.grad;
    
    auto itr11 = g.begin();
    while( itr11 != g.end() ){
        //cout<<"Rsub "<<*itr11<<","<<-c<<endl;
        EXPECT_EQ( *itr11++, -c );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestNegGrad) 
{
    Tensor a = { 1.0, 3.0, -2.0, 7.0 };
    
    VariableTensor va(a);
    
    NegOp  op1;
    op1.set_inputs( &va );
    
    op1.grad = xt::ones_like( a );
    op1.backward();
    auto& g = va.grad;
    
    auto itr11 = g.begin();
    while( itr11 != g.end() ){
        //cout<<"Neg "<<*itr11<<endl;
        EXPECT_EQ( *itr11++, -1.0 );
    }
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
    auto z = op1.output;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
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
    auto& g = va.grad;
    
    auto itr11 = q.begin();
    auto itr12 = g.begin();
    while( itr11 != q.end() ){
        //cout<<"pow"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto z = op1.output;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto& g = va.grad;
    
    auto itr11 = q.begin();
    auto itr12 = g.begin();
    while( itr11 != q.end() ){
        //cout<<"exp"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto z = op1.output;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
}

TEST(MyTestCase, TestLogGrad) 
{
    Tensor a = { 1., 2., 3., 0.5 };
    Tensor q = xt::zeros_like( a );
    
    auto itr1 = a.begin();
    auto itr2 = q.begin();
    while( itr1 != a.end() )
    {
        *itr2 = 1.0/ *itr1;
        //cout<<"log"<<*itr1<<","<<*itr2<<endl;
        itr1++; itr2++; 
    }
    
    VariableTensor va(a);
    
    LogOp  op1;
    op1.set_inputs( &va );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    auto& g = va.grad;
    
    auto itr11 = q.begin();
    auto itr12 = g.begin();
    while( itr11 != q.end() ){
        //cout<<"log"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto z = op1.output;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto& g = va.grad;
    
    auto itr11 = q.begin();
    auto itr12 = g.begin();
    while( itr11 != q.end() ){
        //cout<<"log"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto z = op1.output;
    //cout<<"sum"<<z<<endl;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto z = op1.output;
    //cout<<"sum0"<<z<<endl;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto z = op1.output;
    //cout<<"sum1"<<z<<endl;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto& g = va.grad;
    //cout<<"sum"<<g<<endl;
    
    auto itr11 = qa.begin();
    auto itr12 = g.begin();
    while( itr11 != qa.end() ){
        //cout<<"sum"<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto& g = va.grad;
    //cout<<"sum0 "<<g<<endl;
    
    auto itr11 = qa.begin();
    auto itr12 = g.begin();
    while( itr11 != qa.end() ){
        //cout<<"sum0 "<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto& g = va.grad;
    //cout<<"sum1 "<<g<<endl;
    
    auto itr11 = qa.begin();
    auto itr12 = g.begin();
    while( itr11 != qa.end() ){
        //cout<<"sum1 "<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
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
    auto z = op1.output;
    //cout<<"mean"<<z<<endl;
    
    auto itr11 = z.begin();
    auto itr12 = q.begin();
    while( itr11 != z.end() ){ 
        EXPECT_EQ( *itr11++, *itr12++);
    }
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
    auto& g = va.grad;
    //cout<<"mean"<<g<<endl;
    
    auto itr11 = qa.begin();
    auto itr12 = g.begin();
    while( itr11 != qa.end() ){
        //cout<<"mean "<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
}