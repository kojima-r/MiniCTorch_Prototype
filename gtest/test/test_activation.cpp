#include <gtest/gtest.h>
#include "minictorch.hpp"


/*
Tensor numerial_diff_relu( Tensor& x, fprec eps=1.0e-4 )
{
    auto grad = xt::zeros_like( x );
    
    auto itr  = x.begin();
    auto itr1 = grad.begin();
    auto itr2 = itr1;
    int l=0;
    while( itr != x.end() )
    {
        fprec  tmp = *itr;
        *itr = tmp + eps;
        
        Tensor x1 = calc_relu( x );
        
        *itr = tmp - eps;
        Tensor x2 = calc_relu( x );
        
        fprec diff = x1[l] - x2[l];
        *itr2 = diff/(2.0*eps);
        //cout<<"diff"<<*itr2<<","<<l<<","<<x1[l]<<","<<x2[l]<<endl;
        
        *itr = tmp;
        itr++;  itr2++; l++;
    }
    
    //cout<<"grad"<<grad<<endl;
    return grad;
}*/

Tensor numerial_diff( Tensor& x, Tensor(*func)(Tensor &a), fprec eps=1.0e-3 )
{
    auto grad = xt::zeros_like( x );
    
    auto itr  = x.begin();
    auto itr1 = grad.begin();
    auto itr2 = itr1;
    int l=0;
    while( itr != x.end() )
    {
        fprec  tmp = *itr;
        *itr = tmp + eps;
        Tensor x1 = func( x );
        
        *itr = tmp - eps;
        Tensor x2 = func( x );
        
        fprec diff = x1[l] - x2[l];
        *itr2 = diff/(2.0*eps);
        //cout<<"diff"<<*itr2<<","<<l<<","<<x1[l]<<","<<x2[l]<<endl;
        
        *itr = tmp;
        itr++;  itr2++; l++;
    }
    
    //cout<<"grad"<<grad<<endl;
    return grad;
}

Tensor numerial_diff1( Tensor& x, Tensor(*func)(Tensor &a,fprec b), fprec e=1.0, fprec eps=1.0e-3 )
{
    auto grad = xt::zeros_like( x );
    
    auto itr  = x.begin();
    auto itr1 = grad.begin();
    auto itr2 = itr1;
    int l=0;
    while( itr != x.end() )
    {
        fprec  tmp = *itr;
        *itr = tmp + eps;
        Tensor x1 = func( x, e );
        
        *itr = tmp - eps;
        Tensor x2 = func( x, e );
        
        fprec diff = x1[l] - x2[l];
        *itr2 = diff/(2.0*eps);
        //cout<<"diff1 "<<*itr2<<","<<l<<","<<x1[l]<<","<<x2[l]<<endl;
        
        *itr = tmp;
        itr++;  itr2++; l++;
    }
    
    //cout<<"grad"<<grad<<endl;
    return grad;
}

Tensor numerial_diff2( Tensor& x, Tensor(*func)(Tensor &a,fprec b,fprec c), fprec e1=1.0, fprec e2=1.0, fprec eps=1.0e-3 )
{
    auto grad = xt::zeros_like( x );
    
    auto itr  = x.begin();
    auto itr1 = grad.begin();
    auto itr2 = itr1;
    int l=0;
    while( itr != x.end() )
    {
        fprec  tmp = *itr;
        *itr = tmp + eps;
        Tensor x1 = func( x, e1, e2 );

        *itr = tmp - eps;
        Tensor x2 = func( x, e1, e2 );
        
        fprec diff = x1[l] - x2[l];
        *itr2 = diff/(2.0*eps);
        //cout<<"diff2 "<<*itr2<<","<<l<<","<<x1[l]<<","<<x2[l]<<endl;
        
        *itr = tmp;
        itr++;  itr2++; l++;
    }
    
    //cout<<"grad"<<grad<<endl;
    return grad;
}

void fail_msg( Tensor& g1, Tensor& g2 )
{
    cout<<" "<<endl;
    cout<<"========== FAILED (Gradient Check) =========="<<endl;
    cout<<"Backprop Grad"<<endl;
    cout<<xt::flatten(g1)<<endl;
    cout<<"Numerical Grad"<<endl;
    cout<<xt::flatten(g2)<<endl;
}

bool allclose( fprec a, fprec b, fprec atol=1.0e-5, fprec rtol=1.0e-8 )
{
    fprec eps = atol + rtol*fabs(b);
    //cout<<"eps"<<eps<<endl;
    return ( fabs(a-b) < eps );
}

// ----- relu -----

Tensor calc_relu( Tensor& a )
{
    VariableTensor va(a);
    ReluOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    return op1.output;
}

Tensor diff_relu( Tensor& a )
{
    VariableTensor va(a);
    ReluOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestRelu)
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    Tensor q = {{ 0.,0.}, { 2., 0.}, { 0.,1.}};
    
    Tensor z = calc_relu( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"relu"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestReluGrad) 
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    Tensor q = {{ 0.,0.}, { 1., 0.}, { 0.,1.}};
    
    Tensor g1 = diff_relu( a );
    
    fprec tol = 1.0e-3;
    auto itr1 = g1.begin();
    auto itr2 = q.begin();
    while( itr1 != g1.end() ){ 
        //fprec g = fabs( *itr1 - *itr2 );
        //EXPECT_LE( g, tol );
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestReluGrad2) 
{
    Tensor a = {{-1.,0.01}, { 2.,-3.}, {-2.,1.}};
   
    Tensor g1 = diff_relu( a );
    Tensor g2 = numerial_diff( a, calc_relu );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"relu_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- elu -----

fprec calc_elu1( fprec x, fprec alpha=1.0 )
{
    if( x > 0.0 )  return x;
    return ( alpha * exp(x) - 1.0 );
}

fprec diff_elu1( fprec x, fprec alpha=1.0 )
{
    if( x > 0.0 )  return 1.0;
    return ( alpha * exp(x) );
}

Tensor calc_elu( Tensor& a, fprec alpha=1.0 )
{
    VariableTensor va( a );
    VariableTensor vb( alpha );
    EluOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    return op1.output;
}

Tensor diff_elu( Tensor& a, fprec alpha=1.0 )
{
    VariableTensor va( a );
    VariableTensor vb( alpha );
    EluOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestElu)
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    Tensor q = xt::zeros_like( a );
    
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = calc_elu1( *itr11 );
        //cout<<"elu "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor z = calc_elu( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"elu"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestEluGrad) 
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = diff_elu1( *itr11 );
        //cout<<"elu "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor g = diff_elu( a );
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestEluGrad2) 
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
   
    Tensor g1 = diff_elu( a );
    Tensor g2 = numerial_diff1( a, calc_elu, 1.0 );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"elu_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- leaky_relu -----

fprec calc_leaky_relu1( fprec x, fprec slope=0.01 )
{
    if( x >= 0.0 )  return x;
    return ( slope * x );
}

fprec diff_leaky_relu1( fprec x, fprec slope=0.01 )
{
    if( x >= 0.0 )  return 1.0;
    return ( slope );
}

Tensor calc_leaky_relu( Tensor& a, fprec slope=0.01 )
{
    VariableTensor va( a );
    VariableTensor vb( slope );
    LeakyReluOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    return op1.output;
}

Tensor diff_leaky_relu( Tensor& a, fprec slope=0.01 )
{
    VariableTensor va( a );
    VariableTensor vb( slope );
    LeakyReluOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestLeakyRelu)
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    Tensor q = xt::zeros_like( a );
    
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = calc_leaky_relu1( *itr11 );
        //cout<<"leaky_relu "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor z = calc_leaky_relu( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"leaky_relu"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestLeakyReluGrad) 
{
    Tensor a = {{-1.,0.}, { 2.,-3.}, {-2.,1.}};
    
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = diff_leaky_relu1( *itr11 );
        //cout<<"leaky_relu "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor g = diff_leaky_relu( a );
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        //cout<<"leaky_relu "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestLeakyReluGrad2) 
{
    Tensor a = {{-1.,0.01}, { 2.,-3.}, {-2.,1.}};
   
    Tensor g1 = diff_leaky_relu( a );
    Tensor g2 = numerial_diff1( a, calc_leaky_relu, 0.01 );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"leaky_relu_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- hardtanh(relu6) -----

fprec calc_hard_tanh1( fprec x, fprec fmin=0.0, fprec fmax=6.0 )
{
    if( x > fmax )  return fmax;
    if( x < fmin )  return fmin;
    return x;
}

fprec diff_hard_tanh1( fprec x, fprec fmin=0.0, fprec fmax=6.0 )
{
    if( x > fmax )  return 0.0;
    if( x < fmin )  return 0.0;
    return 1.0;
}

Tensor calc_hard_tanh( Tensor& a, fprec fmin=0.0, fprec fmax=6.0 )
{
    VariableTensor va( a );
    VariableTensor vb( fmin );
    VariableTensor vc( fmax );
    HardTanhOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    op1.forward();
    return op1.output;
}

Tensor diff_hard_tanh( Tensor& a, fprec fmin=0.0, fprec fmax=6.0 )
{
    VariableTensor va( a );
    VariableTensor vb( fmin );
    VariableTensor vc( fmax );
    HardTanhOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestHardTanh)
{
    Tensor a = {{-1.,0.}, { 7.,-3.}, {-2.,3.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = calc_hard_tanh1( *itr11 );
        //cout<<"hard_tanh "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor z = calc_hard_tanh( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"hard_tanh "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestHardTanhGrad) 
{
    Tensor a = {{-1.,0.01}, { 7.,-3.}, {-2.,3.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = diff_hard_tanh1( *itr11 );
        //cout<<"hardtanh "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor g = diff_hard_tanh( a );
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        //cout<<"hard_tanh "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestHardTanhGrad2) 
{
    Tensor a = {{-1.,0.01}, { 7.,-3.}, {-2.,3.}};
   
    Tensor g1 = diff_hard_tanh( a );
    Tensor g2 = numerial_diff2( a, calc_hard_tanh, 0.0, 6.0 );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"hard_tanh_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- tanh -----

Tensor calc_tanh( Tensor& a )
{
    VariableTensor va( a );
    TanhOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    return op1.output;
}

Tensor diff_tanh( Tensor& a )
{
    VariableTensor va( a );
    TanhOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestTanh)
{
    Tensor a = {{-1.,0.}, { 7.,-3.}, {-2.,3.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = tanh( *itr11 );
        //cout<<"tanh "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor z = calc_tanh( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"tanh "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestTanhGrad) 
{
    Tensor a = {{-1.,0.01}, { 7.,-3.}, {-2.,3.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        fprec s = tanh( *itr11 );
        *itr12 = ( 1.0 - s * s );
        //cout<<"tanh "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor g = diff_tanh( a );
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        //cout<<"tanh "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestTanhGrad2) 
{
    Tensor a = {{-1.,0.01}, { 7.,-3.}, {-2.,3.}};
   
    Tensor g1 = diff_tanh( a );
    Tensor g2 = numerial_diff( a, calc_tanh );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"hard_tanh_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- softplus -----

fprec calc_softplus1( fprec x, fprec beta=1.0, fprec threshold=20.0 )
{
    if( x * beta > threshold )  return x;
    return ( log( 1.0 + exp( beta * x ) )/ beta );
}

fprec diff_softplus1( fprec x, fprec beta=1.0, fprec threshold=20.0 )
{
    if( x * beta > threshold )  return 1.0;
    return ( 1.0/( 1.0 + 1.0 / exp( beta * x ) ) );
}

Tensor calc_softplus( Tensor& a, fprec beta=1.0, fprec threshold=20.0 )
{
    VariableTensor va( a );
    VariableTensor vb( beta );
    VariableTensor vc( threshold );
    SoftplusOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    op1.forward();
    return op1.output;
}

Tensor diff_softplus( Tensor& a, fprec beta=1.0, fprec threshold=20.0 )
{
    VariableTensor va( a );
    VariableTensor vb( beta );
    VariableTensor vc( threshold );
    SoftplusOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestSoftPlus)
{
    Tensor a = {{-1.,0.}, { 4.,-3.}, {-2.,1.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = calc_softplus1( *itr11 );
        //cout<<"softplus "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor z = calc_softplus( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"softplus "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestSoftPlusGrad) 
{
    Tensor a = {{-1.,0}, { 4.,-3.}, {-2.,1.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = diff_softplus1( *itr11 );
        //cout<<"softplus "<<*itr11<<","<<*itr12<<endl;
        itr11++; itr12++;
    }
    
    Tensor g = diff_softplus( a );
    //cout<<"softplus "<<g<<endl;
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        //cout<<"softplus "<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestSoftPlusGrad2) 
{
    Tensor a = {{-1.,0.}, { 4.,-3.}, {-2.,1.}};
   
    Tensor g1 = diff_softplus( a );
    Tensor g2 = numerial_diff2( a, calc_softplus,1.0, 20.0 );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //bool b = allclose( *itr1, *itr2, 1.0e-3, 1.0e-5 );
        //if( !b ) res = false;
        //cout<<"hard_tanh_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        //EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}

// ----- sigmoid -----

fprec calc_sigmoid1( fprec x )
{
    return ( 1.0 / ( 1.0+exp(-x) ) );
}

fprec diff_sigmoid1( fprec x )
{
    fprec f = ( 1.0 / ( 1.0+exp(-x) ) );
    return ( f * ( 1.0 - f ) );
}

Tensor calc_sigmoid( Tensor &a )
{
    VariableTensor va(a);
    SigmoidOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    return op1.output;
}

Tensor diff_sigmoid( Tensor& a )
{
    VariableTensor va(a);
    SigmoidOp  op1;
    op1.set_inputs( &va );
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestSigmoid) 
{
    Tensor a = {{0.,1.,2.}, { 0.,2.,4.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = calc_sigmoid1( *itr11 );
        itr11++;  itr12++;
    }
    
    Tensor z = calc_sigmoid( a );
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //cout<<"sigmoid"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestSigmoidGrad) 
{
    Tensor a = {{0.,1.,2.}, { 0.,2.,4.}};
    Tensor q = xt::zeros_like( a );
    auto itr11 = a.begin();
    auto itr12 = q.begin();
    while( itr11 != a.end() ){ 
        *itr12 = diff_sigmoid1( *itr11 );
        itr11++;  itr12++;
    }
    
    Tensor g = diff_sigmoid( a );
    
    fprec tol = 1.0e-3;
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        //cout<<"diff_sigmoid"<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1, *itr2 );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestSigmoidGrad2) 
{
    Tensor a = {{0.,1.,2.}, { 0.,2.,4.}};
    Tensor q = xt::zeros_like( a );
    
    Tensor g1 = diff_sigmoid( a );
    Tensor g2 = numerial_diff( a, calc_sigmoid );
    
    fprec tol = 1.0e-3;
    bool res = true;
    auto itr1 = g1.begin();
    auto itr2 = g2.begin();
    while( itr1 != g1.end() ){ 
        //cout<<"sigmoid_diff "<<*itr1<<","<<*itr2<<endl;
        fprec g = fabs( *itr1 - *itr2 );
        EXPECT_LE( g, tol );
        itr1++; itr2++;
    }
    
    if( !res ) fail_msg( g1, g2 );
}
