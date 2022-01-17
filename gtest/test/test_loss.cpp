#include <gtest/gtest.h>
#include "minictorch.hpp"


// ----- mse_loss -----

Tensor calc_mseloss( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(1.0);
    
    MseLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    
    op1.forward();
    return op1.output;
}

Tensor diff_mseloss( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(1.0);
    
    MseLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vc );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

TEST(MyTestCase, TestMseLoss) 
{
    Tensor a = {1.,2.,3.};
    Tensor b = {4.,5.,6.};
   
    auto z = calc_mseloss( a, b );
    //cout<<"z"<<z<<endl;
    
    auto as = a.shape();
    Tensor diff = a - b;
    Tensor q = xt::sum(diff*diff)/as[0];
    //cout<<"q"<<q<<endl;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestMseLossGrad) 
{
    Tensor a = {1.,2.,3.};
    Tensor b = {4.,5.,6.};
    
    auto g = diff_mseloss( a, b );
    //cout<<"mse g"<<g<<endl;
    
    auto as = a.shape();
    Tensor diff = a - b;
    Tensor q = diff * 2.0 / as[0];
    //cout<<"mse q"<<q<<endl;
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}


// ----- cross_entropy_loss -----

Tensor calc_cross_entropy( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(-100.0);
    
    CrossEntropyLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( NULL );
    op1.set_inputs( &vc );
    
    op1.forward();
    return op1.output;
}

Tensor diff_cross_entropy( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(-100.0);
    
    CrossEntropyLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( NULL );
    op1.set_inputs( &vc );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

Tensor calc_softmax1( Tensor &x, unsigned int ax=1 )
{
    Tensor xm = xt::amax(x, {ax} );
    xm = xt::expand_dims( xm, {ax} );
    auto xx = x - xm;
    auto xe = xt::exp(xx);
    Tensor xz = xt::sum( xe, {ax} );
    xz = xt::expand_dims( xz, {ax});
    auto x2 = xe / xz;
    return x2;
}

Tensor calc_log_softmax1( Tensor &x, fprec e=1.0e-7 )
{
    Tensor xo = calc_softmax1( x );
    Tensor y  = xt::log( xo + e );
    return y;
}

Tensor diff_softmax1( Tensor &x, unsigned int ax=1 )
{
    Tensor ga = calc_softmax1( x, ax );
    Tensor sg = xt::sum( ga, {ax} );
    sg = xt::expand_dims( sg, {ax} );
    Tensor g = ( ga - ga * sg );
    return g;
}

Tensor diff_log_softmax1( Tensor &x, fprec e=1.0e-7 )
{
    auto   xo = calc_softmax1( x );
    Tensor ga = xt::ones<fprec>( x.shape() );
    Tensor gm = xt::sum( ga, {1} );
    gm.reshape({-1,1});
    Tensor q = ( ga - xo * gm );
    return q;
}

fprec calc_cross_entropy1( Tensor &x, Tensor &t )
{
    auto   xs = x.shape();
    Tensor y  = calc_log_softmax1( x );
    fprec  s  = 0.0;
    for(int k=0;k<xs[0];k++)  s += y(k,t[k]);
    s /= xs[0];
    return -s;
}

Tensor diff_cross_entropy1( Tensor &x, Tensor &t )
{
    auto   xs  = x.shape();
    Tensor xo  = calc_softmax1( x );
    Tensor one = xt::zeros_like( x );
    for(int i=0;i<xs[0];i++)  one(i,t[i]) = 1.0;
    auto g = ( xo - one )/ xs[0];
    return g;
}

TEST(MyTestCase, TestCrossEntropyLoss) 
{
    Tensor a = {{ -1., 0., 1.,  2.},
                {  2., 0., 1., -1.} };
    Tensor b = { 3., 0. };
   
    Tensor z = calc_cross_entropy( a, b );
    //cout<<"z "<<z<<endl;
    
    Tensor q = calc_cross_entropy1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //EXPECT_EQ( *itr1++, *itr2++);
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestCrossEntropyLossGrad) 
{
    Tensor a = {{ -1., 0., 1.,  2.},
                {  2., 0., 1., -1.} };
    Tensor b = { 3., 0. };
   
    Tensor g = diff_cross_entropy( a, b );
    //cout<<"g "<<g<<endl;
    
    Tensor q = diff_cross_entropy1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

// ----- binary_cross_entropy -----

Tensor calc_binary_cross_entropy( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(1.0);
    
    BCELossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( &vc );
    
    op1.forward();
    return op1.output;
}

Tensor diff_binary_cross_entropy( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(1.0);
    
    BCELossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( &vc );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

Tensor calc_binary_cross_entropy1( Tensor &x, Tensor &t, fprec e=1.0e-7 )
{
    auto  y = t*xt::log(x+e) + (1-t) * xt::log(1-x+e);
    Tensor s = -xt::sum(y) / x.size();
    return s;
}

Tensor diff_binary_cross_entropy1( Tensor &x, Tensor &t, fprec e=1.0e-7 )
{
    auto g = ( -t/(x+e) + (1-t)/(1-x+e) ) / x.size();
    return g;
}

TEST(MyTestCase, TestBinaryCrossEntropyLoss) 
{
    Tensor a = {{ 0.1, 0.3, 0.5, 0.9 }, 
                { 0.8, 0.3, 0.2, 0.1 }};
    Tensor b = { {0.,0.,0.,1.},{1.,0.,0.,0.} };
   
    Tensor z = calc_binary_cross_entropy( a, b );
    //cout<<"z "<<z<<endl;
    
    Tensor q = calc_binary_cross_entropy1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestBinaryCrossEntropyLossGrad) 
{
    Tensor a = {{ 0.1, 0.3, 0.5, 0.9 }, 
                { 0.8, 0.3, 0.2, 0.1 }};
    Tensor b = { {0.,0.,0.,1.},{1.,0.,0.,0.} };
   
    Tensor g = diff_binary_cross_entropy( a, b );
    //cout<<"g "<<g<<endl;
    
    Tensor q = diff_binary_cross_entropy1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

// ----- nll_loss -----

Tensor calc_nll_loss( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(-100.0);
    
    NLLLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( NULL );
    op1.set_inputs( &vc );
    
    op1.forward();
    return op1.output;
}

Tensor diff_nll_loss( Tensor &a, Tensor &b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vc(-100.0);
    
    NLLLossOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL);
    op1.set_inputs( NULL );
    op1.set_inputs( &vc );
    
    op1.forward();
    op1.grad = xt::ones_like( a );
    op1.backward();
    return va.grad;
}

Tensor diff_nll_loss1( Tensor &x, Tensor &t, fprec e=1.0e-7 )
{
    auto   xs  = x.shape();
    Tensor one = xt::zeros_like( x );
    for(int i=0;i<xs[0];i++)  one(i,t[i]) = 1.0;
    auto g = -one / xs[0];
    return g;
}

TEST(MyTestCase, TestNLLLoss) 
{
    Tensor a = {{ -1., 0., 1.,  2.},
                {  2., 0., 1., -1.} };
    Tensor b = { 3., 0. };
    Tensor x = calc_log_softmax1( a );
    Tensor z = calc_nll_loss( x, b );
    //cout<<"z "<<z<<endl;
    
    Tensor q = calc_cross_entropy1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        //EXPECT_EQ( *itr1++, *itr2++);
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

TEST(MyTestCase, TestNLLLossGrad) 
{
    Tensor a = {{ -1., 0., 1.,  2.},
                {  2., 0., 1., -1.} };
    Tensor b = { 3., 0. };
    Tensor x = calc_log_softmax1( a );
    Tensor g = diff_nll_loss( x, b );
    //cout<<"g "<<g<<endl;
    
    Tensor q = diff_nll_loss1( a, b );
    //cout<<"q "<<q<<endl;
    
    fprec tol = 1.0e-3;
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        fprec e = fabs( *itr1 - *itr2 );
        EXPECT_LE( e, tol );
        itr1++; itr2++;
    }
}

// ----- softmax -----

TEST(MyTestCase, TestSoftmax) {
    
    Tensor a = {{0.,1.,2.}, {0.,2.,4.}};
    VariableTensor va(a);
    
    SoftmaxOp  op1;
    op1.set_inputs( &va );
    
    op1.forward();
    auto z = op1.output;
    //cout<<"z"<<z<<endl;
    
    auto q = calc_softmax1( a );
    //cout<<"q"<<q<<endl;
    
    fprec tol = 1.0e-4;
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        fprec delta = fabs( *itr1 - *itr2 );
        EXPECT_LE( delta, tol );
        itr1++;  itr2++;
    }
}

TEST(MyTestCase, TestLogSoftmax) {
    
    Tensor a = {{0.,1.,2.}, {0.,2.,4.}};
    VariableTensor va(a);
    
    LogSoftmaxOp  op1;
    op1.set_inputs( &va );
    
    op1.forward();
    auto z = op1.output;
    //cout<<"z"<<z<<endl;
    
    auto q = calc_log_softmax1( a );
    //cout<<"q"<<q<<endl;
    
    fprec tol = 1.0e-4;
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        fprec delta = fabs( *itr1 - *itr2 );
        EXPECT_LE( delta, tol );
        itr1++;  itr2++;
    }
}

TEST(MyTestCase, TestLogSoftmaxGrad) 
{
    Tensor a = {{0.,1.,2.}, {0.,2.,4.}};
    VariableTensor va(a);
    
    LogSoftmaxOp  op1;
    op1.set_inputs( &va );
    
    op1.forward();
    op1.grad = xt::ones<fprec>( {2,3} );
    op1.backward();
    
    auto& g = va.grad;
    //cout<<"g"<<g<<endl;
    
    Tensor q = diff_log_softmax1( a );
    //cout<<"q"<<q<<endl;
    
    fprec tol = 1.0e-4;
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        fprec delta = fabs( *itr1 - *itr2 );
        EXPECT_LE( delta, tol );
        itr1++;  itr2++;
    }
}
