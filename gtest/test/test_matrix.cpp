#include <gtest/gtest.h>
#include "minictorch.hpp"

// ----- matmul -----

TEST(MyTestCase, TestMatMul) 
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}};
    Tensor b = xt::transpose(a);
    
    VariableTensor va(a);
    VariableTensor vb(b);
    
    MatMulOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    
    op1.forward();
    auto z = op1.output;
    
    Tensor q = {{14.,32.}, {32.,77.}};
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestMatMulGrad)
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}};
    Tensor b = xt::transpose(a);
  
    VariableTensor va(a);
    VariableTensor vb(b);
    
    MatMulOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    
    op1.grad = xt::ones<fprec>( {2,2} );
    op1.backward();
    
    auto& ga = va.grad;
    auto& gb = vb.grad;
    
    Tensor qa = {{5.,7.,9.},{5.,7.,9.}};
    Tensor qb = {{5.,5.},{7.,7.},{9.,9.}};
    
    /*
    cout<<"ga"<<ga<<endl;
    cout<<"gb"<<gb<<endl;
    cout<<"qa"<<qa<<endl;
    cout<<"qb"<<qb<<endl;*/
    
    auto itr1 = ga.begin();
    auto itr2 = qa.begin();
    while( itr1 != ga.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++ );
    }
    
    auto itr3 = gb.begin();
    auto itr4 = qb.begin();
    while( itr3 != gb.end() ){ 
        EXPECT_EQ( *itr3++, *itr4++ );
    }
}

// ----- linear -----

TEST(MyTestCase, TestLinear) 
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}};
    Tensor b = a; //xt::transpose(a);
    Tensor d = {1.,2.};
    
    Tensor q = {{15.,34.}, {33.,79.}};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vd(d);
    
    LinearOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vd );
    
    op1.forward();
    auto z = op1.output;

    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestLinearGrad) 
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}};
    Tensor b = a; //xt::transpose(a);
    Tensor d = {1.,2.};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vd(d);
    
    LinearOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vd );
    
    op1.grad = xt::ones<fprec>( {2,2} );
    op1.backward();
    
    auto& ga = va.grad;
    auto& gb = vb.grad;
    auto& gd = vd.grad;
    //cout<<"ga"<<ga<<endl;
    //cout<<"gb"<<gb<<endl;
    //cout<<"gd"<<gd<<endl;
    
    Tensor qa = {{5.,7.,9.},{5.,7.,9.}};
    Tensor qb = {{5.,7.,9.},{5.,7.,9.}};
    Tensor qd = {2.,2.};

    auto itr1 = ga.begin();
    auto itr2 = qa.begin();
    while( itr1 != ga.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
    
    auto itr3 = gb.begin();
    auto itr4 = qb.begin();
    while( itr3 != gb.end() ){ 
        EXPECT_EQ( *itr3++, *itr4++);
    }
    
    auto itr5 = gd.begin();
    auto itr6 = qd.begin();
    while( itr5 != gd.end() ){ 
        EXPECT_EQ( *itr5++, *itr6++);
    }
}

// ----- batchnorm -----

bool train_mode = true;

Tensor calc_batch_norm1( bool mode, Tensor &x, Tensor &gm, Tensor &b, Tensor &rm, Tensor &rv
                       , fprec mom=0.1, fprec eps=1.0e-7 )
{
    auto xs = x.shape();
    
    Tensor xn;
    if( mode ) // train_mode
    {
        Tensor mean = xt::mean( x, {0} );
        Tensor var  = xt::variance( x, {0} );
        xn = ( x - mean ) / xt::sqrt( var + eps );
        int   gz = gm.size();
        auto  xs = x.shape();
        fprec m  = (fprec)xs[1] / (fprec)gz;
        fprec adjust = ( m > 2.0 ) ? (m-1.0): 1.0;  
        rm = rm * (1-mom) + mom * mean; // running_mean
        rv = rv * (1-mom) + mom * var * adjust; // running_var
    } else {
        xn = ( x - rm ) / ( xt::sqrt( rv + eps ) );
    }
    auto y = gm * xn + b;
    return y;
}

Tensor diff_batch_norm1( bool mode, Tensor &x, Tensor &g, Tensor &b, Tensor &rm, Tensor &rv
                       , fprec mom=0.1, fprec eps=1.0e-7 )
{
    Tensor mean = xt::mean( x, {0} );
    Tensor var  = xt::variance( x, {0} );
    Tensor xc,xn,std;
    if( mode ) { // train_mode
        xc  = x - mean;
        std = xt::sqrt( var + eps );
        xn  = xc / std;
    } else {
        xc  = x - rm;
        std = xt::sqrt( rv + eps );
        xn  = xc / std;
    }
    
    auto xs = x.shape();
    Tensor gc = xt::ones<fprec>( xs );
        
    Tensor dxn  = g * gc;
    Tensor dxc  = dxn / std;
    Tensor dvar = xt::sum( ( dxn*xc )/(std*std), {0} );
    dvar = dvar / std;
    dxc  = dxc - xc * dvar / xs[0];
    Tensor dmu = xt::sum( dxc, {0} );
    Tensor gx = dxc - dmu / xs[0];
    Tensor gm = xt::sum( gc * xn , {0} );
    Tensor gb = xt::sum( gc, {0} );
    
    return gx;
}

Tensor calc_batch_norm( Tensor &x, Tensor &g, Tensor &b, Tensor &rm, Tensor &rv)
{
    VariableTensor v0(x);
    VariableTensor v1(g);
    VariableTensor v2(b);
    VariableTensor v3(rm);
    VariableTensor v4(rv);
    VariableTensor v5(0.0);
    VariableTensor v6(0.1);
    VariableTensor v7(1.0e-7);
    
    BatchNormOp op1;
    op1.set_inputs( &v0 );
    op1.set_inputs( &v1 );
    op1.set_inputs( &v2 );
    op1.set_inputs( &v3 );
    op1.set_inputs( &v4 );
    op1.set_inputs( NULL );
    op1.set_inputs( &v6 );
    op1.set_inputs( &v7 );
    
    op1.forward();
    return op1.output;
}

Tensor diff_batch_norm( Tensor &x, Tensor &g, Tensor &b, Tensor &rm, Tensor &rv)
{
    VariableTensor v0(x);
    VariableTensor v1(g);
    VariableTensor v2(b);
    VariableTensor v3(rm);
    VariableTensor v4(rv);
    VariableTensor v5(0.0);
    VariableTensor v6(0.1);
    VariableTensor v7(1.0e-7);
    
    BatchNormOp op1;
    op1.set_inputs( &v0 );
    op1.set_inputs( &v1 );
    op1.set_inputs( &v2 );
    op1.set_inputs( &v3 );
    op1.set_inputs( &v4 );
    op1.set_inputs( NULL );
    op1.set_inputs( &v6 );
    op1.set_inputs( &v7 );
    
    auto xs = x.shape();
    op1.forward();
    op1.grad = xt::ones<fprec>( xs );
    op1.backward();
    return v0.grad;
}

TEST(MyTestCase, TestBatchNorm) 
{
    xt::random::seed( 1 );
    int n1 = 4;
    int n2 = 2;
    
    Tensor x     = xt::random::randn<fprec>( {n1,n2} );
    Tensor gamma = xt::random::randn<fprec>( {n2} );
    Tensor beta  = xt::random::randn<fprec>( {n2} );
    Tensor mean  = xt::random::randn<fprec>( {n2} );
    Tensor var   = xt::abs( xt::random::randn<fprec>( {n2} ) );
    
    auto y = calc_batch_norm( x, gamma, beta, mean, var );
    //cout<<"y"<<y<<endl;
    
    auto q = calc_batch_norm1( true, x, gamma, beta, mean, var );
    //cout<<"q"<<q<<endl;
    
    auto itr1 = y.begin();
    auto itr2 = q.begin();
    while( itr1 != y.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestBatchNormGrad) 
{
    xt::random::seed( 1 );
    int n1 = 4;
    int n2 = 2;
    
    Tensor x     = xt::random::randn<fprec>( {n1,n2} );
    Tensor gamma = xt::random::randn<fprec>( {n2} );
    Tensor beta  = xt::random::randn<fprec>( {n2} );
    Tensor mean  = xt::random::randn<fprec>( {n2} );
    Tensor var   = xt::abs( xt::random::randn<fprec>( {n2} ) );
    
    auto y = diff_batch_norm( x, gamma, beta, mean, var );
    cout<<"y"<<y<<endl;
    
    auto q = diff_batch_norm1( true, x, gamma, beta, mean, var );
    cout<<"q"<<q<<endl;
    
    auto itr1 = y.begin();
    auto itr2 = q.begin();
    while( itr1 != y.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

// ----- droout -----

Tensor calc_dropout1( bool mode, Tensor& x, Tensor &dropout, int kind=1, fprec ratio=0.5 )
{
    Tensor y;
    if( mode ) { // train_mode
        //Tensor r = xt::random::rand<fprec>( x.shape() );
        if( kind == 1 ) { // inverted
            fprec scale = 1.0 / (1.0 - ratio);
            //dropout = xt::where( r > ratio, 1, 0 );
            y  = x * dropout * scale;
        } else {
            //dropout = xt::where( r > ratio, 1, 0 );
            y  = x * dropout;
        }
    } else {
        if( kind == 1 ) { // inverted
            y = x;
        } else {
            y = x * ( 1.0 - ratio );
        }
    }
    return y;
}

Tensor diff_dropout1( bool mode, Tensor &x, Tensor &dropout, int kind=1, fprec ratio=0.5 )
{
    Tensor gc = xt::ones<fprec>( x.shape() );
    
    Tensor gx;
    if( mode ) {  // train_mode
        if( kind == 1 ) { // inverted
            fprec scale = 1.0 / (1.0 - ratio);
            gx = gc * dropout * scale;
        } else {
            gx = gc * dropout;
        }
    } else {
        gx = gc;
    }
    return gx;
}

TEST(MyTestCase, TestDropout) 
{
    xt::random::seed( 1 );
    Tensor a = xt::random::randn<fprec>( {10,10} );
    
    VariableTensor va(a);
    VariableTensor vb(0.5);
    
    DropoutOp op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    
    op1.forward();
    auto y = op1.output;
    //cout<<"y"<<y<<endl;
    
    auto q = calc_dropout1( true, a, op1.dropout );
    //cout<<"q"<<q<<endl;
    
    auto itr1 = y.begin();
    auto itr2 = q.begin();
    while( itr1 != y.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

TEST(MyTestCase, TestDropoutGrad) 
{
    xt::random::seed( 1 );
    Tensor a = xt::random::randn<fprec>( {10,10} );
    
    VariableTensor va(a);
    VariableTensor vb(0.5);
    
    DropoutOp op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    
    op1.forward();
    op1.grad = xt::ones<fprec>( a.shape() );
    op1.backward();
    auto g = va.grad;
    //cout<<"g"<<g<<endl;
    
    auto q = diff_dropout1( true, a, op1.dropout );
    //cout<<"q"<<q<<endl;
    
    auto itr1 = g.begin();
    auto itr2 = q.begin();
    while( itr1 != g.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}