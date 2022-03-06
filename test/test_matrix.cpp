#include <gtest/gtest.h>
#include "minictorch.hpp"


void expect_eq1( Tensor &z, Tensor &q )
{
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){ 
        EXPECT_EQ( *itr1++, *itr2++);
    }
}

void expect_eq4( string s, Tensor& z, Tensor &ga, Tensor& gb, Tensor &gd, Tensor& q, Tensor& qa, Tensor& qb, Tensor &qd )
{
    //cout<<"z"<<z<<endl;
    //cout<<"ga"<<ga<<endl;
    //cout<<"gb"<<gb<<endl;
    //cout<<"gd"<<gd<<endl;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<s<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
        
    auto itr11 = ga.begin();
    auto itr12 = qa.begin();
    while( itr11 != ga.end() ){
        //cout<<s<<*itr11<<","<<*itr12<<endl;
        EXPECT_EQ( *itr11++, *itr12++ );
    }
        
    auto itr21 = gb.begin();
    auto itr22 = qb.begin();
    while( itr21 != gb.end() ){
        //cout<<s<<*itr21<<","<<*itr22<<endl;
        EXPECT_EQ( *itr21++, *itr22++ );
    }
    
    auto itr31 = gd.begin();
    auto itr32 = qd.begin();
    while( itr31 != gd.end() ){
        //cout<<s<<*itr31<<","<<*itr32<<endl;
        EXPECT_EQ( *itr31++, *itr32++ );
    }
}

// ----- linear -----

// linear
//  a * b_t + d : vector x matrix + vector
TEST(MyTestCase, TestLinear1) 
{
    Tensor a = {1.,2.,3.};
    Tensor b = {{1.,2.,3.}, {4.,5.,6.}};  //xt::transpose(a);
    Tensor d =  {1.,2.};
    
    Tensor q  = {{15.,34.}};
    Tensor qa =  {5.,7.,9.};
    Tensor qb = {{1.,2.,3.},{1.,2.,3.}};
    Tensor qd =  {1.,1.};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vd(d);
    
    LinearOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vd );
    
    op1.forward();
    auto z = op1.output;
    
    op1.grad = xt::ones_like( z );
    op1.backward();
 
    expect_eq4( "linear1", z, va.grad, vb.grad, vd.grad, q, qa, qb, qd );
}

// linear
//  a * b_t + d : matrix x matrix + vector
TEST(MyTestCase, TestLinear2) 
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}};
    Tensor b = a; //xt::transpose(a);
    Tensor d =  {1.,2.};
    
    Tensor q  = {{15.,34.}, {33.,79.}};
    Tensor qa = {{5.,7.,9.},{5.,7.,9.}};
    Tensor qb = {{5.,7.,9.},{5.,7.,9.}};
    Tensor qd = {2.,2.};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vd(d);
    
    LinearOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vd );
    
    op1.forward();
    auto z = op1.output;
    
    op1.grad = xt::ones_like( z );
    op1.backward();
    
    expect_eq4( "linear2", z, va.grad, vb.grad, vd.grad, q, qa, qb, qd );
}

// linear
//  a * b_t + d : batched matrix x matrix + vector
TEST(MyTestCase, TestLinear3) 
{
    Tensor a = {{1.,2.,3.}, {4.,5.,6.}, {7.,8.,9.}};
    Tensor b = {{1.,2.,3.}, {4.,5.,6.}};  //xt::transpose(a);
    Tensor d =  {1.,2.};
    
    Tensor q  = {{  15.,   34.},
                 {  33.,   79.},
                 {  51.,  124.}};
    Tensor qa = {{ 5.,  7.,  9.},
                 { 5.,  7.,  9.},
                 { 5.,  7.,  9.}};
    Tensor qb = {{ 12.,  15.,  18.},
                 { 12.,  15.,  18.}};
    Tensor qd =  { 3.,  3.};
    
    VariableTensor va(a);
    VariableTensor vb(b);
    VariableTensor vd(d);
    
    LinearOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( &vd );
    
    op1.forward();
    auto z = op1.output;
    
    op1.grad = xt::ones_like( z );
    op1.backward();
    
    expect_eq4( "linear3", z, va.grad, vb.grad, vd.grad, q, qa, qb, qd );
}

// ----- addmm

std::tuple<Tensor,Tensor,Tensor,Tensor> do_addmm( Tensor &m1, Tensor& m2, Tensor& inp, fprec alpha=1.0, fprec beta=1.0 )
{
    VariableTensor v1(m1);
    VariableTensor v2(m2);
    VariableTensor v3(inp);
    VariableTensor v4(alpha);  
    VariableTensor v5(beta);  
    
    AddMmOp  op1;
    op1.set_inputs( &v3 );
    op1.set_inputs( &v1 );
    op1.set_inputs( &v2 );
    op1.set_inputs( &v5 );
    op1.set_inputs( &v4 );
    op1.forward();
    
    op1.grad = xt::ones_like( op1.output );
    op1.backward();
    
    return std::make_tuple( op1.output, v1.grad, v2.grad, v3.grad);
}

// addmm ( add matrix )
TEST(MyTestCase, TestAddmm1) 
{
    Tensor mat1 = xt::arange<fprec>(1., 7.).reshape({2,3} );
    Tensor mat2 = xt::arange<fprec>(1.,10.).reshape({3,3});
    Tensor inp1 = xt::ones<fprec>({2,3});
    
    Tensor q =  {{ 31.,  37.,  43.},
                 { 67.,  82.,  97.}};
    Tensor qa = {{  6.,  15.,  24.},
                 {  6.,  15.,  24.}};
    Tensor qb = {{ 5.,  5.,  5.},
                 { 7.,  7.,  7.},
                 { 9.,  9.,  9.}};
    Tensor qd = {{ 1.,  1.,  1.},
                 { 1.,  1.,  1.}};
    
    Tensor z,ga,gb,gd;
    std::tie(z,ga,gb,gd) = do_addmm( mat1,mat2, inp1 );
    
    expect_eq4( "addmm  ", z, ga, gb, gd, q, qa, qb, qd );
}

// addmm ( add vector )
TEST(MyTestCase, TestAddmm2) 
{
    Tensor mat1 = xt::arange<fprec>(1., 7.).reshape({2,3} );
    Tensor mat2 = xt::arange<fprec>(1.,10.).reshape({3,3});
    Tensor inp1 = xt::arange<fprec>(1.,4.);
    
    Tensor q =  {{ 31.,  38.,  45.},
                 { 67.,  83.,  99.}};
    Tensor qa = {{  6.,  15.,  24.},
                 {  6.,  15.,  24.}};
    Tensor qb = {{ 5.,  5.,  5.},
                 { 7.,  7.,  7.},
                 { 9.,  9.,  9.}};
    Tensor qd =  { 2.,  2.,  2.};
    
    Tensor z,ga,gb,gd;
    std::tie(z,ga,gb,gd) = do_addmm( mat1,mat2, inp1 );
    
    expect_eq4( "addmm  ",z, ga, gb, gd, q, qa, qb, qd );
}

// addmm ( add constant )
TEST(MyTestCase, TestAddmm3) 
{
    Tensor mat1 = xt::arange<fprec>(1., 7.).reshape({2,3} );
    Tensor mat2 = xt::arange<fprec>(1.,10.).reshape({3,3});
    Tensor inp1 = xt::ones<fprec>({1});
    
    Tensor q =  {{ 31.,  37.,  43.},
                 { 67.,  82.,  97.}};
    Tensor qa = {{  6.,  15.,  24.},
                 {  6.,  15.,  24.}};
    Tensor qb = {{ 5.,  5.,  5.},
                 { 7.,  7.,  7.},
                 { 9.,  9.,  9.}};
    Tensor qd =  { 6.};
    
    Tensor z,ga,gb,gd;
    std::tie(z,ga,gb,gd) = do_addmm( mat1,mat2, inp1 );
    
    expect_eq4( "addmm  ",z, ga, gb, gd, q, qa, qb, qd );
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
    
    expect_eq1( y, q );
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
    //cout<<"y"<<y<<endl;
    
    auto q = diff_batch_norm1( true, x, gamma, beta, mean, var );
    //cout<<"q"<<q<<endl;
    
    expect_eq1( y, q );
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
    
    expect_eq1( y, q );
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
    
    expect_eq1( g, q );
}