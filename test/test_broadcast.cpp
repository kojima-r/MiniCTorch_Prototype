#include <gtest/gtest.h>
#include "minictorch.hpp"


inline void expect_eq1( string s, Tensor& z, Tensor& q )
{
    //cout<<s<<" z"<<z<<endl;
    //cout<<s<<" q"<<z<<endl;
    
    auto itr1 = z.begin();
    auto itr2 = q.begin();
    while( itr1 != z.end() ){
        //cout<<s<<*itr1<<","<<*itr2<<endl;
        EXPECT_EQ( *itr1++, *itr2++ );
    }
}

inline void expect_eq2( string s, Tensor &ga, Tensor& gb, Tensor& qa, Tensor& qb )
{
    //cout<<"ga"<<ga<<endl;
    //cout<<"gb"<<gb<<endl;
        
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
}

inline void expect_eq3( string s, Tensor& z, Tensor &ga, Tensor& gb, Tensor& q, Tensor& qa, Tensor& qb )
{
    //cout<<"z"<<z<<endl;
    //cout<<"ga"<<ga<<endl;
    //cout<<"gb"<<gb<<endl;
    
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
}

// ----- broadcast add  -----

std::tuple<Tensor,Tensor,Tensor> do_bc_add( Tensor &a, Tensor& b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    AddOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.set_inputs( NULL );
    op1.forward();
    
    op1.grad = xt::ones_like( op1.output );
    op1.backward();
    
    return std::make_tuple( op1.output, va.grad, vb.grad );
}

// broadcast add : A(3,3) + B(1)
TEST(MyTestCase, TestBroadcastAdd1) 
{
    Tensor a = xt::ones<fprec>({3,3});
    Tensor b = { 2. };
    Tensor q  = xt::full_like( a, 3.);
    Tensor qa = xt::ones<fprec>({3,3});
    Tensor qb = 9.0;
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_bc_add( a, b );
    
    expect_eq3( "add  ", z, ga, gb, q, qa, qb );
}

// broadcast add : A(1,3) + B(3,1)
TEST(MyTestCase, TestBroadcastAdd2) 
{
    Tensor a = xt::ones<fprec>({3}) * 2.0;
    Tensor b = a;
    b.reshape({-1,1});
    Tensor q  = xt::ones<fprec>({3,3}) * 4.0;
    Tensor qa = xt::full_like(a, 3.0);
    Tensor qb = xt::full_like(b, 3.0);
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_bc_add( a, b );
    
    expect_eq3( "add  ", z, ga, gb, q, qa, qb );
}

// broadcast add : A(2,4,3) + B(4,1)
TEST(MyTestCase, TestBroadcastAdd3) 
{
    Tensor a = xt::arange(24).reshape({2,4,3});
    Tensor b = xt::arange(4).reshape({-1,1});
    Tensor q = {{{  0.,   1.,   2.}, {  4.,   5.,   6.},
                 {  8.,   9.,  10.}, { 12.,  13.,  14.}},
                {{ 12.,  13.,  14.}, { 16.,  17.,  18.},
                 { 20.,  21.,  22.}, { 24.,  25.,  26.}}};
    Tensor qa = xt::full_like( a, 1.0 );
    Tensor qb = xt::full_like( b, 1.0 ) * 6.0;
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_bc_add( a, b );
    
    expect_eq3( "add  ", z, ga, gb, q, qa, qb );
}

// ----- dot -----


std::tuple<Tensor,Tensor,Tensor> do_dot( Tensor &a, Tensor& b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    DotOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    op1.grad = xt::ones_like( op1.output );
    op1.backward();
    
    return std::make_tuple( op1.output, va.grad, vb.grad );
}

TEST(MyTestCase, TestDot) 
{
    Tensor a = xt::arange<fprec>( 1.,5. );
    Tensor b = xt::arange<fprec>( 1.,5. );
    
    Tensor q  = { 30.};
    Tensor qa = { 1.,  2.,  3., 4. };
    Tensor qb = { 1.,  2.,  3., 4. };
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_dot( a, b );

    expect_eq3( "dot  ", z, ga, gb, q, qa, qb );
}

// ----- MatMul -----

std::tuple<Tensor,Tensor,Tensor> do_matmul( Tensor &a, Tensor& b )
{
    VariableTensor va(a);
    VariableTensor vb(b);
    
    MatMulOp  op1;
    op1.set_inputs( &va );
    op1.set_inputs( &vb );
    op1.forward();
    
    op1.grad = xt::ones_like( op1.output );
    op1.backward();
    
    return std::make_tuple( op1.output, va.grad, vb.grad );
}

// matmul : matrix x matrix
TEST(MyTestCase, TestMatMul1) 
{
    Tensor a = xt::arange<fprec>(1.,13.).reshape( {3,4} );
    Tensor b = xt::arange<fprec>(1.,13.).reshape( {4,3} );
    
    Tensor q  = {{  70.,   80.,   90.},
                 { 158.,  184.,  210.},
                 { 246.,  288.,  330.}};
    Tensor qa = {{  6.,  15.,  24.,  33.},
                 {  6.,  15.,  24.,  33.},
                 {  6.,  15.,  24.,  33.}};
    Tensor qb = {{ 15.,  15.,  15.},
                 { 18.,  18.,  18.},
                 { 21.,  21.,  21.},
                 { 24.,  24.,  24.}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : vector x vector
TEST(MyTestCase, TestMatMul2) 
{
    Tensor a = xt::arange<fprec>( 1.,4. );
    Tensor b = xt::arange<fprec>( 1.,4. );
    
    Tensor q  = { 14.};
    Tensor qa = { 1.,  2.,  3.};
    Tensor qb = { 1.,  2.,  3.};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );

    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : vector x matrix
TEST(MyTestCase, TestMatMul3) 
{
    Tensor a = xt::arange<fprec>(1.,4.).reshape( {3} );
    Tensor b = xt::arange<fprec>(1.,13.).reshape( {3,4} );
    
    Tensor q  = {{ 38.,  44.,  50.,  56.}};
    Tensor qa = {{ 10.,  26.,  42.}};
    Tensor qb = {{ 1.,  1.,  1.,  1.},
                 { 2.,  2.,  2.,  2.},
                 { 3.,  3.,  3.,  3.}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : matrix x vector
TEST(MyTestCase, TestMatMul4) 
{
    Tensor a = xt::arange<fprec>(1.,13.).reshape( {3, 4} );
    Tensor b = xt::arange<fprec>(1.,5.).reshape( {4} );
    
    Tensor q  = { 30., 70., 110. };
    q.reshape( { 3,1 } );
    Tensor qa = {{ 1.,  2.,  3.,  4.},
                 { 1.,  2.,  3.,  4.},
                 { 1.,  2.,  3.,  4.}};
    Tensor qb = { 15.,  18.,  21.,  24.};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );

    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : vector x batched tensor
TEST(MyTestCase, TestBroadcastMatMul1) 
{
    Tensor a = xt::arange<fprec>(1.,4.).reshape( {3} );
    Tensor b = xt::arange<fprec>(1.,25.).reshape( {2, 3, 4} );
    
    Tensor q  = {{{  38.,   44.,   50.,   56.}},
                 {{ 110.,  116.,  122.,  128.}}};
    Tensor qa = {{  68.,  100.,  132.}};
    Tensor qb = {{{ 1.,  1.,  1.,  1.},
                  { 2.,  2.,  2.,  2.},
                  { 3.,  3.,  3.,  3.}},
                 {{ 1.,  1.,  1.,  1.},
                  { 2.,  2.,  2.,  2.},
                  { 3.,  3.,  3.,  3.}}};
    
    //cout<<"a"<<a<<endl;
    //cout<<"b"<<b<<endl;
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched tensor x vector
TEST(MyTestCase, TestBroadcastMatMul2) 
{
    Tensor a = xt::arange<fprec>(1.,25.).reshape( {2, 3, 4} );
    Tensor b = xt::arange<fprec>(1.,5.).reshape( {4} );
    
    Tensor q  = { 30., 70., 110., 150., 190., 230.};
    q.reshape( { 2,3,1 } );
    Tensor qa = {{{ 1.,  2.,  3.,  4.},
                  { 1.,  2.,  3.,  4.},
                  { 1.,  2.,  3.,  4.}},
                 {{ 1.,  2.,  3.,  4.},
                  { 1.,  2.,  3.,  4.},
                  { 1.,  2.,  3.,  4.}}};
    Tensor qb = {{ 66.,  66.,  66.,  66.},
                 { 72.,  72.,  72.,  72.},
                 { 78.,  78.,  78.,  78.},
                 { 84.,  84.,  84.,  84.}};

    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched matrix x matrix
TEST(MyTestCase, TestBroadcastMatMul3) 
{
    Tensor a = xt::arange<fprec>(1.,25.).reshape( {2, 3, 4} );
    Tensor b = xt::arange<fprec>(1.,9.).reshape( {4, 2} );
    
    Tensor q  = {{{  50.,   60.},
                  { 114.,  140.},
                  { 178.,  220.}},
                 {{ 242.,  300.},
                  { 306.,  380.},
                  { 370.,  460.}}};
    Tensor qa = {{{  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.}},
                 {{  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.}}};
    Tensor qb = {{ 66.,  66.},
                 { 72.,  72.},
                 { 78.,  78.},
                 { 84.,  84.}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

 // matmul : matrix x batched matrix
TEST(MyTestCase, TestBroadcastMatMul4) 
{
    Tensor a = xt::arange<fprec>(1.,13.).reshape( {3, 4} );
    Tensor b = xt::arange<fprec>(1.,25.).reshape( {2, 4, 3} );
    
    Tensor q  = {{{  70.,   80.,   90.},
                  { 158.,  184.,  210.},
                  { 246.,  288.,  330.}},
                 {{ 190.,  200.,  210.},
                  { 470.,  496.,  522.},
                  { 750.,  792.,  834.}}};
    Tensor qa =  {{  48.,   66.,   84.,  102.},
                  {  48.,   66.,   84.,  102.},
                  {  48.,   66.,   84.,  102.}};
    Tensor qb = {{{ 15.,  15.,  15.},
                  { 18.,  18.,  18.},
                  { 21.,  21.,  21.},
                  { 24.,  24.,  24.}},
                 {{ 15.,  15.,  15.},
                  { 18.,  18.,  18.},
                  { 21.,  21.,  21.},
                  { 24.,  24.,  24.}}};

    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched matrix x batched matrix
TEST(MyTestCase, TestBroadcastMatMul5) 
{
    Tensor a = xt::arange<fprec>(1.,25.).reshape( {2, 3, 4} );
    Tensor b = xt::arange<fprec>(1.,17.).reshape( {2, 4, 2} );
    
    Tensor q  = {{{   50.,    60.},
                  {  114.,   140.},
                  {  178.,   220.}},
                 {{  706.,   764.},
                  {  898.,   972.},
                  { 1090.,  1180.}}};
    Tensor qa = {{{  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.},
                  {  3.,   7.,  11.,  15.}},
                 {{ 19.,  23.,  27.,  31.},
                  { 19.,  23.,  27.,  31.},
                  { 19.,  23.,  27.,  31.}}};
    Tensor qb = {{{ 15.,  15.},
                  { 18.,  18.},
                  { 21.,  21.},
                  { 24.,  24.}},
                 {{ 51.,  51.},
                  { 54.,  54.},
                  { 57.,  57.},
                  { 60.,  60.}}};
    
    //cout<<"a"<<a<<endl;
    //cout<<"b"<<b<<endl;
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched matrix x batched matrix
TEST(MyTestCase, TestBroadcastMatMul6) 
{
    Tensor a = xt::arange<fprec>(1.,19.).reshape( {2,1,3,3} );
    Tensor b = xt::arange<fprec>(1.,19.).reshape( {  2,3,3} );
    
    Tensor q  = {{{{  30.,   36.,   42.},
                   {  66.,   81.,   96.},
                   { 102.,  126.,  150.}},
                  {{  84.,   90.,   96.},
                   { 201.,  216.,  231.},
                   { 318.,  342.,  366.}}},
                 {{{ 138.,  171.,  204.},
                   { 174.,  216.,  258.},
                   { 210.,  261.,  312.}},
                  {{ 435.,  468.,  501.},
                   { 552.,  594.,  636.},
                   { 669.,  720.,  771.}}}};
    Tensor qa = {{{{ 39.,  57.,  75.},
                   { 39.,  57.,  75.},
                   { 39.,  57.,  75.}}},
                 {{{ 39.,  57.,  75.},
                   { 39.,  57.,  75.},
                   { 39.,  57.,  75.}}}};
    Tensor qb = {{{ 51.,  51.,  51.},
                  { 57.,  57.,  57.},
                  { 63.,  63.,  63.}},
                 {{ 51.,  51.,  51.},
                  { 57.,  57.,  57.},
                  { 63.,  63.,  63.}}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
 
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched matrix x batched matrix
TEST(MyTestCase, TestBroadcastMatMul7) 
{
    Tensor a = xt::arange<fprec>(1.,37.).reshape( {3,1,3,4} );
    Tensor b = xt::arange<fprec>(1.,17.).reshape( {2,4,2} );
    
    Tensor q  = {{{{   50.,    60.},
                   {  114.,   140.},
                   {  178.,   220.}},
                  {{  130.,   140.},
                   {  322.,   348.},
                   {  514.,   556.}}},
                 {{{  242.,   300.},
                   {  306.,   380.},
                   {  370.,   460.}},
                  {{  706.,   764.},
                   {  898.,   972.},
                   { 1090.,  1180.}}},
                 {{{  434.,   540.},
                   {  498.,   620.},
                   {  562.,   700.}},
                  {{ 1282.,  1388.},
                   { 1474.,  1596.},
                   { 1666.,  1804.}}}};
    Tensor qa = {{{{ 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.}}},
                 {{{ 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.}}},
                 {{{ 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.},
                   { 22.,  30.,  38.,  46.}}}};
    Tensor qb = {{{ 153.,  153.},
                  { 162.,  162.},
                  { 171.,  171.},
                  { 180.,  180.}},
                 {{ 153.,  153.},
                  { 162.,  162.},
                  { 171.,  171.},
                  { 180.,  180.}}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}

// matmul : batched matrix x batched matrix
TEST(MyTestCase, TestBroadcastMatMul8) 
{
    Tensor a = xt::arange<fprec>(1.,49.).reshape( {2,2,3,4} );
    Tensor b = xt::arange<fprec>(1.,33.).reshape( {2,2,4,2} );
    
    Tensor q  = {{{{   50.,    60.},
                   {  114.,   140.},
                   {  178.,   220.}},
                  {{  706.,   764.},
                   {  898.,   972.},
                   { 1090.,  1180.}}},
                 {{{ 2130.,  2236.},
                   { 2450.,  2572.},
                   { 2770.,  2908.}},
                  {{ 4322.,  4476.},
                   { 4770.,  4940.},
                   { 5218.,  5404.}}}};
    Tensor qa = {{{{  3.,   7.,  11.,  15.},
                   {  3.,   7.,  11.,  15.},
                   {  3.,   7.,  11.,  15.}},
                  {{ 19.,  23.,  27.,  31.},
                   { 19.,  23.,  27.,  31.},
                   { 19.,  23.,  27.,  31.}}},
                 {{{ 35.,  39.,  43.,  47.},
                   { 35.,  39.,  43.,  47.},
                   { 35.,  39.,  43.,  47.}},
                  {{ 51.,  55.,  59.,  63.},
                   { 51.,  55.,  59.,  63.},
                   { 51.,  55.,  59.,  63.}}}};
    Tensor qb = {{{{  15.,   15.},
                   {  18.,   18.},
                   {  21.,   21.},
                   {  24.,   24.}},
                  {{  51.,   51.},
                   {  54.,   54.},
                   {  57.,   57.},
                   {  60.,   60.}}},
                 {{{  87.,   87.},
                   {  90.,   90.},
                   {  93.,   93.},
                   {  96.,   96.}},
                  {{ 123.,  123.},
                   { 126.,  126.},
                   { 129.,  129.},
                   { 132.,  132.}}}};
    
    Tensor z,ga,gb;
    std::tie(z,ga,gb) = do_matmul( a, b );
    
    expect_eq3( "matmul  ", z, ga, gb, q, qa, qb );
}


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

// broadcat tensors : A(3,3),B(3)
TEST(MyTestCase, TestBroadcastTensors1) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor q = {{ 0.,  1.,  2.},
                { 0.,  1.,  2.},
                { 0.,  1.,  2.}};
    
    auto z = do_broadcast( a, b );
    
    expect_eq1( "broadcast  ", z, q );
}

// broadcat tensors : A(3,3),B(3,1)
TEST(MyTestCase, TestBroadcastTensors2) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor c = xt::view( b, xt::all(), xt::newaxis() );
    Tensor q = {{ 0.,  0.,  0.},
                { 1.,  1.,  1.},
                { 2.,  2.,  2.}};
    
    auto z = do_broadcast( a, c );
    
    expect_eq1( "broadcast  ", z, q );
}

// broadcat tensors : D(4),B(3,1)
TEST(MyTestCase, TestBroadcastTensors3) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor b = xt::arange(3);
    Tensor c = xt::view( b, xt::all(), xt::newaxis() );
    Tensor d = xt::arange(4);
    Tensor q = {{ 0.,  1.,  2.,  3.},
                { 1.,  2.,  3.,  4.},
                { 2.,  3.,  4.,  5.}};
    
    auto z = do_broadcast( d, c );
    
    expect_eq1( "broadcast  ", z, q );
}

// broadcat tensors : A(3,3),B(1)
TEST(MyTestCase, TestBroadcastTensors4) 
{
    Tensor a = xt::zeros<fprec>({3, 3});
    Tensor e = (fprec)1.0;
    Tensor q = {{ 1.,  1.,  1.},
                { 1.,  1.,  1.},
                { 1.,  1.,  1.}};
    
    auto z = do_broadcast( a, e );
    
    expect_eq1( "broadcast  ", z, q );
}

// broadcat tensors : error - A(4,3),B(2,3)
TEST(MyTestCase, TestBroadcastTensors5) 
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
    Tensor qb ={{ 24.,  24.,  24.},
                { 30.,  30.,  30.},
                { 36.,  36.,  36.},
                { 41.,  41.,  41.}};
    /*Tensor qb={{{ 12.,  12.,  12.},
                { 15.,  15.,  15.},
                { 18.,  18.,  18.},
                { 21.,  21.,  21.}},
               {{ 12.,  12.,  12.},
                { 15.,  15.,  15.},
                { 18.,  18.,  18.},
                { 20.,  20.,  20.}}};*/
                
    expect_eq3( "batch ",z, ga, gb, q, qa, qb );
}