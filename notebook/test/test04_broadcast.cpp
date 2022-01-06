
#include "minictorch.hpp"

void do_broadcast( Tensor &a, Tensor &b )
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
    
    cout<<"--------------------------------"<<endl;
    cout<<"a"<<a<<endl;
    cout<<"b"<<b<<endl;
    
    list1.forward();
    if( bcast1.forward() )
    {
        unpk0.forward();
        unpk1.forward();
        add1.forward();
        
        cout<<"a_mod"<<unpk0.output<<endl;
        cout<<"b_mod"<<unpk1.output<<endl;
        cout<<"a+b"<<add1.output<<endl;
    }
}

void _broadcast()
{
    Tensor a  = xt::zeros<fprec>({3, 3});
    Tensor b  = xt::arange(3);
    Tensor c  = xt::view( b, xt::all(), xt::newaxis() );
    Tensor d  = xt::arange(4);
    Tensor e  = (fprec)1.0;
    
    Tensor x = xt::zeros<fprec>( {4, 3} );
    Tensor y = xt::arange(6).reshape( {2, 3} );
    
    do_broadcast( a, b );
    do_broadcast( a, c );
    do_broadcast( d, c );
    do_broadcast( a, e );
    do_broadcast( x, y );
}

int main( int argc, char *argv[] )
{
   for (int i=1;i<argc;i+=2)
      printf("argv[%d] = %s %s\n", i, argv[i],argv[i+1]);
      
   _broadcast();
   
   return 1;
}