
#include "minictorch.hpp"

using namespace std;

void print_shape(Tensor x, string ss )
{
    auto s=x.shape();
    cout<<ss<<" shape (";
    for(int i=0;i<s.size();i++){
        cout<<s[i]<<",";
    }
    cout<<")"<<endl;
}

void test01()
{
    Tensor a=
            {{{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}},
            {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}}};
    Tensor b=
            {{{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}},
            {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}}};
    Tensor c({2,3,3});
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                a[i, j,k] = 1;
                b[i, j,k] = 1;
            }
        }
    }
    c=a+b;
    cout<<"a"<<a<<endl;
    cout<<"b"<<b<<endl;
    cout<<"c"<<c<<endl;
    const auto& d = c.shape();
    cout << "Dim size: " << d.size() <<endl;
    //for(int i=0;i<d.size();i++){
    //    cout << ">>" << d[i] <<endl;
    //}
    print_shape( c, "c" );
    
    /////////////
    {
        cout<<"=="<<endl;
        VariableTensor va(a);
        VariableTensor vb(b);
        AddOp op_a_b;
        op_a_b.inputs.push_back(&va);
        op_a_b.inputs.push_back(&vb);
        op_a_b.forward();
        cout<<op_a_b.output<<endl;
        cout<<"=="<<endl;
        op_a_b.grad = xt::ones_like( op_a_b.output );
        op_a_b.backward();
        cout<<op_a_b.grad<<endl;
        cout<<va.grad<<endl;
        cout<<vb.grad<<endl;
    }
};

void test02()
{
    Tensor a=
            {{{1, 2, 3, 4},
            {4, 5, 6, 7},
            {7, 8, 9,10}},
            {{1, 2, 3, 4},
            {4, 5, 6,7},
            {7, 8, 9,9}}};
    Tensor b={{1, 2, 3},
            {4, 5, 6},
            {4, 5, 6},
            {7, 8, 9}};
    auto bb=xt::expand_dims(b,1);
    auto z=xt::repeat(bb,2,1);
    cout<<"b"<<b<<endl;
    cout<<"bb"<<bb<<endl;
    cout<<"z"<<z<<endl;
    {
        cout<<"=="<<endl;
        VariableTensor va(a);
        VariableTensor vb(b);
        MatMulOp op_a_b;
        op_a_b.inputs.push_back(&va);
        op_a_b.inputs.push_back(&vb);
        op_a_b.forward();
        cout<<"a"<<a<<endl;
        cout<<"b"<<b<<endl;
        cout<<"c"<<op_a_b.output<<endl;
        cout<<"=="<<endl;
        //op_a_b.grad=op_a_b.output;
        op_a_b.grad = xt::ones_like( op_a_b.output );
        op_a_b.backward();
        cout<<"va_grad"<<va.grad<<endl;
        cout<<"vb_grad"<<vb.grad<<endl;
    }
}

void test03()
{
    Tensor a1=
            {{{1, 2, 3, 4},
              {4, 5, 6, 7},
              {7, 8, 9,10}},
             {{1, 2, 3, 4},
              {4, 5, 6, 7},
              {7, 8, 9, 9}}};
            
    auto a2 = xt::amax( a1, {0} );
    auto a3 = xt::amax( a1, {1} );
    auto a4 = xt::amax( a1, {2} );
    auto a5 = xt::amax( a1 );
    
    cout<<"a1"<<a1<<endl;
    cout<<"a2"<<a2<<endl;
    cout<<"a3"<<a3<<endl;
    cout<<"a4"<<a4<<endl;
    cout<<"a5"<<a5<<endl;
    
    Tensor b1={{1, 2, 3, 4},
               {4, 5, 6, 7},
               {7, 8, 9,10}};
    auto b2 = xt::amax( b1, {0} );
    auto b3 = xt::amax( b1, {1} );
    auto b4 = xt::amax( b1 );
    cout<<"b1"<<b1<<endl;
    cout<<"b2"<<b2<<endl;
    cout<<"b3"<<b3<<endl;
    cout<<"b4"<<b4<<endl;
}

int main( int argc, char *argv[] )
{
    test02();
    
    return 0;
}