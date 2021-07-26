
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    Tensor  xin;
    Tensor  Constant1;
    
    void LoadParameter()
    {
        // input data
        
        xin          ={ 1.0,2.0,3.0, };
        xin.reshape({1,3});
        
        // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [2, 3], 'constant_value': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'out': [2], 'sorted_id': 1}
        
        Constant1    ={ 5.0,6.0,7.0,8.0,9.0,10.0, };
        Constant1.reshape({2,3});
        
    }
    