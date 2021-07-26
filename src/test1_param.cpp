
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    Tensor  xin;
    Tensor  Constant1;
    
    void LoadParameter()
    {
        // input data
        
        xin          ={ 1.0,2.0,3.0,4.0, };
        xin.reshape({2,2});
        
        // {'name': 'Net/7', 'op': 'prim::Constant', 'in': [], 'shape': [2, 2], 'constant_value': [1.0, 2.0, 3.0, 4.0], 'out': [5], 'sorted_id': 4}
        
        Constant1    ={ 1.0,2.0,3.0,4.0, };
        Constant1.reshape({2,2});
        
    }
    