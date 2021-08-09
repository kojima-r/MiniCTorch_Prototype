
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 1.0,2.0,3.0, };
    
    // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [2, 3], 'constant_value': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'out': [2], 'sorted_id': 1}
    
    Tensor Constant1 ={ 5.0,6.0,7.0,8.0,9.0,10.0, };
    