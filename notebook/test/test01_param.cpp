
    //
    //  test01_param
    //
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 1.0,2.0,3.0,4.0, };
    
    // {'name': 'Net/7', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [2, 2], 'constant_value': [1.0, 2.0, 3.0, 4.0], 'out': [5], 'sorted_id': 4}
    
    Tensor Constant1 ={ 1.0,2.0,3.0,4.0, };
    