
    //
    //  test04-2_param
    //
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 0.0,1.0,2.0, };
    
    // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [3, 3], 'constant_value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'out': [3], 'sorted_id': 0}
    
    Tensor Constant1 ={ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0, };
    