
    //
    //  test05_param
    //
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,
                        6.0,7.0,8.0,9.0, };
    
    // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'output_id': 0, 'shape': [2, 3, 4], 'constant_value': [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0, 9.0], 'out': [2], 'sorted_id': 0}
    
    Tensor Constant1 ={ 1.0,2.0,3.0,4.0,4.0,5.0,6.0,7.0,
                        7.0,8.0,9.0,10.0,1.0,2.0,3.0,4.0,
                        4.0,5.0,6.0,7.0,7.0,8.0,9.0,9.0, };
    