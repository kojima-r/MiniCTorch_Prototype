
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 0.66135216,0.2669241,0.06167726,0.6213173,-0.45190597,-0.16613023,-1.5227685,0.38168392,
                        -1.0276086,-0.5630528,-0.89229053,-0.058250178,-0.19550958,-0.96563596,0.42241532, };
    
    // {'name': 'Net/4', 'op': 'prim::Constant', 'in': [], 'shape': [3], 'constant_value': [2.0, 3.0, 2.0], 'out': [5], 'sorted_id': 1}
    
    Tensor Constant1 ={ 2.0,3.0,2.0, };
    