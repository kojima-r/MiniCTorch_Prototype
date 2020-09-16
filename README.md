# MiniCTorch_Prototype

## Dependency
The following four header only libraries
```
git clone https://github.com/xtensor-stack/xtensor.git
git clone https://github.com/xtensor-stack/xtensor-blas.git
git clone https://github.com/xtensor-stack/xtl.git
git clone https://github.com/nlohmann/json.git
```

## Compile
```
g++ -std=c++14 -I./json/include -I./xtensor-blas/include -I./xtensor/include -I./xtl/include  main.cpp -lcblas
```
