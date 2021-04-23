# MiniCTorch_Prototype

## Python Notebook
Pytorchから計算グラフを抜き出す部分(`network/example01.json`を作成する部分)

https://colab.research.google.com/drive/1vDP4v6oUJop1t4ptFU65ylriYZnhzNCx?usp=sharing

## Dependency
The following four header only libraries
```
git clone https://github.com/xtensor-stack/xtensor.git
git clone https://github.com/xtensor-stack/xtensor-blas.git
git clone https://github.com/xtensor-stack/xtl.git
git clone https://github.com/nlohmann/json.git
```

## Installation

```
pip install .
```
## Example
```
python example.py
minictorch_translator sample.json
```

出力ファイル：`src/example.gen.cpp`

## Compile
```
g++ -std=c++14 -I./json/include -I./xtensor-blas/include -I./xtensor/include -I./xtl/include  main.cpp -lcblas
```
