# MiniCTorch_Prototype


## Requirements:
The following four header only libraries
```
cd MiniCTorch_Prototype

git clone https://github.com/xtensor-stack/xtensor.git
git clone https://github.com/xtensor-stack/xtensor-blas.git
git clone https://github.com/xtensor-stack/xtl.git
```

## Installation

```
pip install .
```

## Example (including unknown operators)
```
python example.py
```

出力ファイル：
- `output/example.cpp`:計算グラフ本体
- `output/example_param.cpp`:パラメータ初期値
- `output/example_data.cpp`:学習データ
- `output/example_train.cpp`:学習プログラム
- `output/Makefile`

## More examples
https://github.com/kojima-r/MiniCTorch_Benchmark

## 実装済み演算
https://docs.google.com/spreadsheets/d/1xPFaXAceqH8FPTJTFEfQup7KIxAGWHLt/edit?usp=sharing&ouid=108859876331580908917&rtpof=true&sd=true

## Compile
```
cd output/
make
```
この例では学習できるパラメータを持つ計算グラフではないので，`./example_train`の生成に関しては失敗する


推定の実行
```
./example
```


## 抽出の仕組み：Python Notebook
Pytorchから計算グラフを抜き出す部分(`network/example01.json`を作成する部分)

https://drive.google.com/file/d/18vHBaYnoydmi2MQZzksFSY2xJ-5xHcNn/view?usp=sharing
