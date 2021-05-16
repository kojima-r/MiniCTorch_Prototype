# MiniCTorch_Prototype

## Python Notebook
Pytorchから計算グラフを抜き出す部分(`network/example01.json`を作成する部分)

https://colab.research.google.com/drive/1vDP4v6oUJop1t4ptFU65ylriYZnhzNCx?usp=sharing

## Dependency
The following four header only libraries
```
cd MiniCTorch_Prototype

git clone https://github.com/xtensor-stack/xtensor.git
git clone https://github.com/xtensor-stack/xtensor-blas.git
git clone https://github.com/xtensor-stack/xtl.git
git clone https://github.com/nlohmann/json.git
```

## Installation

```
pip install .
```
## Example (including unknown operators)
```
python example.py
minictorch_translator sample.json
```

出力ファイル：`src/example.gen.cpp`, `src/Makefile` 

## Compile
```
cd src/
make
```

## Example　(for operation verification)
インストール後に以下のコマンド入力
テスト用の計算グラフを変換
```
 minictorch_translator network/example01.json
```
テスト用の計算グラフ(C++)をコンパイル
```
cd src/
make
```
実行
```
./mini_c_torch
```
