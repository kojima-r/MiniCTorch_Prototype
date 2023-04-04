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

## 変換のためのメソッド（example.py）

> convert_cpp_code( project, folder, model, json_path, input_to_model, input_name_pair, input_data_dict)

[引数]
 - project  :  プロジェクト名 (各ファイルのヘッダーに相当）
 - path     :  生成するc++コードを保存するフォルダ (相対パス)
 - model    :  変換するニューラルネットのクラスオブジェクト（nn.Module）
 - json_path : 計算グラフを保存するJSONファイル名  (相対パスとファイル名で指定)
 - input_to_model   :　modelへの入力データ (forward関数の入力引数に相当, リストで複数指定可)
 - input_name_pair  :　変換するニューラルネットへの入力データ (「入力変数の名前」と「入力データの名前」のペア, リストで指定する)
 - input_data_dict  :　入力データのディクショナリ (キーが「入力データの名前」でバリューが実際のarrayデータとなるディクショナリで指定)
 
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
