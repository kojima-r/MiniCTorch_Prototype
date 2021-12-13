# C++コードへの変換プログラム

## JSONファイルからC++コードへの変換

> convert_cpp_code( project, folder, model, input, json_path )

[引数]
 - project  :  プロジェクト名 (各ファイルのヘッダーに相当）
 - folder   :  生成するc++コードを保存するフォルダ (notebookからの相対アドレス)
 - model    :  ニューラルネットのクラス
 - inputs   :　ニューラルネットへの入力データ (forward関数の入力引数に相当)
 - json_path : 計算グラフを保存するJSONファイル名  (notebookからの相対アドレスで指定)


## pythonの変数から学習用データファイルへの変換プログラム

> convert_data_file( project, folder, **kwargs )

[引数]
 - project : プロジェクト名 (ファイルのヘッダーに相当) 
 - folder  : 生成するデータファイルを保存するフォルダ (notebookからの相対アドレス)
 - kwargs  : 辞書による可変長引数
   - inp_data    : 入力データ配列
   - target_data : 教師データ配列　(回帰での参照データ、分類での正解データに相当する)


## 学習ループのC++コードの生成プログラム

> convert_train_code( project, folder, **kwargs )

[引数]
 - project : プロジェクト名 (ファイルのヘッダーに相当) 
 - folder  : 生成するデータファイルを保存するフォルダ (notebookからの相対アドレス)
 - kwargs  : 辞書による可変長引数
　 - sol         : 解析タイプ (文字："regr"(回帰), "clas"(分類), "vae"(変分オートエンコーダ))
   - epoch       : 学習の反復回数　(正整数：デフォルト200)
   - lr          : 学習率          (正実数：デフォルト0.01)
   - batch       : ミニバッチ数　 （正整数：デフォルト32)
　 - inp_data    : 入力データ配列　(convert_data_fileと同じ)
   - target_data : 参照データ配列　(convert_data_fileと同じ)
   - net_key     : Netクラスのクラス名　　(文字：pythonコードのNetクラスに相当する)
   - loss_key    : Lossクラスのクラス名 　(文字：pythonコードのLossクラスに相当する)
   - pred_key    : 予測変数を求める計算グラフのキーワード(文字)
   - pred_no     : 予測変数を求める計算グラフ番号（正整数)
   - pred_num    : 予測変数の後処理する数 (正整数)
   
#### Example
 ```
     inp_data=d1,net_key="Net",loss_key="loss",pred_key="sigmoid",pre_no=38,pred_num=10
 ```

　"vae"専用
  - z   : 潜在変数を探すキーワード　(潜在変数を入力とする全結合層の名前）

#### Example
　　　vae2のnotebookではself.zが潜在変数にあたりますが、
　　　それを入力としている全結合則はfc3であるので、z="fc3" となります。
```
        :
      self.std = torch.exp( 0.5 * self.log_var )
      q_z = td.normal.Normal( self.mean, self.std )
      self.z = q_z.rsample()

      # decoder
      y = F.relu( self.fc3( self.z ) )
        :
```

# notebook一覧

## example_regression
sin(x)を近似する回帰問題のコード（ミニバッチによる学習も含む）  [sol="regr"]
 
>    損失関数として、torch.nn.MSELoss を採用している。
```     
     　　　MSELoss( y, t )
```  
- y:評価値
- t:参照値(教師データ)

## example_classification
sklearnのirisモデルの分類問題のコード　(ミニバッチによる全モデルの分類) [sol="clas"]
　
>　　損失関数として、torch.nn.CrossEntropyLoss を採用している。
```　　
　　　　　 CrossEntropyLoss( y, t )
```
- y:評価値
- t:参照値(教師データ)

## example_vae_td
変分オートエンコーダ [sol="vae"]

>   （注）Cross_Entropy,KLdivergenceをpytorchの組み込み関数で記述したコード（ミニバッチによる学習も含む）
>　 損失関数は交差エントロピー(e1)とKLダイバージェンス(e2)の和で評価している。
   
```
　　 　 e1  = F.binary_cross_entropy( y , t, reduction="sum" )
        p_z = td.normal.Normal( torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale) )
        e2  = td.kl_divergence( q_z, p_z ).sum()
        loss = e1+e2
```

但し、y:評価値  t: 参照値　 q_z:計算されて平均、標準偏差による正規分布


## example_vae
変分オートエンコーダ [sol="vae"]

>    （注）交差エントロピー,KLダイバージェンスを自作関数で記述したコード（ミニバッチによる学習も含む）
        
# 以下のコードは開発に使用したサンプルコードです。
　
## test01
四則演算のコード
  
## test02
全結合層のコード
  
## test03
各種活性化関数のコード
  
## test_batch
バッチタイプの行列演算のコード
  
## broadcast_check
ブロードキャストのコード
  
## network_dot
JSONファイルの計算グラフのgraphic_vizによる描画コード

