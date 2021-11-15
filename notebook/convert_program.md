# C++コードへの変換プログラム

## JSONファイルからC++コードへの変換

convert_cpp_code( project, folder, model, input, json_path )

[引数]
 project  :  プロジェクト名 (各ファイルのヘッダーに相当）
 folder   :  生成するc++コードを保存するフォルダ (notebookからの相対アドレス)
 model    :  ニューラルネットのクラス
 inputs   :　ニューラルネットへの入力データ (forward関数の入力引数に相当)
 json_path : 計算グラフを保存するJSONファイル名  (notebookからの相対アドレスで指定)


## pythonの変数から学習用データファイルへの変換プログラム

convert_data_file( project, folder, **kwargs )

[引数]
 project : プロジェクト名 (ファイルのヘッダーに相当) 
 folder  : 生成するデータファイルを保存するフォルダ (notebookからの相対アドレス)
 kwargs  : 辞書による可変長引数

  inp_data    : 入力変数
  pred_data   : 入力変数2　(複数特徴量の入力変数に相当する。"mse1"を参照)
　target_data : 教師データの変数　(回帰での参照データ、分類での正解データ)


## 学習ループのC++コードの生成プログラム

convert_train_code( project, folder, **kwargs )

[引数]
 project : プロジェクト名 (ファイルのヘッダーに相当) 
 folder  : 生成するデータファイルを保存するフォルダ (notebookからの相対アドレス)
 kwargs  : 辞書による可変長引数

　 sol       : 解析タイプ ( "mse"(回帰), "cse"(分類), "vae"(変分オートエンコーダ) )
   epoch     : 学習の反復回数　(デフォルト200)
   lr        : 学習率          (デフォルト0.01)
   batch     : ミニバッチ数
　 inp_data  : 入力変数    (convert_data_fileと同じ)
   pred_data : 入力変数２　(convert_data_fileと同じ)

　"vae"専用
　z   : 潜在変数を探すキーワード　(潜在変数を入力とする全結合層の名前）

　　(sample)  
　　　vae2のnotebookではself.zが潜在変数にあたりますが、
　　　それを入力としている全結合則はfc3であるので、z="fc3" となります。
        :
　　　self.std = torch.exp( 0.5 * self.log_var )
      q_z = td.normal.Normal( self.mean, self.std )
      self.z = q_z.rsample()

      # decoder
      y = F.relu( self.fc3( self.z ) )
        :

# notebook一覧

  mse1 :   複数特徴量による回帰問題の例  [sol="mse1"]
   　参考：  https://axa.biopapyrus.jp/deep-learning/object-classification/regression-multiple-features.html

  mse2 :   sinxを近似する回帰問題（ミニバッチによる学習も含む）  [sol="mse"]
     参考：  https://watlab-blog.com/2021/06/14/pytorch-nonlinear-regression/

  cse1 :   skleranのirisモデルの分類問題 (１バッチサイズの分類モデル） [sol="cse1"]

　cse2 :   sklearnのirisモデルの分類モデル (ミニバッチによる全モデルの分類) [sol="cse"]

  vae1 :   変分オートエンコーダ [sol="vae"]
　　 KLdivergenceの評価を直接記述したもの（ミニバッチによる学習も含む）
　　 参考：我妻　「はじめてのディープラーニング２」

　vae2 :   変分オートエンコーダ [sol="vae"]
　　 KLdivergenceの評価をpytorchの評価関数を用いたもの（ミニバッチによる学習も含む）
　　 参考：我妻　「はじめてのディープラーニング２」

  test1 :  四則演算のテストプログラム
  
  test2 :  全結合層のテストプログラム
  
  test3 :  各種活性化関数のテストプログラム
  
  test_batch :  バッチタイプの行列演算のテストプログラム
  
  broadcast_check :　ブロードキャストのテストプログラム
  
  network_dot :  JSONファイルの計算グラフのgraphic_vizによる描画プログラム
  