# textbook
https://zenn.dev/yukiyada/articles/59f3b820c52571
https://github.com/YadaYuki/en_ja_translator_pytorch

# pytorch
nn.module:
- https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/1_Learning%20PyTorch/1_3_neural_networks_tutorial_jp.ipynb

# transformer
https://qiita.com/jun40vn/items/9135bf982d73d9372238
* Query: 計算ターゲット (I)
* Key: 類似度の計算に使う単語ベクトルの集まり（[吾輩,は,猫,で,ある]
* Value: 重み付け和に使うベクトル(ゼロ作ではQueryと同じ)
https://qiita.com/halhorn/items/c91497522be27bde17ce
https://zenn.dev/skypenguins/articles/a7b4cfee63b67c
* selective-second-order-with-skips - 直前の単語とそれ以前の単語一個ずつの2単語のみに注目
* 複数単語の特徴量を出す -> 入力が単語、出力が組み合わせとなるNNで学習
* 位置エンコーディングで二単語間の距離を反映する。同じ単語でも距離によって特徴量が変わる（あまりよくわかってない）

https://qiita.com/shingo_morita/items/43000991d04413136d62
