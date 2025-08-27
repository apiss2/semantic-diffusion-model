# 概要
このリポジトリは Semantic Image Synthesis via Diffusion Models (SDM) の個人的に改良したものです。

- [参考論文](https://arxiv.org/abs/2207.00050)
- [公式実装](https://github.com/WeilunWang/semantic-diffusion-model)


# 環境構築
`uv sync`で基本的に動作する

pytorchのバージョンに関しては固定してしまっているので適宜変更する必要がある

# 使い方
configはデフォルト値。環境変数に書き込むことで数値を変更できる。
トップレベルの変数は大文字、ネストされた設定は各サブconfigの名前に`__`を付けてアクセスできる。

`.env`の例を以下に示す
```
IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_CLASSES = 19
GRAYSCALE = FALSE

dataset__image_dir = "path/to/image_dir"
dataset__label_dir = "path/to/label_dir"
```

