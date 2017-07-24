# -*- coding: utf-8 -*-
"""MNIST CNN Example."""

from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
from PIL import Image
import numpy as np


class MLP(chainer.Chain):
    """ニューラルネットワークの定義"""

    def __init__(self, n_units, n_out):
        w = I.Normal(scale=0.05)  # モデルパラメータの初期化
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                1, 16, 5, 1, 0)      # 1層目の畳み込み層（フィルタ数は16）
            self.conv2 = L.Convolution2D(
                16, 32, 5, 1, 0)     # 2層目の畳み込み層（フィルタ数は32）
            self.l3 = L.Linear(None, n_out, initialW=w)  # クラス分類用

    def __call__(self, x):
        # 最大値プーリングは2×2，活性化関数はReLU
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2)
        y = self.l3(h2)
        return y


def convert_cnn(img):
    """
    自分で用意した手書き文字画像をモデルに合うように変換する処理
    """

    data = np.array(Image.open(img).convert('L').resize(
        (28, 28)), dtype=np.float32)  # ファイルを読込み，リサイズして配列に変換
    data = (255.0 - data) / 255.0  # 白黒反転して正規化
    data = data.reshape(1, 1, 28, 28)  # データの形状を変更
    return data


def main():
    # オプション処理
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--inputimage', '-i', default='',
                        help='入力画像ファイル')
    parser.add_argument('--model', '-m', default='',
                        help='CNNモデルで評価する')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('自分の手書き文字を学習したモデルで評価してみるプログラム')
    print('# 入力画像ファイル: {}'.format(args.inputimage))
    print('# 学習済みモデルファイル: {}'.format(args.model))
    print('')

    # モデルのインスタンス作成
    model = L.Classifier(MLP(args.unit, 10))
    # モデルの読み込み
    chainer.serializers.load_npz(args.model, model)

    # 入力画像を28x28のグレースケールデータ（0〜1に正規化）に変換する
    img = convert_cnn(args.inputimage)
    x = chainer.Variable(np.asarray(img))  # 配列データをchainerで扱う型に変換

    y = model.predictor(x)  # フォワード
    c = F.softmax(y).data.argmax()
    print('判定結果は{}です。'.format(c))


if __name__ == '__main__':
    main()
