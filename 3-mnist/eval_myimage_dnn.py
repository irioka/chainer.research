# -*- coding: utf-8 -*-
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

# DNN
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        w = I.Normal(scale=0.05)
        super(MLP, self).__init__(            
            l1=L.Linear(None, n_units, initialW=w),  # n_in -> n_units    入力については，"None"にしておくと入力に応じて自動的にノード数を定義してくれます．便利です．
            l2=L.Linear(None, n_units, initialW=w),  # n_units -> n_units
            l3=L.Linear(None, n_out, initialW=w),  # n_units -> n_out
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x)) # 活性化関数はReLUを利用
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def forward(self, x):
        h1 = F.relu(self.l1(x)) # 活性化関数はReLUを利用
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)        
"""
自分で用意した手書き文字画像をモデルに合うように変換する処理
"""
def convert_dnn(img):    
    data = np.array(Image.open(img).convert('L').resize((28, 28)), dtype=np.float32)  # ファイルを読込み，リサイズして配列に変換        
    data = (255.0 - data) / 255.0 # 白黒反転して正規化
    data = data.reshape(1, 784) # データの形状を変更
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
    img = convert_dnn(args.inputimage)
    x = chainer.Variable(np.asarray(img)) # 配列データをchainerで扱う型に変換
    
    y = model.predictor(x) # フォワード
    c = F.softmax(y).data.argmax()    
    print('判定結果は{}です。'.format(c))        

if __name__ == '__main__':
    main()
