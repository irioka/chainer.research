# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions


class LogicCircuit(chainer.Chain):
    def __init__(self):
        w = I.Normal(scale=1.0)  # モデルパラメータの初期化（平均0，分散1の分布に従う）
        super(LogicCircuit, self).__init__(
            l1=L.Linear(None, 2, initialW=w, initial_bias=0.5),
        )

    def __call__(self, x):
        y = self.l1(x)
        return y


def main():

    epoch = 20
    batchsize = 4

    # データの作成
    trainx = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=np.float32)
    trainy = np.array([0, 0, 0, 1], dtype=np.int32)
    train = chainer.datasets.TupleDataset(trainx, trainy)
    test = chainer.datasets.TupleDataset(trainx, trainy)

    # ニューラルネットワークの登録
    model = L.Classifier(LogicCircuit(), lossfun=F.softmax_cross_entropy)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # イテレータの定義
    train_iter = chainer.iterators.SerialIterator(train, batchsize)  # 学習用
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)  # 評価用

    # アップデータの登録
    # アップデータはミニバッチ毎のデータから求めた損失値を使って，誤差逆伝播とパラメータの更新を行う機構
    updater = training.StandardUpdater(train_iter, optimizer)
    # トレーナーの登録
    # トレーナーは学習を実行するところ
    # 学習とは，1．入力データが各層を伝搬していき出力値を計算，2．出力値と教師データから損失（誤差）を計算，3．誤差を逆伝播させ誤差が小さくなるように，パラメータを更新，という一連の流れを言う
    # 1データ毎にこれをやると効率が悪いので，ある程度のデータの塊（これをミニバッチ）で行う．例えばミニバッチサイズ100だと，100個のデータの誤差の平均値を基にパラメータを更新する．
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # 学習状況の表示や保存
    trainer.extend(extensions.Evaluator(test_iter, model))  # エポック数の表示
    trainer.extend(extensions.dump_graph('main/loss'))  # ニューラルネットワークの構造
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))  # 誤差のグラフ
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))  # 精度のグラフ
    trainer.extend(extensions.LogReport())  # ログ
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))  # 計算状態の表示
    # エポック毎にトレーナーの状態（モデルパラメータも含む）を保存する（なんらかの要因で途中で計算がとまっても再開できるように）
    trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))

#    chainer.serializers.load_npz("result/snapshot_iter_500", trainer)

    # 学習開始
    trainer.run()

    # 途中状態の保存
    chainer.serializers.save_npz("result/AND.model", model)

    # 学習結果の評価
    for i in range(len(trainx)):
        x = chainer.Variable(trainx[i].reshape(1, 2))
        result = F.softmax(model.predictor(x))
        print("input: {}, result: {}".format(trainx[i], result.data.argmax()))


if __name__ == '__main__':
    main()
