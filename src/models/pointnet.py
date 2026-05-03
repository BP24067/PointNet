import torch
import torch.nn as nn
import torch.optim as optim

#非線形化
class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels#入力の特徴量
        self.output_channels = output_channels#出力の特徴量

        self.main = nn.Sequential(#非線形化までの一連の流れを定義
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_channels))

    def forward(self, input_data):
        return self.main(input_data)
#MaxPooling（順不変性の獲得）
class MaxPool(nn.Module):
    def __init__(self,num_channels,num_points):
        super().__init__()
        self.num_channels = num_channels#次元の数の指定
        self.num_points = num_points#検索範囲の指定
        self.main = nn.MaxPool1d(self.num_points)#入力されたバッチに対して各特徴量を最大のものを選び一つに統合する

    def forward(self,input_data):
        out = input_data.view(-1,self.num_channels,self.num_points)
        out = self.main(out)
        out = out.view(-1,self.num_channels)
        return out

#Tnet:座標データに対して
class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(#傾きの打ち消し
            NonLinear(3, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),#特徴量から、傾きを求める
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9)#傾きを打ち消すための逆算係数を求める（様々な変換のために活性化関数は用いない）
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 3, 3)#batchsize,変換行列
        #元の座標に対して、回転、縮小拡大、鏡映、スキューを一度に行う（平行移動は行わない）
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        out = out.view(-1, 3)#バッチサイズ、整えた座標を返す
        return out


#Tnet:特徴量に対して
class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out

class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonLinear(3, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),#Inputを高次元に行ってるようなもの、意味的には同じだが、環境による違いがあるものに対してその環境による違いをなくす方法
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels),
            )
    #PointNetのSeqで最初のパイプラインに引数が必要だから、引数を渡す必要がある
    def forward(self, input_data):
        return self.main(input_data)

#テストコード
if __name__ == "__main__":
    #ダミーデータ作成
    dummy_input = torch.randn(2*1024,3)
    #識別するクラス数
    model = PointNet(1024,num_labels=2)

    result = model(dummy_input)
    print('input',dummy_input.shape)
    print('output',result.shape)
