import torch
import torch.nn as nn
import models.pointnet as PN
from datasets.dataset import PointNetDataset
from torch.utils.data import DataLoader

#GPUを使うように設定（GPUがない場合はCPUを使用）
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#データセットの呼び出し
dataset = PointNetDataset(root_dir="data/train")
#データローダーの作成
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#モデルの呼び出し
model = PN.PointNet(num_points=1024, num_labels=2).to(device)#GPUへ転送
#損失関数
criterion = nn.CrossEntropyLoss() 
#最適化手法
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 

num_epochs = 50
#データに対して、バッチ分取り出す
for epoch in range(num_epochs):
    for i,(data,label) in enumerate(train_loader):
        # データもGPUに転送し、モデルとデータをGPUで計算できるようにする
        data = data.to(device)
        label = label.to(device)
        #データの関係でエラーがでるので、データの形を変える必要がある
        data = data.view(-1, 3)#1バッチで使うデータを並べる
        #勾配の初期化
        optimizer.zero_grad()
        #順伝播
        outputs = model(data)
        #損失の計算
        loss = criterion(outputs, label)
        #逆伝搬
        loss.backward()
        #パラメータの更新
        optimizer.step()

        print(f"Epoch[{epoch+1}/{num_epochs}],Step[{i+1}],loss:{loss.item():.4f}")
        
#モデルの保存
torch.save(model.state_dict(), "pointnet_model.pth")
print('学習完了！ モデルを保存しました。')
#1エポックにつき n回のステップ（更新）」が行われ、
#それが m周（エポック）繰り返される ため、合計で nm回 モデルの重みが更新される