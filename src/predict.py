import torch
import trimesh
import numpy as np
import os
from models.pointnet import PointNet

# モデルの呼び出し
model = PointNet(num_points=1024, num_labels=2)

# 重みのロード
model.load_state_dict(torch.load('pointnet_model.pth'))
# 評価モードevalでの実行(すべてのニューロンが有効になる)
model.eval()
# 未知のデータを読み込む
# ファイルの選択
test_path = 'data/raw/キューブ.ply'
# ファイルを読み込む(datasetと同じ前処理を行う必要がある)
mesh = trimesh.load(test_path)
points, _ = trimesh.sample.sample_surface(mesh, model.num_points)
# 正規化
points -= np.mean(points, axis=0)  # バイアスを引く
#最大の距離が 1 になるようにスケーリング
furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
points /= furthest_distance
#Numpyからテンソルに変換(バッチからならview(-1, 3)で変形する必要あり)
tensor = torch.from_numpy(points).float()

# 推論
# 自動微分の無効化、推論の実行
with torch.no_grad():
    output = model(tensor)
    # outputの中で一番大きな値を正解とする
    prediction = torch.argmax(output, dim=1).item()
    output = torch.softmax(output, dim=1).numpy()
# 実行結果
labels = {0: "立方体 (Cube)", 1: "球体 (Sphere)"}
print(f"テストファイル: {test_path}")
print(f"出力(確率): {output}")
print(f"AIの予測: {labels[prediction]}")