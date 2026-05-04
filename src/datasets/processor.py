import glob
import trimesh
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# 読み込むファイルのパスを指定
# ルートディレクトリから書いた方がいいらしい、
class PointNetDataset(Dataset):
    def __init__(self, root_dir,num_points=1024):
        
        self.root_dir = root_dir
        self.num_points = num_points
        # root_dir以下のすべてのplyファイルを取得
        self.files = glob.glob(os.path.join(root_dir, "*/*.ply"))#"*/*.ply"はそれ以下のすべての対象の拡張子を調べるという意味
        #ラベルを定義
        self.categories = {"cube": 0, "sphere": 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """指定されたインデックスのデータを取得するメソッド
        ラベル、テンソルを返す、データのバイアス、スケーリングを行う"""
        file_path = self.files[idx]
        # ファイルの親ディレクトリ名をカテゴリとする
        category = os.path.basename(os.path.dirname(file_path)) 
        label  = self.categories[category]#フォルダ名からラベルを取得
        # ファイルを読み込む
        mesh = trimesh.load(file_path)
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        # 正規化
        points -= np.mean(points, axis=0)  # バイアスを引く
        #最大の距離が 1 になるようにスケーリング
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
        points /= furthest_distance
        #Numpyからテンソルに変換
        tensor = torch.from_numpy(points).float()
        return tensor, label

if __name__ == "__main__":
    dataset = PointNetDataset(root_dir="data/train")
    print(f"データセット作成完了！ 合計数: {len(dataset)}")
    
    # 実際に1つ取り出してみる,読み込み確認
    if len(dataset) > 0:
        sample_data, sample_label = dataset[0]
        print(f"データの形: {sample_data.shape}") #1024,3
        print(f"ラベル: {sample_label}")           #01