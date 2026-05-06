import glob
import trimesh
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from trimesh.transformations import shear_matrix

# 読み込むファイルのパスを指定
# ルートディレクトリから書いた方がいいらしい
#より複雑なノイズにも対応できるように、trainの時に、Jittering,flip,shearの処理を追加する
class PointNetDataset(Dataset):
    def __init__(self, root_dir,num_points=1024,is_train=True):
        
        self.root_dir = root_dir
        self.num_points = num_points
        self.is_train = is_train
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
        # データ拡張（学習時のみ）
        if self.is_train:
            #Jittering
            noise =np.random.normal(0,0.02,size=points.shape)
            points += noise
            #Flip（今回はcube,sphereが対称的な形状のため、コメントアウト、if文を使わないコードの方が望ましいかも）
            # if np.random.random() > 0.5:
            #     points[:, 0] = -points[:, 0] # X軸反転
            # if np.random.random() > 0.5:
            #     points[:, 2] = -points[:, 2] # Z軸反転
            #Shear
            shear_matrix = np.eye(3)#単位行列作成
            shear_matrix[0, 1] = np.random.uniform(-0.2, 0.2) # XをY方向に歪ませる
            shear_matrix[0, 2] = np.random.uniform(-0.2, 0.2) # XをZ方向に歪ませる
            points = np.dot(points, shear_matrix)
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