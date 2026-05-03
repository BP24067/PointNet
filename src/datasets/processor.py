import trimesh
import os
import numpy as np
import torch

# 読み込むファイルのパスを指定
# プロジェクトのルートディレクトリ（PointNet/）からの相対パスで書くのがコツです
FILE_PATH = 'data/raw/キューブ.ply'
Samplepoints = 1024

def load_inspect(Spadata):

    

    # 2. 基本情報を出力
    print("-" * 30)
    print(f"データの型: {type(Spadata)}") # trimesh.base.Trimesh と出れば成功
    
    # 頂点 (X, Y, Zの座標リスト)
    print(f"頂点の数: {len(Spadata.vertices)}")
    # 面 (どの頂点を繋いで三角形にするかの指示リスト)
    print(f"面の数: {len(Spadata.faces)}")
    
    # 最初の3つの頂点だけ見てみる
    print("\n最初の3つの頂点座標 (これがAIの入力の素になります):")
    print(Spadata.vertices[:3])
    print("-" * 30)#点線が引ける

def Sampling(mesh):
    points, _ = trimesh.sample.sample_surface(mesh, Samplepoints)
    return points

def normalize_points(points):
    #バイアス（平均）を求め、その分をすべての点から引く
    points -= np.mean(points,axis=-0)

    #最大距離が1になるようにスケーリング
    # 2. 最大の距離が 1.0 になるようにスケーリングする（Scaling）
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    
    return points

def to_tensor(points):
    # NumPy配列をPyTorchテンソルに変換し、型をfloat32にする
    # PointNetは [バッチサイズ, 次元数, 点の数] という形を好むことが多いですが、
    # まずは基本の [1024, 3] を作ります。
    tensor = torch.from_numpy(points).float()
    return tensor

if __name__ == "__main__":
    #モデルを読み込む
    mesh = trimesh.load(FILE_PATH)
    #サンプリング（離散化）
    points = Sampling(mesh)
    #正規化
    points = normalize_points(points)
    #テンソル化
    final_tensor = to_tensor(points)

    #dataの詳細
    load_inspect(mesh)
    print(f"データの形状 (Shape): {points.shape}") # (1024, 3) と出れば成功

    print(f"データの型: {final_tensor.dtype}") # torch.float32 と出れば成功
    print(f"最大値: {torch.max(final_tensor):.4f}, 最小値: {torch.min(final_tensor):.4f}")
    

