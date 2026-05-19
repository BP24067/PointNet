import io
import torch
import trimesh
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.models.pointnet import PointNet

router = APIRouter()

# 事前準備（サーバー起動時）にモデルをロードしておく
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# モデルの呼び出し
model = PointNet(num_points=1024, num_labels=2).to(device)

try:
    # 重みのロード
    model.load_state_dict(torch.load("pointnet_model.pth", map_location=device))
    # 評価モードevalでの実行(すべてのニューロンが有効になる)
    model.eval()
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")


# APIのレスポンス（スキーマ）を定義
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# API通信+アルゴリズムの実装
@router.post("/predict", response_model=PredictionResponse)#通信の受け口を定義
async def predict_3d_model(file: UploadFile = File(...)):
    # 送られてきた.ply以外のファイルを拒否する（バリデーション）
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="PLY形式のファイルをアップロードしてください。")

    try:
        #通信：ファイルをメモリ上で読み込む(test_path を読み込む処理に相当)
        contents = await file.read()
        mesh = trimesh.load(io.BytesIO(contents), file_type='ply')

        #データの前処理
        # ファイルを読み込む(datasetと同じ前処理を行う必要がある)
        points, _ = trimesh.sample.sample_surface(mesh, 1024) # model.num_points＝1024
        
        # 正規化
        points -= np.mean(points, axis=0)  # バイアスを引く
        
        # 最大の距離が 1 になるようにスケーリング
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
        if furthest_distance > 0:
            points /= furthest_distance
            
        # Numpyからテンソルに変換し、GPU(またはCPU)へ転送
        tensor = torch.from_numpy(points).float().to(device)

        # 自動微分の無効化、推論の実行
        with torch.no_grad():
            output = model(tensor)
            
            # outputの中で一番大きな値を正解とする
            prediction = torch.argmax(output, dim=1).item()
            
            # ソフトマックスで確率に変換し、Numpy配列として取り出す(今回はGPUで計算してたので.cpuで戻す)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        # 実行結果のマッピング
        labels = {0: "Cube", 1: "Sphere"}
        predicted_class = labels[prediction]
        
        #labels[prediction] を使って一番高い確率を取得
        confidence = probabilities[prediction]

        # 通信：JSON形式でフロントエンドへ返却
        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(confidence * 100, 2), # %表記に直す
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {str(e)}")