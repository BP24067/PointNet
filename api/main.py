from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import predict
#FastAPIのインスタンスを作成
app = FastAPI(title="PointNet 3D Shape Classifier API")

# フロントエンドからの通信を許可する設定（CORS対策）
app.add_middleware(
    CORSMiddleware,#他のサーバ（index.html）からの通信を許可する
    allow_origins=["*"], # 開発環境のためすべて許可(本来は、ホワイトリスト式でURLを指定)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "PointNet API is running."}
