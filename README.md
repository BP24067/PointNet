# PointNet 3D Shape Classifier

本プロジェクトは、3D点群データ（PLY形式）を入力として、PointNetを用いて立方体（Cube）と球体（Sphere）の2クラス分類を行うモデルの実装・検証プロジェクトです。

## 🎯 目的
3D点群認識の代表的なネットワークであるPointNetのパイプライン（データ生成、学習、推論）を基礎から構築し、機械学習におけるデータ拡張およびモデルの最適化手法の理解を深めることを目的としています。

## 🛠️ 技術スタックと環境
* **言語**: Python
* **フレームワーク**: PyTorch
* **3Dモデリング・データ生成**: Blender Python
* **点群処理**: Trimesh

## 🚀 プロジェクトの構成
```text
PointNet/
├── data/               # 学習およびテストデータ（※Git除外設定）
├── src/                # モデルおよび学習・推論のソースコード
├── scripts/            # Blenderを用いたデータ自動生成スクリプト
├── docs/               # 技術レポート（experiment_log.mdなど）
└── README.md           # プロジェクトの説明書

## Acknowledgments
本プロジェクトのPointNetモデルの実装にあたり、以下の素晴らしい記事とリポジトリを参考に（リバースエンジニアリング）させていただきました。
* [PointNetの理論と実装（点群データ） - Qiita](https://qiita.com/opeco17/items/707a5c57bca41a145122)