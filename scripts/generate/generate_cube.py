import bpy
import random
import os
import math

# ⚠️ 注意：実行する前に、ご自身の環境に合わせて以下の保存先パスを書き換えてください。
save_dir = r"YOUR_PATH_HERE\PointNet\data\train\sphere"

# フォルダが存在しなければ作成する
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 500個作ります！
num_samples = 500

for i in range(num_samples):
    # 1. 画面上のものをすべて消す（初期化）
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 2. 新しい立方体を出す
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object

    # 3. ランダムに歪ませる（0.5倍 〜 1.5倍）
    obj.scale[0] = random.uniform(0.5, 1.5) # X方向
    obj.scale[1] = random.uniform(0.5, 1.5) # Y方向
    obj.scale[2] = random.uniform(0.5, 1.5) # Z方向

    # 4. ランダムに回転させる（0度 〜 360度）
    obj.rotation_euler[0] = random.uniform(0, math.pi * 2)
    obj.rotation_euler[1] = random.uniform(0, math.pi * 2)
    obj.rotation_euler[2] = random.uniform(0, math.pi * 2)

    # 5. 三角面化モディファイアーを追加 (コンテキストエラーを回避するために直接API)
    obj.modifiers.new(name="Triangulate", type='TRIANGULATE')

    # 6. PLY形式で保存
    filename = f"phere_{i:03d}.ply" # cube_000.ply, cube_001.ply...
    filepath = os.path.join(save_dir, filename)

    # 書き出し処理 (Blender 5.1.1 の仕様)
    bpy.ops.wm.ply_export(
        filepath=filepath,
        ascii_format=False,           # ASCIIをオフにしてバイナリ出力
        export_selected_objects=True, # 選択されているオブジェクトのみ
        apply_modifiers=True          # モディファイアーを適用
    )

print("500個の立方体の生成が完了しました！")