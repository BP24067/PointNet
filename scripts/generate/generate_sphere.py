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
    # 画面上のものをすべて消す（初期化）
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # UV球かICO球をランダムに選んで生成する
    if random.choice(['UV', 'ICO']) == 'UV':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1)
        
    obj = bpy.context.active_object

    # ランダムに歪ませる（0.5倍 〜 1.5倍）
    obj.scale[0] = random.uniform(0.5, 1.5)
    obj.scale[1] = random.uniform(0.5, 1.5)
    obj.scale[2] = random.uniform(0.5, 1.5)

    # ランダムに回転させる（0度 〜 360度）
    obj.rotation_euler[0] = random.uniform(0, math.pi * 2)
    obj.rotation_euler[1] = random.uniform(0, math.pi * 2)
    obj.rotation_euler[2] = random.uniform(0, math.pi * 2)

    # 三角面化モディファイアーを追加
    obj.modifiers.new(name="Triangulate", type='TRIANGULATE')

    # ファイル名を変更
    filename = f"sphere_{i:03d}.ply" # sphere_000.ply, sphere_001.ply...
    filepath = os.path.join(save_dir, filename)

    # 書き出し処理 (Blender 5.x の仕様)
    bpy.ops.wm.ply_export(
        filepath=filepath,
        ascii_format=False,           # ASCIIをオフにしてバイナリ出力
        export_selected_objects=True, # 選択されているオブジェクトのみ
        apply_modifiers=True          # モディファイアーを適用
    )

print("500個の球体（UV球・ICO球ミックス）の生成が完了しました！")