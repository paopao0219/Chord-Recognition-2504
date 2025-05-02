import os
import shutil

# 设置路径
mp3_dir = './test_jay_mp3'
pt_dir = './pt'

# 遍历 mp3 文件
for fname in os.listdir(mp3_dir):
    if fname.endswith('.mp3'):
        song_name = os.path.splitext(fname)[0]  # 去掉.mp3后缀

        # 找到所有匹配的 pt 文件
        matched_pts = [f for f in os.listdir(pt_dir) if f.startswith(song_name) and f.endswith('.pt')]

        if not matched_pts:
            print(f"⚠️ 没有找到对应的pt文件: {song_name}")
            continue

        # 创建目标文件夹 pt/xxx/
        target_dir = "./test_jay_pt"

        # 移动所有匹配到的 pt 文件
        for pt_file in matched_pts:
            src_path = os.path.join(pt_dir, pt_file)
            dst_path = os.path.join(target_dir, pt_file)
            shutil.move(src_path, dst_path)
            print(f"✅ 移动: {pt_file} -> {target_dir}")

print("🎯 所有文件整理完成！")
