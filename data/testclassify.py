import os
import torch
from collections import Counter

def count_chords_in_pt_folder(pt_dir, ignore_value=-100, num_classes=25):
    """
    遍历 pt 文件夹，统计所有标签中各和弦 ID 的出现次数
    """
    label_counter = Counter()
    total_frames = 0

    for fname in os.listdir(pt_dir):
        if not fname.endswith(".pt"):
            continue

        fpath = os.path.join(pt_dir, fname)
        try:
            data = torch.load(fpath)
            labels = data['label'].flatten()
            labels = labels[labels != ignore_value]  # 忽略 padding
            label_counter.update(labels.tolist())
            total_frames += len(labels)
        except Exception as e:
            print(f"⚠️ 读取失败: {fname}, 错误: {e}")

    print(f"\n📊 在文件夹 '{pt_dir}' 中统计的和弦标签分布：\n")
    for i in range(num_classes):
        count = label_counter[i]
        print(f"和弦 {i:2d}: {count} 次")
    print(f"\n🧮 总帧数（有效标签数）: {total_frames}\n")

    return label_counter

# 示例调用
if __name__ == "__main__":
    pt_folder = "./val_pt"  # 根据你的需要更换为 val_pt, test_pt 等
    count_chords_in_pt_folder(pt_folder)
