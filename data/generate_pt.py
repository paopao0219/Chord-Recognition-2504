import os
import torch
from collections import Counter
from data.preprocess import Preprocess
from data.chords import Chords

# ==== 全局统计器 ====
label_counter = Counter()

# ==== 配置 ====
config = {
    "mp3": {
        "song_hz": 22050,
        "inst_len": 10.0,
        "skip_interval": 5.0
    },
    "feature": {
        "n_bins": 168,
        "bins_per_octave": 24,
        "hop_length": 512
    }
}

data_splits = {
    "train": {"mp3_dir": "./train_mp3", "lab_dir": "./train_lab", "save_dir": "./train_pt", "shift": True},
    "val": {"mp3_dir": "./val_mp3", "lab_dir": "./val_lab", "save_dir": "./val_pt", "shift": False},
    "test": {"mp3_dir": "./test_mp3", "lab_dir": "./test_lab", "save_dir": "./test_pt", "shift": False},
    "test_jay": {"mp3_dir": "./test_jay_mp3", "lab_dir": "./test_jay_lab", "save_dir": "./test_jay_pt", "shift": False}
}

preprocessor = Preprocess(
    config=config,
    feature_to_use="cqt",
    dataset_names=[],
    root_dir="."
)

chord_tool = Chords()

# ==== 遍历各个数据集 ====
for split_name, paths in data_splits.items():
    mp3_dir = paths["mp3_dir"]
    lab_dir = paths["lab_dir"]
    save_dir = paths["save_dir"]
    shift = paths["shift"]

    os.makedirs(save_dir, exist_ok=True)

    print(f"===== 处理数据集: {split_name.upper()} =====")

    for fname in sorted(os.listdir(mp3_dir)):
        if not fname.endswith(".mp3"):
            continue

        song_name = fname[:-4]
        mp3_path = os.path.join(mp3_dir, fname)
        lab_path = os.path.join(lab_dir, song_name + ".lab")

        if not os.path.exists(lab_path):
            print(f"⚠️ LAB 文件缺失，跳过: {song_name}")
            continue

        print(f"🎵 处理: {song_name} (shift={shift})")
        try:
            # 调用 Preprocess 生成 pt 文件
            pt_paths = preprocessor.generate_full_song_data(
                mp3_path=mp3_path,
                lab_path=lab_path,
                save_dir=save_dir,
                shift=shift
            )

            # ==== 新增：累积统计 ====
            for pt_path in pt_paths:
                data = torch.load(pt_path)
                labels = data['label'].flatten()
                labels = labels[labels != -100]
                label_counter.update(labels.tolist())


        except Exception as e:
            print(f"❌ 错误处理 {song_name}: {e}")

# ==== 最后输出类别统计 ====
print("\n===== 各类别样本数量统计 =====")
for cls in range(25):
    print(f"类别 {cls}: {label_counter[cls]} 次")

print("\n✅ 全部处理完成！")
