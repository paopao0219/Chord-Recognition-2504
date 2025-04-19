import os
from data.preprocess import Preprocess, FeatureTypes

# ==== 配置（要和训练用的 config 一致） ====
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

# ==== 数据目录设置 ====
base_dir = "./Test_full"
mp3_dir = os.path.join(base_dir, "mp3")
lab_dir = os.path.join(base_dir, "lab")

preprocessor = Preprocess(
    config=config,
    feature_to_use=FeatureTypes.cqt,
    dataset_names=[],
    root_dir=base_dir
)

# ==== 遍历所有 .mp3 和 .lab ====
for fname in sorted(os.listdir(mp3_dir)):
    if fname.endswith(".mp3"):
        song_name = fname[:-4]
        mp3_path = os.path.join(mp3_dir, fname)
        lab_path = os.path.join(lab_dir, song_name + ".lab")

        if not os.path.exists(lab_path):
            print(f"⚠️ LAB 文件缺失，跳过: {song_name}")
            continue

        print(f"🎵 处理: {song_name}")
        try:
            preprocessor.generate_full_song_data(mp3_path, lab_path)
        except Exception as e:
            print(f"❌ 错误处理 {song_name}: {e}")
