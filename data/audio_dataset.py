import os
import torch
from torch.utils.data import Dataset
from data.preprocess import Preprocess, FeatureTypes
from data.chords import Chords
import random

class AudioDataset(Dataset):
    def __init__(self, config, root_dir, featuretype=FeatureTypes.cqt, augment=True):
        self.config = config
        self.root_dir = root_dir
        self.augment = augment
        self.chord_tool = Chords()
        self.preprocessor = Preprocess(config, featuretype, [], root_dir)

        self.mp3_conf = config['mp3']
        self.feat_conf = config['feature']
        self.mp3_str, self.feat_str = self.preprocessor.config_to_folder()[2:]

        self.data = []
        self._load_all()
    def __len__(self):
        return len(self.data)

    def _load_all(self):
        base_dir = os.path.join(self.root_dir, "pt")
        for fname in os.listdir(base_dir):
            if fname.endswith(".pt"):
                self.data.append(os.path.join(base_dir, fname))

    def __getitem__(self, idx):
        path = self.data[idx]
        data = torch.load(path)
        feature = data['feature']  # (T, F)
        label = data['label']  # (T,)
        return feature.float(), label.long()

if __name__ == "__main__":
    import os
    import json
    from torch.utils.data import DataLoader
    from data.preprocess import FeatureTypes

    # 示例 config（与 preprocess 测试一致）
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

    # 注意路径需要指向生成的 .pt 文件目录
    root_dir = r"/data/Test_full"  # 你自己的 full_song_pt 父目录
    dataset = AudioDataset(
        config=config,
        root_dir=root_dir,
        featuretype=FeatureTypes.cqt,
        augment=False
    )

    print(f"✅ 加载成功: 共 {len(dataset)} 首歌曲")

    # 随便取一首歌查看
    feature, label = dataset[0]
    print(f"特征 shape: {feature.shape}")  # (T, F)
    print(f"标签 shape: {label.shape}")   # (T,)
    print(f"前10帧的标签: {label[:10].tolist()}")

    # 可选：创建 dataloader 看看 batch
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in loader:
        x, y = batch  # x: (B, T, F), y: (B, T)
        print(f"batch 特征: {x.shape}, batch 标签: {y.shape}")
        break



