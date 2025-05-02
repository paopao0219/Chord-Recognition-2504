import os
import torch
from torch.utils.data import Dataset
from data.preprocess import Preprocess
from data.chords import Chords
import random

class AudioDataset(Dataset):
    def __init__(self, config, root_dir, featuretype="cqt", augment=True):
        self.config = config
        self.root_dir = root_dir
        self.augment = augment
        self.chord_tool = Chords()
        self.preprocessor = Preprocess(config, featuretype, [], root_dir)

        self.mp3_conf = config['mp3']
        self.feat_conf = config['feature']
        self.mp3_str, self.feat_str = self.preprocessor.config_to_folder()

        self.data = []
        self._load_all()

    def __len__(self):
        return len(self.data)

    def _load_all(self):
        # 只加载传入目录中的 .pt 文件
        for fname in os.listdir(self.root_dir):
            if fname.endswith(".pt"):
                self.data.append(os.path.join(self.root_dir, fname))

    def __getitem__(self, idx):
        path = self.data[idx]
        data = torch.load(path)
        feature = data['feature']  # (T, F)
        label = data['label']      # (T,)
        return feature.float(), label.long()
