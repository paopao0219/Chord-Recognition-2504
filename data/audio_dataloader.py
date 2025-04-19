import torch
from torch.utils.data import DataLoader, random_split
from data.audio_dataset import AudioDataset


def collate_pad_fn(batch):
    """
    batch: list of (feature: [T, F], label: [T])
    Returns: padded feature (B, T, F), padded label (B, T)
    """
    from torch.nn.utils.rnn import pad_sequence

    feats = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)  # (B, T, F)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # (B, T)

    return feats_padded, labels_padded

class AudioDataloader:
    def __init__(
            self,
            config,
            data_root_dir="./pt",
            batch_size=32,
            val_split=0.2,
            shuffle=True,
            augment=True,
            featuretype=None,
            seed=42
    ):
        self.config = config
        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.augment = augment
        self.featuretype = featuretype
        self.seed = seed

        self._build()

    def _build(self):
        # 加载完整数据集
        full_dataset = AudioDataset(
            config=self.config,
            root_dir=self.data_root_dir,  # 现在作为"唯一来源目录"
            augment=self.augment,
            featuretype=self.featuretype
        )

        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * 0.1)  # 例如 10% 做测试
        train_len = total_len - val_len - test_len

        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [train_len, val_len, test_len], generator=generator
        )

        # 构造 DataLoader
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn = collate_pad_fn

        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pad_fn
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pad_fn
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


