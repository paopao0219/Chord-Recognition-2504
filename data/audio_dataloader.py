import torch
from torch.utils.data import DataLoader
from data.audio_dataset import AudioDataset


def collate_pad_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    feats = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return feats_padded, labels_padded


class AudioDataloader:
    def __init__(
            self,
            config,
            data_root_dir="./pt",
            batch_size=32,
            shuffle=True,
            augment=True,
            featuretype="cqt"
    ):
        self.config = config
        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.featuretype = featuretype

        self._build()

    def _build(self):
        self.train_set = AudioDataset(
            config=self.config,
            root_dir=f"{self.data_root_dir}/train_pt",
            augment=self.augment,
            featuretype=self.featuretype
        )

        self.val_set = AudioDataset(
            config=self.config,
            root_dir=f"{self.data_root_dir}/val_pt",
            augment=False,
            featuretype=self.featuretype
        )

        self.test_set = AudioDataset(
            config=self.config,
            root_dir=f"{self.data_root_dir}/test_pt",
            augment=False,
            featuretype=self.featuretype
        )

        self.test_jay_set = AudioDataset(
            config=self.config,
            root_dir=f"{self.data_root_dir}/test_jay_pt",
            augment=False,
            featuretype=self.featuretype
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn=collate_pad_fn
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

        self.test_jay_loader = DataLoader(
            self.test_jay_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pad_fn
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader, self.test_jay_loader
