import os
import librosa
import data.chords
import re
from enum import Enum
import pyrubberband as pyrb
import torch
import math
import numpy as np
from tqdm import tqdm

from data import chords


class FeatureTypes(Enum):
    cqt = 'cqt'

class Preprocess():
    def __init__(self, config, feature_to_use, dataset_names, root_dir):
        self.config = config
        self.dataset_names = dataset_names
        self.root_path = root_dir + '/'

        self.time_interval = config["feature"]["hop_length"] / config["mp3"]["song_hz"]
        self.no_of_chord_datapoints_per_sequence = math.ceil(config["mp3"]['inst_len'] / self.time_interval)
        self.Chord_class = chords.Chords()

        self.feature_name = feature_to_use
        self.is_cut_last_chord = False

    def config_to_folder(self):
        mp3_config = self.config["mp3"]
        feature_config = self.config["feature"]
        mp3_string = "%d_%.1f_%.1f" % (
            mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
        feature_string = "%s_%d_%d_%d" % (
            self.feature_name.value, feature_config['n_bins'], feature_config['bins_per_octave'],
            feature_config['hop_length'])

        return mp3_config, feature_config, mp3_string, feature_string

    def generate_full_song_data(self, mp3_path, lab_path):
        import librosa
        import numpy as np
        import torch
        import os

        mp3_conf = self.config["mp3"]
        feat_conf = self.config["feature"]
        sr = mp3_conf['song_hz']
        hop = feat_conf['hop_length']
        frame_time = hop / sr

        filename = os.path.splitext(os.path.basename(mp3_path))[0]

        # 加载标签
        lab_lines = []
        with open(lab_path, 'r') as f:
            for line in f:
                start, end, chord = line.strip().split()
                lab_lines.append((float(start), float(end), chord))

        # 遍历不同的音高偏移量（增强）
        for shift in [-2, -1, 0, +1, +2]:
            try:
                # 加载音频并进行音高变换
                y, _ = librosa.load(mp3_path, sr=sr)
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)

                # 计算 CQT
                cqt = librosa.cqt(y_shifted, sr=sr,
                                  n_bins=feat_conf['n_bins'],
                                  bins_per_octave=feat_conf['bins_per_octave'],
                                  hop_length=hop)
                feature = np.log(np.abs(cqt) + 1e-6).T  # shape: (T, F)
                n_frames = feature.shape[0]

                # 创建标签序列
                labels = np.full((n_frames,), 24)  # 24 = 'N'
                for start, end, chord in lab_lines:
                    start_idx = int(start / frame_time)
                    end_idx = int(end / frame_time)
                    shifted_chord = self.Chord_class.transpose_chord(chord, shift)
                    chord_id = self.Chord_class.label_to_majmin25_id(shifted_chord)
                    labels[start_idx:end_idx] = chord_id

                # 保存
                out = {
                    'feature': torch.tensor(feature).float(),
                    'label': torch.tensor(labels).long()
                }

                # 获取 mp3 上一级目录（如 Test_full）
                base_dir = os.path.dirname(os.path.dirname(mp3_path))

                # 构造 pt 文件夹路径
                pt_dir = os.path.join(base_dir, "pt")
                os.makedirs(pt_dir, exist_ok=True)

                # 构造保存路径
                save_name = f"{filename}_shift{shift}.pt"
                save_path = os.path.join(pt_dir, save_name)

                # 保存
                torch.save(out, save_path)
                print(f"✅ Saved: {save_path}  [frames={n_frames}]")

            except Exception as e:
                print(f"❌ Error in {filename} shift {shift}: {e}")

    def get_all_single_chord_files(self):
        res_list = []
        for chord_name in sorted(os.listdir(self.root_path)):
            chord_path = os.path.join(self.root_path, chord_name)
            if not os.path.isdir(chord_path): continue
            for fname in os.listdir(chord_path):
                if fname.endswith(".mp3"):
                    res_list.append((chord_name, os.path.join(chord_path, fname)))
        return res_list

    def generate_single_chord_data(self, file_list):
        mp3_conf = self.config["mp3"]
        feat_conf = self.config["feature"]
        mp3_str, feat_str = self.config_to_folder()[2:]
        chord_tool = self.Chord_class

        for label_str, fpath in tqdm(file_list):
            print(label_str,fpath)
            try:
                chord_id = chord_tool.label_to_majmin25_id(label_str)
            except:
                print(f"[Warning] Skipping label '{label_str}'")
                continue

            try:
                y, sr = librosa.load(fpath, sr=mp3_conf['song_hz'])
            except:
                print(f"[Error] Can't load {fpath}")
                continue


            for shift in [-2, -1, 0, 1, 2]:
                try:
                    y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                    adjusted_id = (chord_id + shift * 2) % 24

                    cqt = librosa.cqt(y_aug, sr=sr,
                                      n_bins=feat_conf['n_bins'],
                                      bins_per_octave=feat_conf['bins_per_octave'],
                                      hop_length=feat_conf['hop_length'])
                    feature = np.log(np.abs(cqt) + 1e-6)

                    max_len = int(mp3_conf['inst_len'] * sr / feat_conf['hop_length'])
                    if feature.shape[1] < max_len:
                        pad = max_len - feature.shape[1]
                        feature = np.pad(feature, ((0, 0), (0, pad)), mode='constant')
                    else:
                        feature = feature[:, :max_len]

                    out = {
                        'feature': torch.tensor(feature).unsqueeze(0),
                        'chord': adjusted_id,
                        'etc': f"{shift:+d}"
                    }

                    save_dir = os.path.join(self.root_path, "result_single", mp3_str, feat_str, label_str)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.basename(fpath).replace(".mp3", "")
                    save_path = os.path.join(save_dir, f"{filename}_{shift:+d}.pt")
                    torch.save(out, save_path)
                except Exception as e:
                    print(f"[Error] Processing {fpath}: {e}")
        # for label_str, fpath in tqdm(file_list):
        #     print(label_str,fpath)
        #     try:
        #         chord_id = chord_tool.label_to_majmin25_id(label_str)
        #     except:
        #         print(f"[Warning] Skipping label '{label_str}'")
        #         continue
        #
        #     try:
        #         y, sr = librosa.load(fpath, sr=mp3_conf['song_hz'])
        #     except:
        #         print(f"[Error] Can't load {fpath}")
        #         continue
        #
        #     try:
        #
        #         cqt = librosa.cqt(y, sr=sr,
        #                           n_bins=feat_conf['n_bins'],
        #                           bins_per_octave=feat_conf['bins_per_octave'],
        #                           hop_length=feat_conf['hop_length'])
        #         feature = np.log(np.abs(cqt) + 1e-6)
        #
        #         max_len = int(mp3_conf['inst_len'] * sr / feat_conf['hop_length'])
        #         if feature.shape[1] < max_len:
        #             pad = max_len - feature.shape[1]
        #             feature = np.pad(feature, ((0, 0), (0, pad)), mode='constant')
        #         else:
        #             feature = feature[:, :max_len]
        #
        #         out = {
        #             'feature': torch.tensor(feature).unsqueeze(0),
        #             'chord': chord_id,
        #         }
        #
        #         save_dir = os.path.join(self.root_path, "result_single", mp3_str, feat_str, label_str)
        #         os.makedirs(save_dir, exist_ok=True)
        #         filename = os.path.basename(fpath).replace(".mp3", "")
        #         save_path = os.path.join(save_dir, f"{filename}.pt")
        #         torch.save(out, save_path)
        #     except Exception as e:
        #         print(f"[Error] Processing {fpath}: {e}")
if __name__ == "__main__":
    import json

    # 示例配置（可按需替换）
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

    # 示例路径（请替换为你的实际路径）
    mp3_path = r"/data/Test_full\mp3\01_-_A_Hard_Day's_Night.mp3"
    lab_path = r"/data/Test_full\lab\01_-_A_Hard_Day's_Night.lab"


    # 初始化预处理器
    preprocess = Preprocess(config=config,
                            feature_to_use=FeatureTypes.cqt,
                            dataset_names=[],
                            root_dir=".")

    # 执行预处理并生成 .pt 文件
    preprocess.generate_full_song_data(mp3_path, lab_path)
