import os
import torch
import numpy as np
import librosa

from data.chords import Chords  # 确保引入正确！

class Preprocess:
    def __init__(self, config, feature_to_use, dataset_names, root_dir):
        self.config = config
        self.feature_to_use = feature_to_use
        self.dataset_names = dataset_names
        self.root_dir = root_dir
        self.Chord_class = Chords()  # 实例化 Chords 工具！

    def generate_full_song_data(self, mp3_path, lab_path, save_dir="./pt", shift=True):
        mp3_conf = self.config["mp3"]
        feat_conf = self.config["feature"]
        sr = mp3_conf['song_hz']
        hop = feat_conf['hop_length']
        frame_time = hop / sr

        filename = os.path.splitext(os.path.basename(mp3_path))[0]

        # 加载lab文件
        lab_lines = []
        with open(lab_path, 'r') as f:
            for line in f:
                start, end, chord = line.strip().split()
                lab_lines.append((float(start), float(end), chord))

        shifts = [-2, -1, 0, +1, +2] if shift else [0]

        saved_pt_paths = []  # <<< 新增，保存每次生成的 .pt 文件路径

        for shift_amt in shifts:
            try:
                y, _ = librosa.load(mp3_path, sr=sr)
                if shift_amt != 0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_amt)

                cqt = librosa.cqt(y, sr=sr,
                                  n_bins=feat_conf['n_bins'],
                                  bins_per_octave=feat_conf['bins_per_octave'],
                                  hop_length=hop)
                feature = np.log(np.abs(cqt) + 1e-6).T
                n_frames = feature.shape[0]

                labels = np.full((n_frames,), 24)
                for start, end, chord in lab_lines:
                    start_idx = int(start / frame_time)
                    end_idx = int(end / frame_time)
                    try:
                        shifted_chord = self.Chord_class.transpose_chord(chord, shift_amt)
                        chord_id = self.Chord_class.label_to_majmin25_id(shifted_chord)
                    except Exception as e:
                        print(f"⚠️ 转换和弦 '{chord}' 时出错: {e}")
                        chord_id = 24
                    labels[start_idx:end_idx] = chord_id

                os.makedirs(save_dir, exist_ok=True)
                save_name = f"{filename}_shift{shift_amt}.pt"
                save_path = os.path.join(save_dir, save_name)

                out = {
                    'feature': torch.tensor(feature).float(),
                    'label': torch.tensor(labels).long()
                }
                torch.save(out, save_path)
                saved_pt_paths.append(save_path)  # <<< 保存路径
                print(f"✅ Saved: {save_path}  [frames={n_frames}]")

            except Exception as e:
                print(f"❌ Error processing {filename} shift {shift_amt}: {e}")

        return saved_pt_paths  # <<< 最后把所有保存的路径返回

    def config_to_folder(self):
        """
        根据 config 返回 mp3 参数字符串 和 特征参数字符串
        例如：
            mp3_str  = "mp3_22050hz_10.0s_skip0.5"
            feat_str = "cqt_168_24_512"
        """
        mp3_conf = self.config["mp3"]
        feat_conf = self.config["feature"]

        # 音频参数字符串
        mp3_str = f"mp3_{mp3_conf['song_hz']}hz_{mp3_conf['inst_len']}s_skip{mp3_conf['skip_interval']}"

        # 特征类型
        feature_type ="cqt"

        # 特征参数字符串
        feat_str = f"{feature_type}_{feat_conf['n_bins']}_{feat_conf['bins_per_octave']}_{feat_conf['hop_length']}"

        return mp3_str, feat_str

