# encoding: utf-8
"""
This module provides simplified chord evaluation functionality.
Now supports only 25 chord classes: 12 major, 12 minor, and 1 "no chord" class.
"""

import numpy as np
import pandas as pd
PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

class Chords:
    def __init__(self):
        pass

    def pitch(self, pitch_str):
        base_pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root = pitch_str[0].upper()
        base_pitch = base_pitch_map.get(root, -1)
        if base_pitch == -1:
            raise ValueError(f"Unknown pitch: {pitch_str}")
        # 检查是否有 # 或 b 修饰
        modifiers = pitch_str[1:]
        for m in modifiers:
            if m == '#':
                base_pitch += 1
            elif m == 'b':
                base_pitch -= 1
        return base_pitch % 12

    def convert_to_majmin25(self, root, is_major):
        if root == -1:
            return 24
        return root * 2 if is_major else root * 2 + 1

    def label_to_majmin25_id(self, label_str):
        label_str = label_str.strip()
        if label_str.upper() == 'N':
            return 24
        # 判定是否是小调
        is_minor = label_str.endswith('m')
        root_str = label_str[:-1] if is_minor else label_str
        root = self.pitch(root_str)
        return self.convert_to_majmin25(root, not is_minor)

    def idx_to_chord(self, idx):
        if idx == 24:
            return "N"
        root = PITCH_CLASS[idx // 2]
        quality = "" if idx % 2 == 0 else "m"
        return root + quality

    def idx_to_pitch(self, idx):
        """
        将 pitch 索引（0-11）映射回字母，如 0->C, 1->C#, 2->D ...
        """
        return PITCH_CLASS[idx % 12]

    def transpose_chord(self, chord_label, shift):
        """
        对和弦标签进行音高移调。支持 'A', 'Am', 'C#', 'C#m', 'N' 格式。
        chord_label: str, 原和弦
        shift: int, 上下平移的半音数量
        return: str, 移调后的和弦
        """
        if chord_label == 'N':
            return 'N'

        chord_label = chord_label.strip()
        is_minor = chord_label.endswith('m')
        root_str = chord_label[:-1] if is_minor else chord_label

        try:
            root_idx = self.pitch(root_str)  # 把字母转成pitch index（0-11）
            new_root = (root_idx + shift) % 12
            quality = "" if not is_minor else "m"
            return self.idx_to_pitch(new_root) + quality
        except Exception as e:
            print(f"⚠️ Failed to transpose chord: {chord_label}, shift={shift}, error: {e}")
            return 'N'
