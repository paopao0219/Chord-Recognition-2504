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

    def modify(self, base_pitch, modifier):
        for m in modifier:
            if m == 'b':
                base_pitch -= 1
            elif m == '#':
                base_pitch += 1
            else:
                raise ValueError('Unknown modifier: {}'.format(m))
        return base_pitch

    def pitch(self, pitch_str):
        base_pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root = pitch_str[0].upper()
        base_pitch = base_pitch_map.get(root, -1)
        if base_pitch == -1:
            raise ValueError(f"Unknown pitch: {pitch_str}")
        return self.modify(base_pitch, pitch_str[1:]) % 12

    def chord(self, label):
        if label == 'N':
            return -1, -1, np.zeros(12, dtype=int), False

        label = label.strip()

        if ':' not in label:
            label = f"{label}:maj"

        if 'm' in label and ':' not in label:
            label = label.replace('m', ':min')

        if 'm:maj' in label:
            root_str, quality = label.split('m', 1)[0],'min'
        else:
            root_str, quality = label.split(':', 1)[0],'maj'

        is_major = False if 'min' in quality else True
        root = self.pitch(root_str)
        return root, 0, np.zeros(12, dtype=int), is_major

    def convert_to_majmin25(self, root, is_major):
        if root == -1:
            return 24
        return root * 2 if is_major else root * 2 + 1

    def label_to_majmin25_id(self, label_str):
        if label_str == 'N':
            return 24
        label_str = label_str.strip()
        if ':' not in label_str:
            label_str = f"{label_str}:maj"
        elif 'm' in label_str and ':' not in label_str:
            label_str = label_str.replace('m', ':min')
        root, bass, ivs, is_major = self.chord(label_str)
        return self.convert_to_majmin25(root, is_major)

    def idx_to_chord(self, idx):
        if idx == 24:
            return "N"
        root = PITCH_CLASS[idx // 2]
        quality = "" if idx % 2 == 0 else "m"
        return root + quality

    def transpose_chord(self, chord_label, shift):
        """对和弦标签进行音高平移"""
        if chord_label == 'N':
            return chord_label
        try:
            root, _, _, is_major = self.chord(chord_label)
            if root == -1:
                return 'N'
            new_root = (root + shift) % 12
            quality = 'maj' if is_major else 'min'
            return f"{self.idx_to_pitch(new_root)}:{quality}"
        except:
            return 'N'

    def idx_to_pitch(self, idx):
        return PITCH_CLASS[idx % 12]

