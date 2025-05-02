import os
import re

# 降调 → 升调映射
flat_to_sharp = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'db': 'C#', 'eb': 'D#', 'gb': 'F#', 'ab': 'G#', 'bb': 'A#',
}


# 简化和弦
def simplify_chord(chord):
    if chord == 'N':
        return 'N'

    # 匹配根音（带b/#）+ 可选修饰
    match = re.match(r"^([A-Ga-g][b#]?)(?::([^\s]+))?$", chord)
    if not match:
        return 'N'

    root, quality = match.groups()

    # 转为升调
    root = flat_to_sharp.get(root, root.upper())

    # 判断是否小调
    if quality and re.search(r'\b(min|m)', quality, re.IGNORECASE):
        return root + 'm'
    else:
        return root


# 处理单个lab文件
def process_lab_file(file_path):
    output_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, chord = parts
                simplified = simplify_chord(chord)
                output_lines.append(f"{start} {end} {simplified}")

    # 覆盖写回
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')


# 遍历并处理文件夹
def process_all_lab_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.lab'):
                full_path = os.path.join(root, file)
                print(f"Processing: {full_path}")
                process_lab_file(full_path)


if __name__ == "__main__":
    folder_path = "./Test_full/test_jay_"  # 替换为你自己的路径
    process_all_lab_files(folder_path)
