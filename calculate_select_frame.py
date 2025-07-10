import os

import pandas as pd

video_info = {
    "backhand_chop": 1159,
    "backhand_flick": 1134,
    "backhand_push": 1560,
    "backhand_topspin": 855,
    "forehand_chop": 1042,
    "forehand_drive": 576,
    "forehand_smash": 1045,
    "forehand_topspin": 1567,
}

input_dir = "inference/ori_select_frame"
output_dir = "inference/select_frame"
os.makedirs(output_dir, exist_ok=True)

for name, offset in video_info.items():
    shift_amount = offset + 16
    csv_name = f"{name}_left_01_annotations.csv"
    input_path = os.path.join(input_dir, csv_name)
    output_path = os.path.join(output_dir, csv_name)

    # 讀取 CSV
    df = pd.read_csv(input_path)

    # 1️⃣ 保留 end_frame <= shift_amount 的 row
    df_keep = df[df["end_frame"] <= shift_amount].copy()

    # 2️⃣ 將整份 df 複製一份，加上 shift_amount
    df_shifted = df.copy()
    df_shifted["start_frame"] += shift_amount
    df_shifted["end_frame"] += shift_amount

    # 3️⃣ 合併結果：保留 + 偏移
    df_combined = pd.concat([df_keep, df_shifted], ignore_index=True)

    # 4️⃣ 儲存
    df_combined.to_csv(output_path, index=False)
    print(f"✅ 已處理並儲存：{output_path}")
