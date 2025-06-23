import os

import cv2
import pandas as pd
from tqdm import tqdm

# 設定資料夾路徑
video_dir = "inference/output/videos"
csv_dir = "inference/select_frame"
output_dir = "inference/output/left_cut_videos"
os.makedirs(output_dir, exist_ok=True)

# 可接受的影片前綴清單
video_prefixes = [
    "all_",
    "all_no_analyze_",
    "all_no_analyze_no_area_",
    "pose_",
    "pose_white_bg_",
    "seg_and_center_no_analyze_",
    "seg_",
    "seg_no_analyze_",
    "seg_no_analyze_no_bbox_",
]

# 分割比例：只保留前 60% 的範圍
split_ratio_train = 0.6

csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

for csv_file in tqdm(csv_files, desc="📄 處理CSV檔案"):
    base_name = csv_file.replace("_left_01_annotations.csv", "_01")

    # 載入 CSV 並計算前 60% 的 frame 範圍
    df = pd.read_csv(os.path.join(csv_dir, csv_file))
    num_ranges = len(df)
    train_indices = list(range(int(num_ranges * split_ratio_train)))
    train_df = df.iloc[train_indices]

    frame_mask = set()
    for start, end in train_df[["start_frame", "end_frame"]].values:
        frame_mask.update(range(start, end + 1))

    # 嘗試每一種前綴
    for prefix in video_prefixes:
        video_name = f"{prefix}{base_name}.avi"
        video_path = os.path.join(video_dir, video_name)

        if not os.path.exists(video_path):
            continue  # 該前綴影片不存在，跳過

        tqdm.write(f"🎬 處理影片：{video_name}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        half_width = width // 2
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")

        output_name = f"{prefix}{base_name}.avi"
        output_path = os.path.join(output_dir, output_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (half_width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 1

        with tqdm(total=total_frames, desc=f"🎞️ 擷取: {output_name}", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id in frame_mask:
                    left_half = frame[:, :half_width]
                    out.write(left_half)
                frame_id += 1
                pbar.update(1)

        cap.release()
        out.release()
        tqdm.write(f"✅ 輸出成功：{output_path}")
