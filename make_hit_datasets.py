import os
import shutil

import numpy as np
import pandas as pd

FRAME_NUM = 32
padding = FRAME_NUM // 2

base_dir = "hit_datasets"
annos_dir = os.path.join(base_dir, "annotations")
video_dir = os.path.join(base_dir, "videos")
frame_select_dir = os.path.join(base_dir, "frame_select")

hit_annos_dir = os.path.join(base_dir, "hit_annotations")
hit_video_dir = os.path.join(base_dir, "hit_videos")

if not os.path.exists(hit_annos_dir):
    os.makedirs(hit_annos_dir)
if not os.path.exists(hit_video_dir):
    os.makedirs(hit_video_dir)

frame_selects = os.listdir(frame_select_dir)
for frame_select in frame_selects:
    frame_select_path = os.path.join(frame_select_dir, frame_select)
    video_name = frame_select.replace("_annotations.csv", "")

    # 取得所有影像的 frame 數字
    all_frames = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(video_dir, video_name))])
    first_frame = min(all_frames)  # 取得最小影像編號
    max_frame = max(all_frames)  # 取得最大影像編號

    df = pd.read_csv(frame_select_path)
    for idx, row in df.iterrows():
        # 向量化處理 start_frame 和 end_frame
        start_frame = df["start_frame"].values - padding
        end_frame = df["end_frame"].values + padding

        # 產生範圍內的 frame 編號
        frame_list = np.arange(start_frame, end_frame + 1)

        # 限制範圍，小於 first_frame 的設為 first_frame，大於 max_frame 的設為 max_frame
        frame_list = np.clip(frame_list, first_frame, max_frame)

        dst_dir = os.path.join(hit_video_dir, f"{video_name}_sub_{idx:03d}")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # 複製對應範圍內的影像
        for frame_idx, frame_num in enumerate(frame_list):
            src_path = os.path.join(video_dir, video_name, f"{frame_num:05d}.png")
            dst_path = os.path.join(dst_dir, f"{frame_idx:05d}.png")
            shutil.copy(src_path, dst_path)
