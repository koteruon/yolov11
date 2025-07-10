import os

import cv2
from tqdm import tqdm

# 使用者提供的 frame 編號（從 1 開始）
video_info = {
    "backhand_chop_01": 1159,
    "backhand_flick_01": 1134,
    "backhand_push_01": 1560,
    "backhand_topspin_01": 855,
    "forehand_chop_01": 1042,
    "forehand_drive_01": 576,
    "forehand_smash_01": 1045,
    "forehand_topspin_01": 1567,
}
frame_offset = 16

input_dir = "inference/ori_videos"
output_dir = "inference/videos"
os.makedirs(output_dir, exist_ok=True)

# 處理每部影片
for name, user_frame_number in video_info.items():
    extract_until = user_frame_number + frame_offset  # 人類編號

    input_path = os.path.join(input_dir, f"{name}.mp4")
    output_path = os.path.join(output_dir, f"{name}.avi")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 無法打開影片: {input_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 第一段 clip1：frame 1 到 extract_until
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clip1_frames = min(extract_until, total_frames)
    print(f"📦 擷取 clip1：frame 1 ~ {clip1_frames}")
    pbar1 = tqdm(total=clip1_frames, desc=f"{name} - clip1")
    frame_count = 1
    while frame_count <= clip1_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        pbar1.update(1)
    pbar1.close()

    # 第二段 clip2：frame 1 到 total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"📦 擷取 clip2：frame 1 ~ {total_frames}")
    pbar2 = tqdm(total=total_frames, desc=f"{name} - clip2")
    frame_count = 1
    while frame_count <= total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        pbar2.update(1)
    pbar2.close()

    cap.release()
    out.release()
    print(f"✅ 已儲存至：{output_path}")
