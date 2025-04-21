import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def calculate_biggest_pad_mask(masks):
    mask = None
    max_paddle_area = -1
    bb_boxes_pad = None

    for pred_m_pad in masks:
        if not np.any(pred_m_pad):
            continue
        pred_m_pad = pred_m_pad.astype(np.uint8)
        pred_m_pad_resize = cv2.resize(pred_m_pad, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        num_labels_resize, _, stats_resize, _ = cv2.connectedComponentsWithStats(pred_m_pad_resize)
        if num_labels_resize > 1:  # 確保有連通區域（排除背景）
            areas = stats_resize[1:, cv2.CC_STAT_AREA]  # 獲取每個連通區域的面積
            paddle_idx = np.argmax(areas) + 1  # 找到面積最大的區域（+1 因為背景是 0）
            paddle_area = int(areas[paddle_idx - 1])
            if max_paddle_area < paddle_area:
                max_paddle_area = paddle_area
                bb_boxes_pad = stats_resize[paddle_idx][: cv2.CC_STAT_AREA].tolist()
                bb_boxes_pad = [
                    bb_boxes_pad[0],
                    bb_boxes_pad[1],
                    bb_boxes_pad[0] + bb_boxes_pad[2],
                    bb_boxes_pad[1] + bb_boxes_pad[3],
                ]
                mask = pred_m_pad_resize

    return mask, max_paddle_area, bb_boxes_pad


def draw_pad_mask_on_frame(frame, pred_m_pad, bb_boxes_pad):
    if bb_boxes_pad is not None:
        # bbox
        frame = cv2.rectangle(
            frame,
            bb_boxes_pad[:2],
            bb_boxes_pad[2:],
            (255, 0, 0),
            2,
        )
        # seg
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[pred_m_pad == 1] = [255, 0, 0]
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.8, 0)

    return frame


# Load a pretrained YOLO11n-seg model
model = YOLO("runs/segment/table_tennis_stroke_postures_all_20250314_02/weights/best.pt")

# Define paths
input_video_path = "C0092.mp4"
pose_output_video_path = f"pose_{input_video_path}"
seg_output_video_path = f"seg_{input_video_path}"


cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
seg_out = cv2.VideoWriter(seg_output_video_path, fourcc, fps, (frame_width, frame_height))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
frame_count = 0
while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    for r in results:
        if r.masks is not None:
            score_pad = torch.max(r.boxes.conf).item()
            masks = r.masks.data.cpu().numpy()
            mask, paddle_area, bb_boxes_pad = calculate_biggest_pad_mask(masks)
            frame = draw_pad_mask_on_frame(frame, mask, bb_boxes_pad)
    seg_out.write(frame)

    # Update the progress bar
    progress_bar.update(1)

# Release resources
cap.release()
seg_out.release()
