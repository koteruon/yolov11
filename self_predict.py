import json
import os
from copy import deepcopy

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class Predict:
    def __init__(self, seg_model_path=None):
        # Define stroke index
        self.stroke_id = {
            "backhand_chop": 1,
            "backhand_flick": 2,
            "backhand_push": 3,
            "backhand_topspin": 4,
            "forehand_chop": 5,
            "forehand_drive": 6,
            "forehand_smash": 7,
            "forehand_topspin": 8,
        }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize YOLO model
        if seg_model_path:
            self.seg_model = YOLO(seg_model_path)
            self.seg_model.to(device)
            self.seg_verbose = False

        # Initialize YOLO pose model
        self.pose_model = YOLO("weights_dir/yolo11x-pose.pt")
        self.pose_model.to(device)
        self.pose_verbose = False

        # Plt variables
        self.plot_width = 920
        self.plot_height = 300
        self.plot_x = 10
        self.plot_y = 0
        figure, (self.area_axes, self.center_axes) = plt.subplots(
            1, 2, figsize=(self.plot_width / 100, self.plot_height / 100)
        )
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        self.frame_count_range = 37

        # annotation
        self.stage = "train"
        self.person_bbox_filename = f"stroke_postures_train_gt.json"
        self.person_bbox_dict = {
            "categories": [{"supercategory": "person", "id": 1, "name": "person"}],
            "annotations": [],
            "images": [],
        }
        self.person_keypoint_filename = f"stroke_postures_train_bbox_kpts.json"
        self.person_keypoint_list = []
        self.object_detection_filename = f"stroke_postures_train_object_detection.json"
        self.object_detection_list = []

    def load_video_data(
        self,
        input_video_path,
        pose_output_video_path,
        pose_white_bg_output_video_path,
        seg_output_video_path,
        seg_no_analyze_output_video_path,
        all_output_video_path,
        all_no_analyze_output_video_path,
    ):
        # Get filename and video id
        self.input_video_filename, _ = os.path.splitext(os.path.basename(input_video_path))
        self.input_video_stroke_name, self.input_video_id = self.input_video_filename.rsplit("_", 1)
        self.input_video_id = int(self.input_video_id)
        self.input_video_stroke_id = self.stroke_id[self.input_video_stroke_name]

        # Initialize video capture and writer
        self.cap = cv2.VideoCapture(input_video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.pose_out = cv2.VideoWriter(pose_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.pose_white_bg_out = cv2.VideoWriter(
            pose_white_bg_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        self.seg_out = cv2.VideoWriter(seg_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.seg_no_analyze_out = cv2.VideoWriter(
            seg_no_analyze_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        self.all_out = cv2.VideoWriter(all_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.all_no_analyze_out = cv2.VideoWriter(
            all_no_analyze_output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 獲取影片總幀數

        # Tracking variables
        self.frame_count = 0
        self.paddle_center_list = []
        self.paddle_area_list = []

    def draw_pad_mask_on_frame(self, frame, pred_m_pad, bb_boxes_pad):
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

    def calculate_biggest_pad_mask(self, masks):
        mask = None
        max_paddle_area = -1
        bb_boxes_pad = None

        for pred_m_pad in masks:
            if not np.any(pred_m_pad):
                continue
            pred_m_pad = pred_m_pad.astype(np.uint8)
            pred_m_pad_resize = cv2.resize(
                pred_m_pad, (self.frame_width, self.frame_height), interpolation=cv2.INTER_NEAREST
            )
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

    def add_center_and_area_to_list(self, paddle_area, bb_boxes_pad):
        if bb_boxes_pad is not None:
            paddle_center_x = (bb_boxes_pad[0] + bb_boxes_pad[2]) / 2
            paddle_center_y = (bb_boxes_pad[1] + bb_boxes_pad[3]) / 2
            self.paddle_center_list.append((self.frame_count, paddle_center_x, paddle_center_y))
            self.paddle_area_list.append((self.frame_count, paddle_area))
            return [paddle_center_x, paddle_center_y]
        else:
            self.paddle_center_list.append((self.frame_count, None, None))
            self.paddle_area_list.append((self.frame_count, None))
            return [None, None]

    def seg_frame(self, frame, frame_with_pose):
        """Process a single frame using the YOLO model."""
        bb_boxes_pad = None
        score_pad = None
        paddle_area = -1

        results = self.seg_model(frame, verbose=self.seg_verbose)
        for r in results:
            if r.masks is not None:
                score_pad = torch.max(r.boxes.conf).item()
                masks = r.masks.data.cpu().numpy()
                mask, paddle_area, bb_boxes_pad = self.calculate_biggest_pad_mask(masks)
                frame = self.draw_pad_mask_on_frame(frame, mask, bb_boxes_pad)
                frame_with_pose = self.draw_pad_mask_on_frame(frame_with_pose, mask, bb_boxes_pad)
        paddle_center = self.add_center_and_area_to_list(paddle_area, bb_boxes_pad)

        return frame, frame_with_pose, bb_boxes_pad, score_pad, paddle_center, paddle_area

    def clear_and_set_plt_axes(self, x_min, x_max):
        # clear every axis
        self.area_axes.cla()
        self.center_axes.cla()

        # Configure paddle area data
        self.area_axes.set_xlim((x_min, x_max))
        self.area_axes.set_ylim((0, 5000))
        self.area_axes.set_xlabel("Frame")
        self.area_axes.set_ylabel("Paddle Pixels Area")
        area_line = mlines.Line2D(
            [], [], marker=None, linewidth=1, color=(0 / 255, 0 / 255, 255 / 255), label="Paddle area", linestyle="-"
        )
        self.area_axes.legend(handles=[area_line])

        # Configure paddle center data
        self.center_axes.set_xlim((200, 1000))
        self.center_axes.set_ylim((200, 1000))
        self.center_axes.set_xlabel("x-position")
        self.center_axes.set_ylabel("y-position")
        paddle_line = mlines.Line2D(
            [],
            [],
            marker="o",
            markersize=5,
            color=(218 / 255, 165 / 255, 32 / 255),
            label="Paddle route",
            linestyle="None",
        )
        self.center_axes.legend(handles=[paddle_line])

    def draw_area_axes(self, x_min, x_max):
        # Plot paddle area data
        x_data = range(x_min, x_max)
        y_data = self.paddle_area_list[x_min : x_max + 1]
        y_data = np.array(y_data, dtype=float)[:, 1]
        nan_mask = np.isnan(y_data)
        if np.all(np.isnan(y_data)):
            y_data = np.zeros_like(y_data)
        else:
            y_data[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), y_data[~nan_mask])  # 插值
        self.area_axes.plot(x_data, y_data, "bo-", markersize=1)

    def draw_center_axes(self, centers, color_components):
        color_rgbs_norm = np.array([(c / 255, max(c - 30, 0) / 255, 0 / 255) for c in color_components])
        self.center_axes.scatter(centers[:, 0], self.frame_height - centers[:, 1], c=color_rgbs_norm, marker="o", s=50)

    def calculate_paddle_centers(self):
        # Determine frame range for paddle center plot
        frame_min_pad = max(0, len(self.paddle_center_list) - self.frame_count_range)

        # Extract the paddle center data
        paddle_centers = np.array(
            self.paddle_center_list[frame_min_pad : min(self.frame_count, len(self.paddle_center_list))]
        )
        centers = paddle_centers[:, 1:3]
        centers = centers.astype(float)

        # 建立mask
        nan_mask = np.isnan(centers)
        valid_mask = ~nan_mask

        # 從前往後填充
        indices = np.where(valid_mask, np.arange(len(centers))[:, None], 0)  # 獲取有效值的索引
        np.maximum.accumulate(indices, axis=0, out=indices)  # 前向填充索引
        centers = centers[indices, np.arange(centers.shape[1])]

        # 從後往前填充
        indices = np.where(valid_mask, np.arange(len(centers))[:, None], len(centers) - 1)  # 獲取有效值的索引
        np.minimum.accumulate(indices[::-1], axis=0, out=indices[::-1])  # 後向填充索引
        centers = centers[indices, np.arange(centers.shape[1])]

        # 只有nan則補0
        nan_rows = np.isnan(centers).all(axis=1)
        centers[nan_rows] = [0, 0]

        centers = centers.astype(int)

        return centers

    def convert_plot_to_image(self, frame):
        fig = plt.gcf()
        fig.set_size_inches(self.plot_width / 100, self.plot_height / 100)
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())
        # Resize and overlay the plot image on the frame
        plot_img_resized = cv2.resize(plot_img, (self.plot_width, self.plot_height))
        plot_img_resized_rgb = cv2.cvtColor(plot_img_resized, cv2.COLOR_RGBA2BGR)
        frame[self.plot_y : self.plot_y + self.plot_height, self.plot_x : self.plot_x + self.plot_width] = (
            plot_img_resized_rgb
        )
        return frame

    def draw_plt(self, frame, frame_with_pose):
        # Determine x-axis limits for area plot
        x_min = max(0, self.frame_count - 120)
        x_max = self.frame_count

        # Initialize plot
        self.clear_and_set_plt_axes(x_min, x_max)

        # Plot paddle area data
        self.draw_area_axes(x_min, x_max)

        # Calculate paddle centers
        centers = self.calculate_paddle_centers()

        # Calculate the color components
        gradient = np.linspace(1, 0, len(centers))
        color_components = (255 - (gradient * 255)).astype(int)

        # Scatter the valid centers
        self.draw_center_axes(centers, color_components)

        # Convert the plot to an image
        frame = self.convert_plot_to_image(frame)
        frame_with_pose = self.convert_plot_to_image(frame_with_pose)

        # Draw circles on the frame
        color_bgrs = [(0, max(c - 30, 0), c) for c in color_components]
        for center, color_bgr in zip(centers, color_bgrs):
            cv2.circle(frame, tuple(center), 5, tuple(map(int, color_bgr)), -1)
            cv2.circle(frame_with_pose, tuple(center), 5, tuple(map(int, color_bgr)), -1)

        return frame, frame_with_pose

    def plot_pose(self, ori_left_frame, pose_result_names, pose_result_boxes, pose_result_keypoints, has_bbox=True):
        img = ori_left_frame  # 1080*960
        img_shape = img.shape[:2]
        names = pose_result_names
        annotator = Annotator(
            deepcopy(img),
            line_width=None,
            font_size=None,
            font="Arial.ttf",
            pil=False,  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Detect results
        if has_bbox:
            pred_boxes = pose_result_boxes
            # c, conf, id = int(pred_boxes.cls), float(pred_boxes.conf), None
            # name = ("" if id is None else f"id:{id} ") + names[c]
            # label = f"{name} {conf:.2f}" if conf else name
            c = 9  # (189, 0, 255): 紫色（Purple）
            label = ""
            box = pred_boxes.xyxy.squeeze()
            annotator.box_label(
                box,
                label,
                color=colors(c, True),
                rotated=False,
            )

        # Plot Pose results
        for i, k in enumerate(reversed(pose_result_keypoints.data)):
            annotator.kpts(
                k,
                img_shape,
                radius=5,
                kpt_line=True,
                conf_thres=0.5,
                kpt_color=None,
            )

        return annotator.result()

    def pose_frame(self, frame):
        ori_left_frame = frame[:, : self.frame_width // 2]
        pose_results = self.pose_model(ori_left_frame, verbose=self.pose_verbose)

        # get pose bbox
        bb_boxes_person = None
        keypoint = None
        score_keypoint = None
        pose_result = pose_results[0]

        assert pose_result.boxes.xyxy.shape[0] >= 1
        assert pose_result.keypoints.data.shape[0] >= 1
        boxes_max_conf = torch.argmax(pose_result.boxes.conf)
        pose_result_boxes = pose_result.boxes[boxes_max_conf]
        bb_boxes_person = pose_result_boxes.xyxy[0].cpu().tolist()  # 用pose的bbox當作GT的bbox

        pose_max_conf = torch.argmax(torch.max(pose_result.keypoints.conf, dim=-1).values)
        pose_result_keypoints = pose_result.keypoints[pose_max_conf]
        keypoint = pose_result_keypoints.data[0].cpu().tolist()
        score_keypoint = torch.max(pose_result_keypoints.conf).item()

        pose_result_names = pose_result.names
        pose_left_frame = self.plot_pose(ori_left_frame, pose_result_names, pose_result_boxes, pose_result_keypoints)
        frame[:, : self.frame_width // 2] = pose_left_frame

        white_frame = np.full_like(frame, 255)
        left_white_frame = white_frame[:, : self.frame_width // 2]
        pose_left_white_frame = self.plot_pose(
            left_white_frame, pose_result_names, None, pose_result_keypoints, has_bbox=False
        )
        white_frame[:, : self.frame_width // 2] = pose_left_white_frame

        return frame, white_frame, bb_boxes_person, keypoint, score_keypoint

    def save_ori_frame(self, frame):
        """Save the original frame."""
        save_path = os.path.join(self.image_dir, f"{self.frame_count:05d}.png")
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, frame)

    def append_annotation_data(
        self, bb_boxes_person, keypoint, score_keypoint, bb_boxes_pad, score_pad, paddle_center, paddle_area
    ):
        # bbox gt = yolo detect
        if bb_boxes_person is not None:
            image_id = int(f"{self.input_video_stroke_id}{self.input_video_id:03d}{self.frame_count:05d}")
            data = {
                "id": image_id,
                "img_path": os.path.join(self.input_video_filename, f"{self.frame_count:05d}.png"),
                "height": self.frame_height,
                "width": self.frame_width,
                "movie": self.input_video_filename,
                "timestamp": f"{self.frame_count:05d}",
            }
            self.person_bbox_dict["images"].append(data)

            # annotation
            widths = bb_boxes_person[2] - bb_boxes_person[0]
            heights = bb_boxes_person[3] - bb_boxes_person[1]
            areas = widths * heights
            assert areas >= 0
            data = {
                "id": int(f"{image_id}001"),
                "image_id": image_id,
                "category_id": 1,
                "action_ids": [self.input_video_stroke_id],
                "person_id": 1,
                "bbox": bb_boxes_person,  # xyxy
                "area": areas,
                "keypoints": [],
                "iscrowd": 0,
            }
            self.person_bbox_dict["annotations"].append(data)

            # keypoint
            data = {
                "image_id": image_id,
                "category_id": 0,
                "bbox": bb_boxes_person,  # xyxy
                "keypoints": keypoint,
                "score": score_keypoint,
            }
            self.person_keypoint_list.append(data)

        # object
        # bb_boxes_pad 的格式是 xywh，其中xy在最左上角
        if bb_boxes_pad is not None:
            data = {
                "image_id": image_id,
                "category_id": 0,
                "bbox": bb_boxes_pad,  # xyxy
                "score": score_pad,
                "center": paddle_center,
                "area": paddle_area,
            }
            self.object_detection_list.append(data)

    def save_annotation_data(self):
        with open(os.path.join(self.anno_dir, self.person_bbox_filename), "w", encoding="utf-8") as f:
            json.dump(self.person_bbox_dict, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.anno_dir, self.person_keypoint_filename), "w", encoding="utf-8") as f:
            json.dump(self.person_keypoint_list, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.anno_dir, self.object_detection_filename), "w", encoding="utf-8") as f:
            json.dump(self.object_detection_list, f, ensure_ascii=False, indent=4)

    def process_video(self):
        """Process the entire video frame by frame."""
        progress_bar = tqdm(total=self.total_frames, desc="Processing Video", unit="frame")
        while self.cap.isOpened():
            self.frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                break

            self.save_ori_frame(frame)

            pose_frame = frame.copy()
            seg_frame = frame.copy()

            # pose the frame
            pose_frame, pose_white_frame, bb_boxes_person, keypoint, score_keypoint = self.pose_frame(pose_frame)

            # seg the frame
            seg_frame, all_frame, bb_boxes_pad, score_pad, paddle_center, paddle_area = self.seg_frame(
                seg_frame, pose_frame.copy()
            )
            seg_no_analyze_frame = seg_frame.copy()
            all_no_analyze_frame = all_frame.copy()

            # draw plot
            seg_frame, all_frame = self.draw_plt(seg_frame, all_frame)

            # Write the frame to the output video
            self.pose_out.write(pose_frame)
            self.pose_white_bg_out.write(pose_white_frame)
            self.seg_out.write(seg_frame)
            self.seg_no_analyze_out.write(seg_no_analyze_frame)
            self.all_out.write(all_frame)
            self.all_no_analyze_out.write(all_no_analyze_frame)

            # append annotation data
            self.append_annotation_data(
                bb_boxes_person, keypoint, score_keypoint, bb_boxes_pad, score_pad, paddle_center, paddle_area
            )

            # Update the progress bar
            progress_bar.update(1)

        # Save results
        self.save_annotation_data()

        # Release resources
        self.cap.release()
        self.pose_out.release()
        self.pose_white_bg_out.release()
        self.seg_out.release()
        self.seg_no_analyze_out.release()
        self.all_out.release()
        self.all_no_analyze_out.release()

    def create_dataset_dir(self):
        base_dir = "hit_datasets"
        self.annos_dir = os.path.join(base_dir, "annotations")
        self.video_dir = os.path.join(base_dir, "videos")
        self.image_dir = os.path.join(self.video_dir, self.input_video_filename)
        self.anno_dir = os.path.join(self.annos_dir, self.input_video_filename)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(self.annos_dir):
            os.makedirs(self.annos_dir)
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.anno_dir):
            os.makedirs(self.anno_dir)


if __name__ == "__main__":
    # Paths
    seg_model_path = "runs/segment/table_tennis_stroke_postures_all_20250314_01/weights/best.pt"

    video_dir = f"inference/videos"
    video_names = os.listdir(video_dir)
    for video_name in video_names:
        input_video_path = f"inference/videos/{video_name}"
        pose_output_video_path = f"inference/output/videos/pose_{video_name}"  # 人體骨架含物件框
        pose_white_bg_output_video_path = (
            f"inference/output/videos/pose_white_bg_{video_name}"  # 人體骨架不含物件框(背景空白)
        )
        seg_output_video_path = f"inference/output/videos/seg_{video_name}"  # 球拍面積分割含分析
        seg_no_analyze_output_video_path = (
            f"inference/output/videos/seg_no_analyze_{video_name}"  # 球拍面積分割不含分析
        )
        all_output_video_path = f"inference/output/videos/all_{video_name}"  # 結合人體骨架和球拍面積含分析
        all_no_analyze_output_video_path = (
            f"inference/output/videos/all_no_analyze_{video_name}"  # 結合人體骨架和球拍面積不含分析
        )

        # Create Predict instance and process the video
        predictor = Predict(seg_model_path)
        predictor.load_video_data(
            input_video_path,
            pose_output_video_path,
            pose_white_bg_output_video_path,
            seg_output_video_path,
            seg_no_analyze_output_video_path,
            all_output_video_path,
            all_no_analyze_output_video_path,
        )
        predictor.create_dataset_dir()
        predictor.process_video()
