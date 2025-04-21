import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

cur_seed = 2

random.seed(cur_seed)
np.random.seed(cur_seed)
torch.manual_seed(cur_seed)
torch.cuda.manual_seed(cur_seed)
torch.cuda.manual_seed_all(cur_seed)

frame_nbr = 32
padding = frame_nbr // 2
split_radio = 0.6

stroke_id_map = {
    1: {"stroke_list": ["backhand_chop"], "videos": [None]},
    2: {"stroke_list": ["backhand_flick"], "videos": [None]},
    3: {"stroke_list": ["backhand_push"], "videos": [None]},
    4: {"stroke_list": ["backhand_topspin"], "videos": [None]},
    5: {"stroke_list": ["forehand_chop"], "videos": [None]},
    6: {"stroke_list": ["forehand_drive"], "videos": [None]},
    7: {"stroke_list": ["forehand_smash"], "videos": [None]},
    8: {"stroke_list": ["forehand_topspin"], "videos": [None]},
}


def is_in_range(frame_number, stroke_video_info, is_annotation=False):
    # 可能超出頭尾
    if is_annotation:
        min_frame = stroke_video_info["min_frame"]
        max_frame = stroke_video_info["max_frame"]
        if frame_number < min_frame + padding or frame_number > max_frame - padding - 1:
            return []
    # 其他類別
    if is_annotation:
        last_train_end = video_map["split_frame"]
        first_test_start = video_map["split_frame"] + 1
    else:
        last_train_end = video_map["split_frame"] + padding
        first_test_start = video_map["split_frame"] + 1 - padding
    result = []
    if frame_number <= last_train_end:
        result.append("train")
    if frame_number >= first_test_start:
        result.append("test")
    return result


videos_dir = "./hit_datasets/videos"
videos = os.listdir(videos_dir)

for video in tqdm(videos):
    for stroke_id, stroke_map in stroke_id_map.items():
        stroke_list = stroke_map["stroke_list"]
        stroke_name = stroke_list[0]
        if stroke_name in video:
            video_path = os.path.join(videos_dir, video)
            numbers = [int(f[:-4]) for f in os.listdir(video_path) if f.endswith(".png") and f[:-4].isdigit()]
            assert len(numbers) > 0
            video_map = {}
            min_frame = min(numbers)
            max_frame = max(numbers)
            total = max_frame - min_frame + 1
            split_index = int(total * 0.8)
            split_frame = min_frame + split_index - 1
            video_map["min_frame"] = min_frame
            video_map["max_frame"] = max_frame
            video_map["split_frame"] = split_frame
            stroke_map["videos"].append(video_map)


# -------------------------------person_bbox_kpts--------------------------------

# 設定資料夾路徑（請換成你存放 JSON 檔案的資料夾）
input_folder = "./hit_datasets/annotations"
output_folder = "./hit_datasets/annotations"
output_train_file = "stroke_postures_train_person_bbox_kpts.json"
output_test_file = "stroke_postures_test_person_bbox_kpts.json"

# 建立一個空 list 來儲存所有資料
merged_train_data = []
merged_test_data = []

# 遍歷資料夾中的每一個檔案
folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
for filename in tqdm(folders):
    input_dir = os.path.join(input_folder, filename)
    file_path = os.path.join(input_dir, "stroke_postures_train_bbox_kpts.json")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            datas = json.load(f)
            for data in datas:
                image_id = data["image_id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                categories = is_in_range(stroke_frame_number, stroke_video_info, is_annotation=False)
                for category in categories:
                    if category == "train":
                        merged_train_data.append(data)
                    elif category == "test":
                        merged_test_data.append(data)
                    else:
                        raise ValueError(f"未預期的類別：{category}")
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

# 將合併後的資料寫入一個新的 JSON 檔案
with open(os.path.join(output_folder, output_train_file), "w", encoding="utf-8") as f:
    json.dump(merged_train_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_train_file)}")

with open(os.path.join(output_folder, output_test_file), "w", encoding="utf-8") as f:
    json.dump(merged_test_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_test_file)}")


# -------------------------------gt--------------------------------

# 設定資料夾路徑（請換成你存放 JSON 檔案的資料夾）
input_folder = "./hit_datasets/annotations"
output_folder = "./hit_datasets/annotations"
output_train_file = "stroke_postures_train_gt.json"
output_test_file = "stroke_postures_test_gt.json"

# 建立一個空 list 來儲存所有資料
existing_category_ids = set()
merged_train_data = {"categories": [], "annotations": [], "images": []}
merged_test_data = {"categories": [], "annotations": [], "images": []}

# 遍歷資料夾中的每一個檔案
folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
for filename in tqdm(folders):
    input_dir = os.path.join(input_folder, filename)
    file_path = os.path.join(input_dir, "stroke_postures_train_gt.json")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            datas = json.load(f)
            # categories
            categories = datas["categories"]
            for category in categories:
                category_id = category["id"]
                if category_id not in existing_category_ids:
                    merged_train_data["categories"].append(category)
                    merged_test_data["categories"].append(category)
                    existing_category_ids.add(category_id)

            # annotations
            annotations = datas["annotations"]
            for annotation in annotations:
                image_id = annotation["image_id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                categories = is_in_range(stroke_frame_number, stroke_video_info, is_annotation=True)
                for category in categories:
                    if category == "train":
                        merged_train_data["annotations"].append(annotation)
                    elif category == "test":
                        merged_test_data["annotations"].append(annotation)
                    else:
                        raise ValueError(f"未預期的類別：{category}")

            # images
            images = datas["images"]
            for image in images:
                image_id = image["id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                categories = is_in_range(stroke_frame_number, stroke_video_info, is_annotation=True)
                for category in categories:
                    if category == "train":
                        merged_train_data["images"].append(image)
                    elif category == "test":
                        merged_test_data["images"].append(image)
                    else:
                        raise ValueError(f"未預期的類別：{category}")

        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

# 將合併後的資料寫入一個新的 JSON 檔案
with open(os.path.join(output_folder, output_train_file), "w", encoding="utf-8") as f:
    json.dump(merged_train_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_train_file)}")

with open(os.path.join(output_folder, output_test_file), "w", encoding="utf-8") as f:
    json.dump(merged_test_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_test_file)}")


# -------------------------------gt--------------------------------

# 設定資料夾路徑（請換成你存放 JSON 檔案的資料夾）
input_folder = "./hit_datasets/annotations"
output_folder = "./hit_datasets/annotations"
output_train_file = "stroke_postures_train_object_detection.json"
output_test_file = "stroke_postures_test_object_detection.json"

# 建立一個空 list 來儲存所有資料
merged_train_data = []
merged_test_data = []

# 遍歷資料夾中的每一個檔案
folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
for filename in tqdm(folders):
    input_dir = os.path.join(input_folder, filename)
    file_path = os.path.join(input_dir, "stroke_postures_train_object_detection.json")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            datas = json.load(f)
            for data in datas:
                image_id = data["image_id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                categories = is_in_range(stroke_frame_number, stroke_video_info, is_annotation=False)
                for category in categories:
                    if category == "train":
                        merged_train_data.append(data)
                    elif category == "test":
                        merged_test_data.append(data)
                    else:
                        raise ValueError(f"未預期的類別：{category}")
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

# 將合併後的資料寫入一個新的 JSON 檔案
with open(os.path.join(output_folder, output_train_file), "w", encoding="utf-8") as f:
    json.dump(merged_train_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_train_file)}")

with open(os.path.join(output_folder, output_test_file), "w", encoding="utf-8") as f:
    json.dump(merged_test_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_test_file)}")


# -------------------------------skateformer--------------------------------


def generate_range(num):
    right_span = frame_nbr // 2
    left_span = frame_nbr - right_span
    base = (num // 100000) * 100000  # 取得數字所屬的百位區間
    tail_digit = num % 100000  # 取得尾數

    min_val = tail_digit - left_span + 1
    max_val = tail_digit + right_span
    valid_numbers = np.arange(min_val, max_val + 1) + base

    return np.pad(valid_numbers, (0, frame_nbr - len(valid_numbers)), constant_values=-1).tolist()


def one_hot_vector(labels):
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(labels)


def split_dataset(train_skes_joints, train_labels, valid_skes_joints, valid_labels):
    train_x = train_skes_joints
    train_y = one_hot_vector(train_labels)
    test_x = valid_skes_joints
    test_y = one_hot_vector(valid_labels)

    save_name = "./hit_datasets/annotations/stroke_posture.npz"
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    print(f"輸出檔案：{save_name}")


stroke_posture_train_gt_path = "./hit_datasets/annotations/stroke_postures_train_gt.json"
stroke_posture_test_gt_path = "./hit_datasets/annotations/stroke_postures_test_gt.json"
stroke_posture_train_person_bbox_kpts = "./hit_datasets/annotations/stroke_postures_train_person_bbox_kpts.json"
stroke_posture_test_person_bbox_kpts = "./hit_datasets/annotations/stroke_postures_test_person_bbox_kpts.json"

train_ske_joints = []
train_labels = []
with open(stroke_posture_train_person_bbox_kpts, "r", encoding="utf-8") as f_person:
    image_id_map_skatelon = {}
    try:
        persons = json.load(f_person)
        for person in tqdm(persons):
            image_id = person["image_id"]
            skatelon = person["keypoints"]
            image_id_map_skatelon[image_id] = skatelon
    except json.JSONDecodeError as e:
        print(f"讀取錯誤：{filename} - {e}")

    with open(stroke_posture_train_gt_path, "r", encoding="utf-8") as f_gt:
        try:
            datas = json.load(f_gt)
            annotations = datas["annotations"]
            for annotation in tqdm(annotations):
                image_id = annotation["image_id"]
                action_id = annotation["action_ids"]
                ske_joints = []
                skatelon_image_id_list = generate_range(image_id)
                for skatelon_image_id in skatelon_image_id_list:
                    ske_joints.append(image_id_map_skatelon[skatelon_image_id])
                ske_joints = np.array(ske_joints, dtype=np.float32)
                assert len(ske_joints) == frame_nbr
                ske_joints = ske_joints[:, :, :2]
                ske_joints = ske_joints.reshape(frame_nbr, -1)
                train_ske_joints.append(ske_joints)
                train_labels.append(action_id)
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

test_ske_joints = []
test_labels = []
with open(stroke_posture_test_person_bbox_kpts, "r", encoding="utf-8") as f_person:
    image_id_map_skatelon = {}
    try:
        persons = json.load(f_person)
        for person in tqdm(persons):
            image_id = person["image_id"]
            skatelon = person["keypoints"]
            image_id_map_skatelon[image_id] = skatelon
    except json.JSONDecodeError as e:
        print(f"讀取錯誤：{filename} - {e}")

    with open(stroke_posture_test_gt_path, "r", encoding="utf-8") as f_gt:
        try:
            datas = json.load(f_gt)
            annotations = datas["annotations"]
            for annotation in tqdm(annotations):
                image_id = annotation["image_id"]
                action_id = annotation["action_ids"]
                ske_joints = []
                skatelon_image_id_list = generate_range(image_id)
                for skatelon_image_id in skatelon_image_id_list:
                    ske_joints.append(image_id_map_skatelon[skatelon_image_id])
                ske_joints = np.array(ske_joints, dtype=np.float32)
                assert len(ske_joints) == frame_nbr
                ske_joints = ske_joints[:, :, :2]
                ske_joints = ske_joints.reshape(frame_nbr, -1)
                test_ske_joints.append(ske_joints)
                test_labels.append(action_id)
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

split_dataset(train_ske_joints, train_labels, test_ske_joints, test_labels)
