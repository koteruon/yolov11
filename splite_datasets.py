import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

cur_seed = 2

random.seed(cur_seed)
np.random.seed(cur_seed)
torch.manual_seed(cur_seed)
torch.cuda.manual_seed(cur_seed)
torch.cuda.manual_seed_all(cur_seed)

frame_nbr = 32
padding = frame_nbr // 2

stroke_id_map = {
    1: {"padding": ["backhand_chop"], "nonpadding": ["backhand_chop"]},
    2: {"padding": ["backhand_flick"], "nonpadding": ["backhand_flick"]},
    3: {"padding": ["backhand_push"], "nonpadding": ["backhand_push"]},
    4: {"padding": ["backhand_topspin"], "nonpadding": ["backhand_topspin"]},
    5: {"padding": ["forehand_chop"], "nonpadding": ["forehand_chop"]},
    6: {"padding": ["forehand_drive"], "nonpadding": ["forehand_drive"]},
    7: {"padding": ["forehand_smash"], "nonpadding": ["forehand_smash"]},
    8: {"padding": ["forehand_topspin"], "nonpadding": ["forehand_topspin"]},
}

select_frames_dir = "./hit_datasets/select_frame"
select_frames = os.listdir(select_frames_dir)

for select_frame in tqdm(select_frames):
    for stroke_id, stroke_dict in stroke_id_map.items():
        stroke_padding_list = stroke_dict["padding"]
        stroke_padding_name = stroke_padding_list[0]
        stroke_nonpadding_list = stroke_dict["nonpadding"]
        stroke_nonpadding_name = stroke_nonpadding_list[0]
        if stroke_padding_name in select_frame and stroke_nonpadding_name in select_frame:
            match = re.search(r"_(\d+)_annotations.csv", select_frame)
            number = int(match.group(1))
            while len(stroke_padding_list) <= number:
                stroke_padding_list.append(None)
            while len(stroke_nonpadding_list) <= number:
                stroke_nonpadding_list.append(None)
            df_nonpadding = pd.read_csv(os.path.join(select_frames_dir, select_frame))
            # 將範圍分為訓練集 (train) 和測試集 (test)，比例為 8:2 且均勻分配
            num_ranges = len(df_nonpadding)
            # train_indices = np.random.choice(range(num_ranges), size=int(num_ranges * 0.8), replace=False)
            train_indices = list(range(int(num_ranges * 0.8)))
            test_indices = [i for i in range(num_ranges) if i not in train_indices]
            df_nonpadding["category"] = ["train" if i in train_indices else "test" for i in range(num_ranges)]
            # 儲存nonpadding的
            df_nonpadding["start_range"] = df_nonpadding["start_frame"]
            df_nonpadding["end_range"] = df_nonpadding["end_frame"]
            stroke_nonpadding_list[number] = df_nonpadding
            # 保留前後padding的地方
            df_padding = df_nonpadding.copy()
            df_padding["start_range"] = df_padding["start_frame"] - padding
            df_padding["end_range"] = df_padding["end_frame"] + padding
            stroke_padding_list[number] = df_padding


def is_in_range(df, frame_number):
    # 查找數字是否在任何範圍內
    match = df[(df["start_range"] <= frame_number) & (df["end_range"] >= frame_number)]
    if not match.empty:
        # 返回該範圍的類別 (train 或 test)
        return match["category"].iloc[0]
    else:
        # 如果不在任何範圍內，返回 neither
        return "neither"


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
                stroke_list = stroke_id_map[stroke_id]["padding"]
                df = stroke_list[stroke_video_id]
                category = is_in_range(df, stroke_frame_number)
                if category == "train":
                    merged_train_data.append(data)
                elif category == "test":
                    merged_test_data.append(data)
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
                stroke_list = stroke_id_map[stroke_id]["nonpadding"]
                df = stroke_list[stroke_video_id]
                category = is_in_range(df, stroke_frame_number)
                if category == "train":
                    merged_train_data["annotations"].append(annotation)
                elif category == "test":
                    merged_test_data["annotations"].append(annotation)

            # images
            images = datas["images"]
            for image in images:
                image_id = image["id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_list = stroke_id_map[stroke_id]["nonpadding"]
                df = stroke_list[stroke_video_id]
                category = is_in_range(df, stroke_frame_number)
                if category == "train":
                    merged_train_data["images"].append(image)
                elif category == "test":
                    merged_test_data["images"].append(image)

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
                stroke_list = stroke_id_map[stroke_id]["nonpadding"]
                df = stroke_list[stroke_video_id]
                category = is_in_range(df, stroke_frame_number)
                if category == "train":
                    merged_train_data.append(data)
                elif category == "test":
                    merged_test_data.append(data)
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

# 將合併後的資料寫入一個新的 JSON 檔案
with open(os.path.join(output_folder, output_train_file), "w", encoding="utf-8") as f:
    json.dump(merged_train_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_train_file)}")

with open(os.path.join(output_folder, output_test_file), "w", encoding="utf-8") as f:
    json.dump(merged_test_data, f, ensure_ascii=False, indent=4)

print(f"合併完成，輸出檔案：{os.path.join(output_folder,output_test_file)}")
