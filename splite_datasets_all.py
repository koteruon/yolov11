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
split_radio_train = 0.8
split_radio_test = 0.2

stroke_id_map = {
    1: {"stroke_list": ["backhand_chop"], "videos": [None], "test_count": 10},
    2: {"stroke_list": ["backhand_flick"], "videos": [None], "test_count": 9},
    3: {"stroke_list": ["backhand_push"], "videos": [None], "test_count": 13},
    4: {"stroke_list": ["backhand_topspin"], "videos": [None], "test_count": 7},
    5: {"stroke_list": ["forehand_chop"], "videos": [None], "test_count": 9},
    6: {"stroke_list": ["forehand_drive"], "videos": [None], "test_count": 5},
    7: {"stroke_list": ["forehand_smash"], "videos": [None], "test_count": 8},
    8: {"stroke_list": ["forehand_topspin"], "videos": [None], "test_count": 13},
}

select_frames_dir = "./hit_datasets/select_frame"
select_frames = os.listdir(select_frames_dir)

for select_frame in tqdm(select_frames):
    for stroke_id, stroke_map in stroke_id_map.items():
        stroke_list = stroke_map["stroke_list"]
        stroke_name = stroke_list[0]
        if stroke_name in select_frame:
            match = re.search(r"_(\d+)_annotations.csv", select_frame)
            number = int(match.group(1))
            while len(stroke_list) <= number:
                stroke_list.append(None)
            df = pd.read_csv(os.path.join(select_frames_dir, select_frame))
            # 將範圍分為訓練集 (train) 和測試集 (test)，比例為 8:2 且均勻分配
            num_ranges = len(df)
            test_count = stroke_map["test_count"]

            # train_indices = np.random.choice(range(num_ranges), size=int(num_ranges * split_radio), replace=False)
            # train_indices = list(range(int(num_ranges * split_radio_train)))
            # no_test_indices = list(range(int(num_ranges * (1 - split_radio_test))))
            # test_indices = [i for i in range(num_ranges) if i not in no_test_indices]
            test_indices = list(range(num_ranges - test_count, num_ranges))
            train_indices = list(range(0, num_ranges - test_count))

            df["category"] = [
                "train" if i in train_indices else "test" if i in test_indices else "other" for i in range(num_ranges)
            ]
            # 儲存nonpadding的
            df["start_range"] = df["start_frame"]
            df["end_range"] = df["end_frame"]
            # 取得關鍵值
            last_train_end = df.loc[train_indices[-1], "end_range"]
            first_test_start = df.loc[test_indices[0], "start_range"]
            df["last_train_end"] = last_train_end
            df["first_test_start"] = first_test_start
            stroke_list[number] = df


def is_in_range(df, frame_number, stroke_video_info, is_annotation=False):
    # 可能超出頭尾
    if is_annotation:
        min_frame = stroke_video_info["min_frame"]
        max_frame = stroke_video_info["max_frame"]
        if frame_number < min_frame + padding or frame_number > max_frame - padding:
            return []
    # 查找數字是否在任何範圍內
    match = df[(df["start_range"] <= frame_number) & (df["end_range"] >= frame_number)]
    # 返回該範圍的類別 (train 或 test)
    if not match.empty:
        result = [match["category"].iloc[0]]
        if result[0] == "train" and frame_number >= df["first_test_start"].iloc[0]:
            result.append("test_other")
        if result[0] == "test" and frame_number <= df["last_train_end"].iloc[0]:
            result.append("train_other")
        if result[0] == "other":
            result.remove("other")
        return result
    # 其他類別
    if is_annotation:
        last_train_end = df["last_train_end"].iloc[0]
        first_test_start = df["first_test_start"].iloc[0]
    else:
        last_train_end = df["last_train_end"].iloc[0] + padding
        first_test_start = df["first_test_start"].iloc[0] - padding
    if frame_number <= last_train_end:
        return ["train_other"]
    elif frame_number >= first_test_start:
        return ["test_other"]
    else:
        return []


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
            video_map["min_frame"] = min_frame
            video_map["max_frame"] = max_frame
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
                df = stroke_list[stroke_video_id]
                categories = is_in_range(df, stroke_frame_number, stroke_video_info, is_annotation=False)
                for category in categories:
                    if category == "train" or category == "train_other":
                        merged_train_data.append(data)
                    elif category == "test" or category == "test_other":
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
            train_data_len = 0
            test_data_len = 0
            merge_train_other_data = []
            merge_test_other_data = []
            annotations = datas["annotations"]
            for annotation in annotations:
                image_id = annotation["image_id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                df = stroke_list[stroke_video_id]
                categories = is_in_range(df, stroke_frame_number, stroke_video_info, is_annotation=True)
                for category in categories:
                    if category == "train":
                        train_data_len += 1
                        merged_train_data["annotations"].append(annotation)
                    elif category == "test":
                        test_data_len += 1
                        merged_test_data["annotations"].append(annotation)
                    elif category == "train_other":
                        annotation["action_ids"][0] = 9
                        # merged_train_data["annotations"].append(annotation)
                        merge_train_other_data.append(annotation)
                    elif category == "test_other":
                        annotation["action_ids"][0] = 9
                        merged_test_data["annotations"].append(annotation)
                        # merge_test_other_data.append(annotation)
                    else:
                        raise ValueError(f"未預期的類別：{category}")
            train_data_samples = train_data_len  # // len(stroke_id_map)
            merge_train_other_data_len = len(merge_train_other_data)
            train_indices = np.round(np.linspace(0, merge_train_other_data_len - 1, train_data_samples)).astype(int)
            selected_train_other_data = [merge_train_other_data[i] for i in train_indices]
            merged_train_data["annotations"].extend(selected_train_other_data)
            # test_data_samples = test_data_len // len(stroke_id_map)
            # merge_test_other_data_len = len(merge_test_other_data)
            # test_indices = np.round(np.linspace(0, merge_test_other_data_len - 1, test_data_samples)).astype(int)
            # selected_test_other_data = [merge_test_other_data[i] for i in test_indices]
            # merged_test_data["annotations"].extend(selected_test_other_data)
            # images
            images = datas["images"]
            for image in images:
                image_id = image["id"]
                stroke_id = image_id // 100000000
                stroke_video_id = (image_id % 100000000) // 100000
                stroke_frame_number = image_id % 100000
                stroke_video_info = stroke_id_map[stroke_id]["videos"][stroke_video_id]
                stroke_list = stroke_id_map[stroke_id]["stroke_list"]
                df = stroke_list[stroke_video_id]
                categories = is_in_range(df, stroke_frame_number, stroke_video_info, is_annotation=True)
                for category in categories:
                    if category == "train" or category == "train_other":
                        merged_train_data["images"].append(image)
                    elif category == "test" or category == "test_other":
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
                df = stroke_list[stroke_video_id]
                categories = is_in_range(df, stroke_frame_number, stroke_video_info, is_annotation=False)
                for category in categories:
                    if category == "train" or category == "train_other":
                        merged_train_data.append(data)
                    elif category == "test" or category == "test_other":
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


def split_dataset(
    train_skes_joints,
    train_labels,
    valid_skes_joints,
    valid_labels,
    train_paths,
    train_timestamps,
    test_paths,
    test_timestamps,
):
    train_x = train_skes_joints
    train_y = one_hot_vector(train_labels)
    test_x = valid_skes_joints
    test_y = one_hot_vector(valid_labels)

    save_name = "./hit_datasets/annotations/stroke_posture.npz"
    np.savez(
        save_name,
        x_train=train_x,
        y_train=train_y,
        x_test=test_x,
        y_test=test_y,
        train_paths=train_paths,
        train_timestamps=train_timestamps,
        test_paths=test_paths,
        test_timestamps=test_timestamps,
    )

    print(f"輸出檔案：{save_name}")


stroke_posture_train_gt_path = "./hit_datasets/annotations/stroke_postures_train_gt.json"
stroke_posture_test_gt_path = "./hit_datasets/annotations/stroke_postures_test_gt.json"
stroke_posture_train_person_bbox_kpts = "./hit_datasets/annotations/stroke_postures_train_person_bbox_kpts.json"
stroke_posture_test_person_bbox_kpts = "./hit_datasets/annotations/stroke_postures_test_person_bbox_kpts.json"

train_paths = []
train_timestamps = []
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
            images = datas["images"]
            images_map = {}
            for image in tqdm(images):
                image_id = image["id"]
                movie = image["movie"]
                path = "data/stroke_postures/videos/" + movie
                timestamp = int(image["timestamp"])
                images_map[image_id] = (path, timestamp)

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
                path, timestamp = images_map[image_id]
                train_paths.append(path)
                train_timestamps.append(timestamp)
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

test_paths = []
test_timestamps = []
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
            images = datas["images"]
            images_map = {}
            for image in tqdm(images):
                image_id = image["id"]
                movie = image["movie"]
                path = "data/stroke_postures/videos/" + movie
                timestamp = int(image["timestamp"])
                images_map[image_id] = (path, timestamp)

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
                path, timestamp = images_map[image_id]
                test_paths.append(path)
                test_timestamps.append(timestamp)
        except json.JSONDecodeError as e:
            print(f"讀取錯誤：{filename} - {e}")

split_dataset(
    train_ske_joints,
    train_labels,
    test_ske_joints,
    test_labels,
    train_paths,
    train_timestamps,
    test_paths,
    test_timestamps,
)
