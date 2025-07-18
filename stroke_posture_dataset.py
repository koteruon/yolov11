import json
import os

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def generate_range(num, frame_nbr):
    right_span = frame_nbr // 2
    left_span = frame_nbr - right_span
    base = (num // 100000) * 100000
    tail_digit = num % 100000
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
    save_path,
):
    train_x = train_skes_joints
    train_y = one_hot_vector(train_labels)
    test_x = valid_skes_joints
    test_y = one_hot_vector(valid_labels)

    np.savez(
        save_path,
        x_train=train_x,
        y_train=train_y,
        x_test=test_x,
        y_test=test_y,
        train_paths=train_paths,
        train_timestamps=train_timestamps,
        test_paths=test_paths,
        test_timestamps=test_timestamps,
    )

    print(f"輸出檔案：{save_path}")


def prepare_stroke_posture_dataset(
    frame_nbr,
    stroke_posture_train_gt_path,
    stroke_posture_test_gt_path,
    stroke_posture_train_person_bbox_kpts,
    stroke_posture_test_person_bbox_kpts,
    dataset_save_path,
):
    def load_data(gt_path, kpts_path):
        paths, timestamps, ske_joints_list, labels = [], [], [], []
        with open(kpts_path, "r", encoding="utf-8") as f_person:
            image_id_map = {p["image_id"]: p["keypoints"] for p in json.load(f_person)}

        with open(gt_path, "r", encoding="utf-8") as f_gt:
            datas = json.load(f_gt)
            images_map = {
                image["id"]: ("data/stroke_postures/videos/" + image["movie"], int(image["timestamp"]))
                for image in datas["images"]
            }

            for annotation in tqdm(datas["annotations"]):
                image_id = annotation["image_id"]
                action_id = annotation["action_ids"]
                skatelon_ids = generate_range(image_id, frame_nbr)
                joints = [image_id_map.get(i, np.zeros((17, 3))) for i in skatelon_ids]
                joints = np.array(joints, dtype=np.float32)[:, :, :2].reshape(frame_nbr, -1)
                assert joints.shape[0] == frame_nbr
                ske_joints_list.append(joints)
                labels.append(action_id)
                path, timestamp = images_map[image_id]
                paths.append(path)
                timestamps.append(timestamp)

        return ske_joints_list, labels, paths, timestamps

    print("處理訓練集...")
    train_ske_joints, train_labels, train_paths, train_timestamps = load_data(
        stroke_posture_train_gt_path, stroke_posture_train_person_bbox_kpts
    )

    print("處理測試集...")
    test_ske_joints, test_labels, test_paths, test_timestamps = load_data(
        stroke_posture_test_gt_path, stroke_posture_test_person_bbox_kpts
    )

    split_dataset(
        train_ske_joints,
        train_labels,
        test_ske_joints,
        test_labels,
        train_paths,
        train_timestamps,
        test_paths,
        test_timestamps,
        dataset_save_path,
    )


if __name__ == "__main__":

    hit_dir = "hit_datasets_test"

    stroke_posture_train_gt_path = os.path.join(hit_dir, "annotations/M-3_01/stroke_postures_train_gt.json")
    stroke_posture_test_gt_path = os.path.join(hit_dir, "annotations/M-3_01/stroke_postures_train_gt.json")
    stroke_posture_train_person_bbox_kpts = os.path.join(
        hit_dir, "annotations/M-3_01/stroke_postures_train_bbox_kpts.json"
    )
    stroke_posture_test_person_bbox_kpts = os.path.join(
        hit_dir, "annotations/M-3_01/stroke_postures_train_bbox_kpts.json"
    )
    dataset_save_path = os.path.join(hit_dir, "annotations/M-3_01/stroke_posture.npz")

    prepare_stroke_posture_dataset(
        32,
        stroke_posture_train_gt_path,
        stroke_posture_test_gt_path,
        stroke_posture_train_person_bbox_kpts,
        stroke_posture_test_person_bbox_kpts,
        dataset_save_path,
    )
