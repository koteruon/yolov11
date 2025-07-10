import json
from collections import Counter

# 讀取 JSON 檔案
json_path = "hit_datasets/annotations/stroke_postures_train_gt.json"
with open(json_path, "r") as f:
    data = json.load(f)

# 統計 action_ids 中從 1~9 的數量
counter = Counter()
for ann in data["annotations"]:
    for action_id in ann.get("action_ids", []):
        if 1 <= action_id <= 9:
            counter[action_id] += 1

# 印出統計結果（從 1~9）
for i in range(1, 10):
    print(f"action_id 為 {i} 的數量：{counter[i]}")
