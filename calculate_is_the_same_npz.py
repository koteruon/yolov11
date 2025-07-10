import numpy as np

# 載入兩個 npz 檔案
file1 = np.load("hit_datasets/annotations/stroke_posture.npz")
file2 = np.load("hit_datasets_now_used/annotations/stroke_posture.npz")

# 取得包含 "test" 的 key
test_keys1 = [k for k in file1.files if "test" in k]
test_keys2 = [k for k in file2.files if "test" in k]

# 取得兩者都存在的 test key
common_test_keys = set(test_keys1) & set(test_keys2)

# 比較每個共同的 test key
all_match = True
for key in sorted(common_test_keys):
    arr1 = file1[key]
    arr2 = file2[key]
    if np.array_equal(arr1, arr2):
        print(f"✅ Key '{key}' 完全一致")
    else:
        print(f"❌ Key '{key}' 不一致")
        all_match = False

# 額外提示
if not common_test_keys:
    print("兩個檔案之間沒有共同的 'test' 類 key")
elif all_match:
    print("🎉 所有包含 'test' 的 key 都完全一致！")
else:
    print("⚠️ 有些 'test' key 對應的資料不相同")
