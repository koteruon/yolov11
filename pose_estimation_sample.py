from copy import deepcopy

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator


def plot_pose(ori_left_frame, pose_result_names, pose_result_keypoints):
    img = ori_left_frame  # 1080*960
    img_shape = img.shape[:2]
    annotator = Annotator(
        deepcopy(img),
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        example=pose_result_names,
    )

    # Plot Pose results (預期是 list of [17, 3])
    for k in reversed(pose_result_keypoints):
        annotator.kpts(
            k,
            img_shape,
            radius=10,
            kpt_line=True,
            conf_thres=0.5,
            kpt_color=None,
        )

    return annotator.result()


# 建立空白畫布
canvas = np.full((1080, 960, 3), 255, dtype=np.uint8)

# 定義關鍵點 [x, y, conf]
keypoints_xy = np.array(
    [
        [480, 300],
        [460, 290],
        [500, 290],
        [440, 300],
        [520, 300],
        [400, 400],
        [560, 400],
        [360, 520],
        [600, 520],
        [330, 640],
        [630, 640],
        [420, 640],
        [540, 640],
        [420, 800],
        [540, 800],
        [420, 960],
        [540, 960],
    ],
    dtype=float,
)

# 加入 conf=1.0，變成 [17, 3]
keypoints_full = np.hstack([keypoints_xy, np.ones((17, 1), dtype=float)])

# 包成 list of [17, 3] (支援多人的結構)
pose_keypoints_batch = [keypoints_full]

# 呼叫繪圖函數
pose_left_frame = plot_pose(canvas, ["sample"], pose_keypoints_batch)

# 儲存圖片
cv2.imwrite("pose_sample_output.png", pose_left_frame)
