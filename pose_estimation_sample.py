from copy import deepcopy

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

colors.pose_palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [0, 204, 255],  # [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


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

    annotator.limb_color = colors.pose_palette[[9, 9, 5, 5, 7, 7, 7, 7, 11, 0, 11, 0, 16, 16, 16, 16, 16, 16, 16]]
    annotator.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 7, 7, 11, 0, 11, 0, 7, 7, 9, 5, 9, 5]]

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
        [480, 300],  # 鼻子
        [500, 290],  # 左眼
        [460, 290],  # 右眼
        [520, 300],  # 左耳
        [440, 300],  # 右耳
        [560, 400],  # 左肩
        [400, 400],  # 右肩
        [600, 520],  # 左肘
        [360, 520],  # 右肘
        [630, 640],  # 左手腕
        [330, 640],  # 右手腕
        [540, 640],  # 左臀
        [420, 640],  # 右臀
        [540, 800],  # 左膝
        [420, 800],  # 右膝
        [540, 960],  # 左腳踝
        [420, 960],  # 右腳踝
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
