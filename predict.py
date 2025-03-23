from ultralytics import YOLO

# Load a pretrained YOLO11n-seg model
model = YOLO("runs/segment/table_tennis_stroke_postures_202501142/weights/best.pt")

# Define paths
input_video_path = "inference/videos/forehand_drive_01.mp4"

# 對影片執行推論，並自動儲存結果
results = model(input_video_path, save=True)
