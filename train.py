import torch
from ultralytics import YOLO

# Load a model
# model = YOLO("./weights_dir/yolo11x-seg.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YOLO("yolo11x-seg.yaml")
# model = YOLO("yolo11l-seg.yaml")
# model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11s-seg.yaml")
# model = YOLO("yolo11n-seg.yaml")

# Train the model
train_results = model.train(
    name="table_tennis_stroke_postures_all_20250314_02",
    data="yolo_datasets/table_tennis_stroke_postures_20250314_02/data.yaml",  # path to dataset YAML
    epochs=50,  # 100 number of training epochs
    imgsz=1920,  # training image size in
    device="0",  # device to run on, e.g., device=0 or device=cpu
    workers=4,  # number of dataloader workers
    batch=1,  # batch size
)

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
