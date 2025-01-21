import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO(r'D:\yolo_code\yolov8-X\yolov8-main\ultralytics\cfg\models\v8\yolov8-ASF-P2.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()