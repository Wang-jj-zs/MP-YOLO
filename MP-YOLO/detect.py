import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolo_code\yolov8-X\yolov8-main\runs\train\MP-YOLO\weights\best.pt') # select your model.pt path
    model.predict(source=r'D:\yolo_code\yolov8-X\yolov8-main\image\gaok3.jpg',
                  imgsz=640,
                  project=r'D:\yolo_code\yolov8-X\yolov8-main\runs\images',
                  name=r'gaok3-detect',
                  save=True,
                )