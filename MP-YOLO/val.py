import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolo_code\yolov8-X\yolov8-main\runs-fuwuq\train\KITTI\yolov8n-kitti-base\weights\best.pt')
    model.val(data=r'D:\yolo_code\yolov8-X\yolov8-main\dataset\KITTI.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project=r'D:\yolo_code\yolov8-X\yolov8-main\runs\val\SCI-4',
              name='yolov8n-base-KITTI',
              )