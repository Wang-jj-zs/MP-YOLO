import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolo_code\yolov8-X\yolov8-main\ultralytics\cfg\models\v8\yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:\yolo_code\yolov8-X\yolov8-main\dataset\Big-DAIR.yaml',
                cache=False,
                imgsz=640,
                epochs=3,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                #amp=False,
                name='yolov8n-DAIR-3',
                )