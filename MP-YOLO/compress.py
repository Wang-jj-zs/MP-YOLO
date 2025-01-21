import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\yolo_code\yolov8-20231217\yolov8-fuben\yolov8-main\runs\train\yolov8n-Rope3D-ASF-P2-Wiou\weights\best.pt',
        'data':r'D:\yolo_code\yolov8-20231217\yolov8-fuben\yolov8-main\dataset\rope.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 4,
        'workers': 4,
        'cache': True,
        'optimizer': 'SGD',
        'amp':False,
        'device': '0',
        'close_mosaic': 20,
        'project':'D:/yolo_code/yolov8-20231217/yolov8-fuben/yolov8-main/runs/prune',
        'name':'yolov8n-asf-p2-lamp-wiou-3060-youlaiyibian-1',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.7,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': r'D:\yolo_code\yolov8-20231217\yolov8-fuben\yolov8-main\ultralytics\cfg\hyp.scratch.sl.yaml',
        'sl_model': None
    }
    
    #prune_model_path = compress(copy.deepcopy(param_dict))
    prune_model_path = r'D:\yolo_code\yolov8-20231217\yolov8-fuben\yolov8-main\runs\prune\yolov8n-asf-p2-lamp-wiou-3060-youlaiyibian-prune\weights\prune.pt'
    finetune(copy.deepcopy(param_dict), prune_model_path)