# train.py
import os
import sys
import types
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.utils.callbacks.raytune as raytune
from spd import replace_conv_with_spd
from ema import EMAAttention

def train(model_path="yolov8n.pt", data_yaml="data.yaml", epochs=20, batch=16, imgsz=640, project="spd_ema_runs", name="train"):
    
    # 1. Disable WandB
    os.environ["WANDB_MODE"] = "disabled"
    import ultralytics
    ultralytics.utils.callbacks.wb = {}

    # 2. Disable RayTune to prevent crashes
    ray_stub = types.SimpleNamespace(train=types.SimpleNamespace(_internal=types.SimpleNamespace(session=None)))
    sys.modules['ray'] = ray_stub
    def disable_ray_callback(*args, **kwargs): return None
    raytune.on_fit_epoch_end = disable_ray_callback
    raytune.on_train_start = disable_ray_callback
    raytune.on_train_end = disable_ray_callback
    raytune.on_train_epoch_end = disable_ray_callback

    model = YOLO(model_path)

    # 3. Inject modifications
    replace_conv_with_spd(model.model, use_modulation=True)
    
    # Dynamic channel detection (safer than hardcoding 64)
    channels = model.model.model[2].cv2.conv.conv.out_channels if hasattr(model.model.model[2], "cv2") else 64
    model.model.model[2] = nn.Sequential(model.model.model[2], EMAAttention(channels))

    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name=name, exist_ok=True)
    return results

if __name__ == "__main__":
    train()

