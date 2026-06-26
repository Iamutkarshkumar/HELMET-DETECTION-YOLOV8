# detect.py
from ultralytics import YOLO
import torch.nn as nn
from spd import replace_conv_with_spd
from ema import EMAAttention

def run_inference(weights="best.pt", source="test/images", project="runs", name="spd_ema_predictions", conf=0.25):
    model = YOLO(weights)

    # RE-APPLY ARCHITECTURE (Crucial for the modified model to work)
    replace_conv_with_spd(model.model, use_modulation=True)
    channels = model.model.model[2].cv2.conv.conv.out_channels if hasattr(model.model.model[2], "cv2") else 64
    model.model.model[2] = nn.Sequential(model.model.model[2], EMAAttention(channels))

    model.predict(source=source, save=True, conf=conf, project=project, name=name, exist_ok=True)

if __name__ == "__main__":
    run_inference()

