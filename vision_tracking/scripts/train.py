import torch
from ultralytics import YOLO

def main():
    best_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device:", best_device)

    model = YOLO('yolo11n.pt')

    model.train(
        data='vision_tracking/scripts/dataset.yaml',
        project='vision_tracking/runs',
        device=best_device,
        epochs=50, 
        imgsz=640, 
        batch=16, 
        workers=8,
        exist_ok=True,
        pretrained=True
    )

    results = model.val()
    print("Validation Results:", results)

    model.export(format='onnx')

if __name__ == '__main__':
    main()