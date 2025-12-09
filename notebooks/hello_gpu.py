import torch
from ultralytics import YOLO

print("=== Hello from the GPU instance! ===")

# Check PyTorch
print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

# Check YOLO import + model load
print("\nLoading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

print("\nRunning a quick dry-run inference on a dummy tensor...")
dummy = torch.zeros((1, 3, 640, 640)).cuda()
results = model(dummy)
print("Inference done!")

print("\n=== All sanity checks passed! ===")

