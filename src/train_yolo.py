from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_baseline(
    data_config: str = "configs/soda.yaml",
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
):
    
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(PROJECT_ROOT / data_config),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        device=0,
        # modifed for test 3
        multi_scale=True,
        scale=0.9,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4
    )


if __name__ == "__main__":
    train_baseline()
