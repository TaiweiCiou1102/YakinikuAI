from ultralytics import YOLO

model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    model.train(
        data='./data/YakinikuData/data.yaml',
        epochs=5,
        imgsz=320,
        device='cpu',
        project='runs/detect',
        name='Yakiniku_training'
    )

    print("訓練完成!模型已儲存")