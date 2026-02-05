from ultralytics import YOLO

model = YOLO('runs/detect/Yakiniku_training3/weights/best.pt')

print("Transform model to ONNX format...")
success = model.export(format='onnx', dynamic=False, simplify=True)

if success:
    print(f'Transformation success! File is located in {success}')
else:
    print("Transformation failed, please check the error message.")