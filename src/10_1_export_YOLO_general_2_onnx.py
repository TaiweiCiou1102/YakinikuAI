from ultralytics import YOLO

model = YOLO('yolo11n.pt')

print("Transform model to ONNX format...")
success = model.export(format = 'ONNX', dynamic=False, simplify=True)

if success:
    print(f"Transformation success! File is located in {success}")
else:
    print(f"Transformation failed, please check the error message.")
