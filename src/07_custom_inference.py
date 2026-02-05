from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

IOU = 0.9
model_path = r'runs/detect/Yakiniku_training3/weights/best.pt'

if not os.path.exists(model_path):
    print(f"錯誤:找不到模型檔案 {model_path}")
    exit()

model = YOLO(model_path)

# 訓練資料集中的驗證資料
#image_path = 'source/test1.jpg'
#image_path = 'source/test2.jpg'
# 外部資料
image_path = 'source/IMG_3438.jpg'

if not os.path.exists(image_path):
    print(f"錯誤：找不到測試圖片 {image_path}")
    exit()

print("正在判斷肉熟了沒...")
results = model.predict(source=image_path, save=True, conf=0.3, imgsz=320, iou=0.5, agnostic_nms=True)

result = results[0]
annotated_frame = result.plot()

output_path = f'result_output_251230.jpg'
cv2.imwrite(output_path, annotated_frame)
print(f"圖片已成功儲存至: {output_path}")
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(annotated_frame_rgb)
plt.axis('off')
plt.title(f"Yakiniku AI: Detected {len(result.boxes)} pieces of meat")
plt.show()

# 顯示每個偵測到的物體與信心度
for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"偵測到: {class_name} (信心度: {confidence:.2f})")