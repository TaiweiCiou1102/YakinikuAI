from ultralytics import YOLO
import cv2 
import matplotlib.pyplot as plt

model = YOLO('yolo11n.pt')

results = model.predict(source='./source/busy_street.jpg', save=True, conf = 0.5)

result = results[0]

annotated_frame = result.plot()

annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,8))
plt.imshow(annotated_frame_rgb)
plt.axis('off')
plt.title(f"Detected {len(result.boxes)} objects")
plt.show()