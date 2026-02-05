#%%
from re import M
import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'runs/detect/Yakiniku_training3/weights/best.onnx'
IMAGE_PATH = 'source/yakiniku_test_2.jpg'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASSES = ['Overcooked', 'Undercooked','Well-Cooked']


#%%
def preprocess_image(image, input_size):
    """
    Turn the format of image as model can recognize.
    """
    # retreive the original size for drawing back
    original_height, original_width = image.shape[:2]
    
    # 1. Resize (縮放到 320x320)
    # YOLO 訓練時通常會用 Letterbox (留黑邊)，這裡為了簡化，我們先用直接縮放
    # 在正式專案中，建議補黑邊以保持長寬比
    img_resized = cv2.resize(image, (input_size, input_size))

    # 2. 顏色格式轉換 (BGR -> RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. normalization (0 ~ 255 -> 0.0 ~ 1.0)
    img_norm = img_rgb/255.0

    # 4. 轉換維度 (H, W, C) -> (C, H, W)
    img_transposed = img_norm.transpose(2,0,1)

    # 5. 增加 Batch 維度 -> (1, 3, 320, 320)
    input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)

    return input_tensor, original_width, original_height

def postprocess(output, conf_thres, iou_thres, orig_w, orig_h, input_size):
    """
    解析 YOLO 的輸出矩陣
    YOLOv8/11 的輸出通常是 (1, 4+cls, 2100) -> (Batch, Box屬性, 預測框數量)
    
    :param output: Description
    :param conf_thres: Description
    :param iou_thres: Description
    :param orig_w: Description
    :param orig_h: Description
    :param input_size: Description
    """

    # 轉置輸出矩陣以便處理: (1, 8400, 7)
    predictions = np.transpose(output[0])

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        # pred 結構: [x_center, y_center, width, height, class0_prob, class1_prob, ...]
        scores = pred[4:]
        max_score = np.max(scores)
        class_id = np.argmax(scores)

        if max_score > conf_thres:
            x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]
            x = int((x_center - w /2) * (orig_w/input_size))
            y = int((y_center - h / 2)*(orig_h/input_size))
            w = int(w * (orig_w / input_size))
            h = int(h * (orig_h / input_size))

            boxes.append([x, y, w, h])
            confidences.append(float(max_score))
            class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "conf": confidences[i],
                "class_id": class_ids[i]
            })
    return results
#%%
if __name__ == "__main__":
    # A. 初始化 ONNX Runtime 引擎
    print(f"載入模型: {MODEL_PATH} ...")
    session = ort.InferenceSession(MODEL_PATH)

    # 取得模型輸入層的資訊 (確認它要多大的圖)
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_size = input_shape[2]
    input_name = model_inputs[0].name
    print(f"模型輸入尺寸: {input_size}x{input_size}")

    # B. 讀取與前處理
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("找不到圖片!")
        exit()
    
    input_tensor, orig_w, orig_h = preprocess_image(image, input_size)

    # C. 推論 (Inference)
    print("正在進行邊緣模擬推論...")
    outputs = session.run(None, {input_name: input_tensor})

    # D. 後處理與繪圖
    detections = postprocess(outputs[0], CONF_THRESHOLD, IOU_THRESHOLD, orig_w, orig_h, input_size)

    print(f"偵測到{len(detections)}個物體")

    #畫圖
    for det in detections:
        x, y, w, h = det['box']
        score = det['conf']
        cls_id = det['class_id']
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)

        print(f" -> {label}: {score:.2f}")

        #畫框框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
#%%   