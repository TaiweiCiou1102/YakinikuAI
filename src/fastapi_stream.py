import cv2
import numpy as np
import onnxruntime as ort
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = 'best.onnx'
CLASSES = ['Overcooked', 'Undercooked','Well-Cooked']

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", 0)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# 初始化 FastAPI App
app = FastAPI()

def preprocess(frame, input_size):
    # 1. Resize
    img_resized = cv2.resize(frame, (input_size, input_size))

    # 2. BGR -> RGB 
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. Normalize
    img_normalized = img_rgb/255.0

    # 4. Transpose (H,W,C) -> (C, H, W)
    img_transposed = img_normalized.transpose(2,0,1)

    # 5. Expand Dim (1,C,H,W)
    # (Batch Size, Channel, Height, Width)
    img_data = np.expand_dims(img_transposed, axis = 0)

    return img_data.astype(np.float32)

def postprocess(output, conf_thres, iou_thres, orig_w, orig_h, input_size):
    
    predictions = np.transpose(output[0])

    boxes, confidences, class_ids = [], [], []

    for pred in predictions:

        scores = pred[4:]
        max_score = np.max(scores)
        if max_score > CONF_THRESHOLD:
            class_id = np.argmax(scores)
            x, y, w, h = pred[0], pred[1], pred[2], pred[3]

            left = int((x - w/2) * (orig_w / input_size))
            top = int ((y - h/2) * (orig_h / input_size))
            width = int( w *  (orig_w / input_size))
            height = int(h * (orig_h / input_size))

            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((boxes[i], confidences[i], class_ids[i]))
    return results

def box_the_frame(frame, detection):
    box, score, cls_id = detection
    x, y, bw, bh = box
    label_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)

    label = f"{label_name} {score:.2f}"
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.status, self.frame = self.capture.read()
        if not self.status:
            print(f"[Notification] Error: fail to activate the camera.")
            return

        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while True:
            if self.stopped:
                return

            status, frame = self.capture.read()
            if status:
                self.frame = frame
                self.status = status
            else:
                self.stopped = True

    def get_frame(self):

        return self.status, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

def generate_frames(camera):

    prev_time = 0
    while True:
        status, frame = camera.get_frame()
        if not status:
            break
        
        h, w, _ = frame.shape

        # A. 推論
        read_time_0 = time.time()
        input_tensor = preprocess(frame, INPUT_SIZE)
        read_time = (time.time() - read_time_0)*1000

        inf_time_0 = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        inf_time = (time.time()-inf_time_0)*1000

        detections = postprocess(outputs[0], CONF_THRESHOLD, IOU_THRESHOLD, w, h, INPUT_SIZE)
        
        # 計算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        proc_time = (curr_time - prev_time)*1000
        prev_time = curr_time

        # B. 繪圖
        for detection in detections:
            box_the_frame(frame, detection)
            
        info = f"FPS: {fps:2f} | Proc: {proc_time}ms "
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # C. 轉換為 JPG Bytes (這是串流的關鍵)
        # imencode 將 numpy 矩陣編碼為 jpg 格式
        encode_time_0 = time.time()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encode_time = (time.time()- encode_time_0)*1000

        print(f" read | Inf | encode : {read_time} | {inf_time} | {encode_time} ms")
        # D. 輸出 MJPEG 格式
        # 必須包含 Content-Type 和邊界符號
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 1. Load Model
print(f"[Action] Load Model: {MODEL_PATH}")

try:
    session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    print(f"[Notification] Error: Unable to load model. Please check if the path is valid. \n")
    print(f"[Notification] Message: {e}")
    exit()

model_inputs = session.get_inputs()
input_name = model_inputs[0].name

print('[Notification] Model load successfully.')

# 2. Start the image stream
print(f"[Action] Start the multi-threaded camera")

threaded_camera = ThreadedCamera(VIDEO_SOURCE)

time.sleep(1.0)


@app.get("/")
def index():
    return {"message": "Yakiniku AI Server is Running! Go to /video_feed"}

@app.get("/video_feed")
def video_feed():
    # 使用 StreamingResponse 回傳 MJPEG 串流
    return StreamingResponse(generate_frames(threaded_camera), media_type="multipart/x-mixed-replace;boundary=frame")

# --- 7. 程式進入點 ---
if __name__ == "__main__":
    print("啟動 FastAPI Server...")
    uvicorn.run(app, port=8000)