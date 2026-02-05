from ctypes.macholib import framework
from tkinter.tix import STATUS
import onnxruntime as ort
import cv2
import numpy as np
import time
import threading

MODEL_PATH = 'yolo11n.onnx'
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
VIDEO_SOURCE = 0
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

class ThreadedCamera:
    def __init__(self, src=0):
        # Activate the camera
        self.capture = cv2.VideoCapture(src)

        # read the first frame to check if it works
        self.status, self.frame = self.capture.read()
        if not self.status:
            print(f"[Notification] Error: fail to activate the camera.")
            return
        
        # 設定執行緒控制開關
        self.stopped = False
        
        self.thread = threading.Thread(target = self.update, daemon = True)
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

if __name__ == "__main__":

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

    # 3. Real-time Inference

    print(f"[Action] Start real-time inference")

    prev_time = 0

    while True:

        status, frame = threaded_camera.get_frame()
        if not status:
            if not status:
                print("[Notification] Image end.")
                break

        # 為了避免拿到同一張圖重複算 (如果相機 FPS 比推論快)
        # 這裡可以做一個簡單的 copy，確保畫面不會在處理一半時被執行緒改掉
        frame_display = frame.copy()

        h, w, _ = frame_display.shape    

        # A. preprocess
        input_tensor = preprocess(frame, INPUT_SIZE)

        # B. inference
        inf_time_0 = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        inf_time = (time.time()-inf_time_0) * 1000

        # C. post-process
        detections = postprocess(outputs[0], CONF_THRESHOLD, IOU_THRESHOLD, w, h, INPUT_SIZE)

        # D. plot the box
        curr_time = time.time()
        fps = 1/ (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        for detection in detections:
            box_the_frame(frame, detection)

        # E. show the frame
        info = f"FPS: {fps:2f} | Inf: {inf_time:.1f}ms"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('YOLO determinator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    threaded_camera.stop()
    cv2.destroyAllWindows()