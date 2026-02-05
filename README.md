# YakinikuAI

**簡介**

YakinikuAI 專案是為了驗證AI影像辨識實作的可行性而生，目的在於幫助公司內部釐清所需要的所有技術。
YakinikuAI是一套能夠從 **網路攝影機串流** 進行即時推論，判斷燒肉／烤肉的熟度（==Undercooked==、==Well-Cooked==、==Overcooked==），採用 **YOLO 風格** 標準訓練流程、ONNX 優化推論與 FastAPI 串流服務。

## 核心功能與技術要點

- **即時影像推論**：使用 `onnxruntime` 執行 ONNX 模型進行即時推論，並對推論結果做 NMS（非極大值抑制）。

- **多線程攝影機讀取**：以 OpenCV (`cv2.VideoCapture`) 結合 Python `threading` 在背景擷取影格，降低 I/O 緩衝與延遲。

- **MJPEG 串流回傳**：使用 `FastAPI` 的 `StreamingResponse` 回傳 MJPEG 串流，供瀏覽器或監控系統即時觀看。

## 簡要工作流程（How it works）

1. 攝影機擷取影格（`ThreadedCamera`）
2. 影格預處理與轉為模型輸入
3. ONNXRuntime 推論
4. 後處理（置信度過濾、NMS）並在影像上繪製框與標籤
5. 使用 `cv2.imencode` 編碼成 JPEG，透過 `StreamingResponse` 以 MJPEG 形式輸出

## 快速開始（Code-Centric）

1. 環境準備

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install fastapi uvicorn onnxruntime opencv-python numpy python-dotenv
```

2. 準備模型

- 將 `best.onnx` 放於專案根目錄（或修改 `MODEL_PATH`）。

3. 設定來源並啟動：

Windows (cmd):

```cmd
set VIDEO_SOURCE=0
python src\fastapi_stream.py
```

或使用 `uvicorn`：

```bash
uvicorn src.fastapi_stream:app --host 0.0.0.0 --port 8000
```

4. 檢視：在瀏覽器打開 `http://localhost:8000/video_feed`。

## 關鍵檔案（快速連結）

- `src/fastapi_stream.py`：主程式，負責擷取、推論、繪製與串流。[src/fastapi_stream.py](src/fastapi_stream.py#L1-L203)
- `data/YakinikuData/data.yaml`：資料訓練。[data/YakinikuData/data.yaml](data/YakinikuData/data.yaml#L1-L20)
- `src/test/3_export_model.py`：範例模型匯出（YOLO → ONNX）。[src/test/3_export_model.py](src/test/3_export_model.py#L1-L20)

---

# YakinikuAI

**Introduction**

The YakinikuAI project was created to validate the feasibility of implementing AI image recognition, aiming to help the company clarify all necessary technical requirements.
YakinikuAI is a system capable of performing real-time inference from a **webcam stream** to determine the doneness of grilled meat (==Undercooked==, ==Well-Cooked==, ==Overcooked==). It utilizes a standard **YOLO-style** training workflow, ONNX-optimized inference, and FastAPI for streaming services.

## Core Features & Technical Highlights

- **Real-time Image Inference**: Executes ONNX models using `onnxruntime` for real-time inference, including Non-Maximum Suppression (NMS) on the results.
- **Multi-threaded Camera Access**: Combines OpenCV (`cv2.VideoCapture`) with Python `threading` to capture frames in the background, reducing I/O buffering and latency.
- **MJPEG Stream Delivery**: Uses FastAPI's `StreamingResponse` to return an MJPEG stream, allowing for real-time viewing via web browsers or monitoring systems.

## Workflow (How it Works)

1. **Frame Capture**: The `ThreadedCamera` captures video frames.
2. **Preprocessing**: Frames are preprocessed and converted into model input format.
3. **Inference**: ONNXRuntime executes the model prediction.
4. **Post-processing**: Confidence filtering and NMS are applied; bounding boxes and labels are drawn on the image.
5. **Output**: Images are encoded into JPEG via `cv2.imencode` and output as an MJPEG stream through `StreamingResponse`.

## Quick Start (Code-Centric)

### 1. Environment Setup

````bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install fastapi uvicorn onnxruntime opencv-python numpy python-dotenv

### 2. Prepare the Model

- Place `best.onnx` in the project root directory (or modify `MODEL_PATH`).

### 3. Set Source and Launch

**Windows (cmd):**

```cmd
set VIDEO_SOURCE=0
python src\fastapi_stream.py
````

**Or using `uvicorn`:**

```bash
uvicorn src.fastapi_stream:app --host 0.0.0.0 --port 8000
```

### 4. Viewing

Open `http://localhost:8000/video_feed` in your browser.

## Key Files (Quick Links)

- `src/fastapi_stream.py`: Main program responsible for capture, inference, drawing, and streaming. [src/fastapi_stream.py](https://www.google.com/search?q=src/fastapi_stream.py%23L1-L203)
- `data/YakinikuData/data.yaml`: Dataset training configuration. [data/YakinikuData/data.yaml](https://www.google.com/search?q=data/YakinikuData/data.yaml%23L1-L20)
- `src/test/3_export_model.py`: Example of model exportation (YOLO → ONNX). [src/test/3_export_model.py](https://www.google.com/search?q=src/test/3_export_model.py%23L1-L20)
