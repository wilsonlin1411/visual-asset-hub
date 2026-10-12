# Visual Asset Hub

> Capture once. Reuse everywhere.

把一張照片拆解成多個物件素材（透明 PNG），自動建立類別、屬性、情境與語意向量，存入素材庫；使用者以中文自然語言搜尋素材，再拖入畫布產出多種成品（商品主圖、IG 1080×1350 貼文、Banner）。

## 回答與註解語言
- 回覆使用繁體中文（台灣用語）；技術名詞保留英文。
- 程式碼、變數、檔名維持英文；程式註解可用英文。

## 目前階段：無硬體 Demo
- 目標硬體：ASUS UGen300（Hailo-10H，USB 3.1 Gen2），**目前手上沒有**。
- Demo 先以 ONNX Runtime 在 CPU/GPU 執行，模型規格對齊 Hailo Model Zoo。
- **所有推論都必須經過 `InferenceBackend` 介面**，日後只替換成 `hailo_backend.py`，其他程式不動。
- 推論層只回傳：遮罩、類別、分數、向量。NMS、mask proto 合成等後處理在主機端自行實作（不要依賴 Ultralytics 內建後處理，因為 Hailo 只回傳原始 tensor）。

## 架構
| 層 | 技術 | 職責 |
|---|---|---|
| 前端 | React + Vite | 上傳、分割確認、素材庫、搜尋、畫布（5 個畫面） |
| 應用服務 | Python + FastAPI | 工作佇列、去背合成、遮罩精修、語意標註、向量檢索、成品匯出 |
| 推論 | ONNX Runtime（demo）→ HailoRT（正式） | YOLOv8-seg、FastSAM、CLIP image encoder |
| 資料 | SQLite + sqlite-vec（或 FAISS），檔案存 PNG | 日後可平移至 PostgreSQL + pgvector |

## 兩條管線
1. **擷取管線**：前處理 → YOLO-seg + FastSAM 分割 → 合併去重 → 使用者確認 → 遮罩精修 → 透明 PNG → CLIP 向量 + 標籤 → 寫入素材庫
2. **檢索管線**：中文查詢 →（翻譯或多語模型）→ CLIP text encoder（主機 CPU）→ 向量比對 + 標籤篩選 → 回傳結果，不重新處理影像

## 已確定的設計決策
- **模型**：yolov8s-seg（640×640）、FastSAM-s、CLIP ViT-B/32（512 維）。
- **CLIP text encoder 永遠在主機 CPU 執行**，必須與 image encoder 對齊同一個向量空間。
- **中文查詢**：使用多語 CLIP 文字編碼器（`sentence-transformers/clip-ViT-B-32-multilingual-v1`，蒸餾對齊 OpenAI ViT-B/32）直接編碼中文，不經翻譯；英文 CLIP text encoder 用於標註（zero-shot prompt）。
- **分割合併**：兩模型遮罩以 IoU 去重（閾值設為設定值）；YOLO 與 FastSAM 抓到同一物件時用「YOLO 類別 + FastSAM 遮罩」；面積 < 0.5% 的遮罩過濾；FastSAM 結果無類別，裁切後交 CLIP zero-shot 分類。
- **Objectness**：每個候選以 CLIP 比對「物件 vs 背景」prompt 得 0–1 分數；低於門檻預設隱藏（不刪除），YOLO 有類別的候選一律不隱藏。
- **遮罩精修**：YOLO-seg 遮罩解析度低，合成 PNG 前必須經過 refinement（邊緣 band 內 GrabCut → guided filter → 羽化 → 邊緣顏色校正，可選 matting 模型）。
- **資料層**：SQLite + sqlite-vec（vec0），純 sqlite3 + 薄 repository；標籤用 `asset_tags` 正規化表；向量 BLOB 存在 `embeddings` 表為正本，vec0 為可重建索引。
- **情境推論**：先用規則表（物件共現組合 → 情境標籤），再以整張原圖對情境 prompt 做 CLIP 比對作為補充。
- **向量版本**：每筆向量記錄 `embedding_model` 與 `backend` 欄位；換成 Hailo（INT8 量化）後需支援全庫重新計算向量。
- **前端**：React + Vite + TypeScript、純 CSS；API 一律在 `/api`；畫面只載縮圖（素材 512 WebP、照片 1600、候選 256）。
- **畫布**：Konva（react-konva）；成品由瀏覽器依設計尺寸輸出、上傳後端存檔（後端只檢查像素尺寸）；字型用系統字型；scene JSON 為設計稿正本。
- **VLM（Qwen2-VL 類）屬 Stage II，demo 不做。**

## 目錄結構
```
backend/
  inference/   base.py, onnx_backend.py, hailo_backend.py(stub), postprocess.py
  services/    capture.py, refine.py, tagging.py, search.py, export.py, queue.py
  db/          models.py, vector_index.py
  api/         FastAPI routes
  config.py
frontend/      React + Vite
models/        ONNX 檔（不進 git）
data/          SQLite、PNG（不進 git）
tests/         含測試照片與搜尋案例
docs/SPEC.md   待填規格
```

## 開發順序（每階段完成且可執行後再進下一步）
1. 推論層：`InferenceBackend` 介面 + ONNX backend + 後處理，CLI 能對單張照片輸出遮罩
2. 擷取管線：合併去重、遮罩精修、透明 PNG
3. 資料層：SQLite schema + 向量索引
4. 標註與檢索：CLIP 標籤、情境規則、中文搜尋 API
5. 前端：上傳 / 確認 / 素材庫 / 搜尋
6. 畫布與匯出（Konva 或 Fabric.js）

## 待確認（見 docs/SPEC.md、docs/ROADMAP.md）
- UGen300 支援的 OS / 架構、HailoRT 版本；YOLO-seg、FastSAM、CLIP 是否有 hailo10h 版 HEF
