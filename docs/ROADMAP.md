# Roadmap 備忘

CLAUDE.md 的 6 個開發階段（無硬體 Demo）已全部完成。這裡記錄後續可以做的事與已知限制。

## 已完成
- 第 1–4 階段：推論層、擷取管線、資料層、CLIP 標註 / 情境 / 中文搜尋 / API
- 第 5 階段：前端（上傳、候選確認、素材庫、搜尋）、縮圖、手動修改類別與標籤
- 第 6 階段：Konva 畫布、三種尺寸、6 個版型、設計稿儲存、PNG / JPEG 精確尺寸匯出

## 換到 UGen300（Hailo-10H）時
- 實作 `backend/inference/hailo_backend.py`（`infer` 餵 uint8 NHWC；YOLO-seg / FastSAM 的 HEF 輸出是各尺度原始 tensor，
  需在 `postprocess.py` 補 DFL decode → `HeadOutput`）。
- `config.BACKEND = "hailo"` 後執行 `python -m backend.cli.library reembed` 為全庫重算 CLIP 向量（INT8 與 FP32 不同向量空間）。
- 待確認：UGen300 支援的 OS / 架構、HailoRT 版本；yolov8s-seg、fast_sam_s、CLIP ViT-B/32 是否有 hailo10h HEF。

## 已知限制（之後視需要處理）
- CLIP zero-shot 對卡通角色（咖波）常誤判；使用者可在素材庫手動修正類別。
- 候選確認逐一送出（每個約 1 秒）；一次確認數十個時可考慮改成背景工作。
- 介面為桌面優先，窄螢幕改單欄，未針對手機最佳化。
- 畫布字型使用系統字型；換到沒有微軟正黑體的電腦，畫面與匯出會用替代字型。
- 版型只自動排前 1–2 個素材，其餘素材保持原位，需手動調整。
