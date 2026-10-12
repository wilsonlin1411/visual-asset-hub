# Visual Asset Hub

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/wilson1411/visual-asset-hub)

**線上 Demo（免安裝）**：https://wilson1411-visual-asset-hub.hf.space

每位訪客有自己的素材庫（已放入 10 張示範照片），閒置 60 分鐘後自動刪除。一段時間沒人使用時 Demo 會休眠，第一次開啟可能要等 1–2 分鐘。

> Capture once. Reuse everywhere. 設計與規格見 [CLAUDE.md](CLAUDE.md)、[docs/SPEC.md](docs/SPEC.md)。
>
> **使用者文件**：網頁操作 → [docs/USER_GUIDE.md](docs/USER_GUIDE.md)；在新電腦安裝與啟動 → [docs/SETUP_NEW_PC.md](docs/SETUP_NEW_PC.md)
>
> **安裝**：模型（約 1.2 GB）與建置好的網頁不在 git 裡，請從 **Releases** 下載 `vah-models.zip`、`vah-frontend-dist.zip`，
> 在專案根目錄用 `tar -xf` 解壓後即可啟動；完整步驟見 [docs/SETUP_NEW_PC.md](docs/SETUP_NEW_PC.md)。

## 參賽資訊

| 項目 | 內容 |
|---|---|
| 競賽 | ASUS UGen AI League 2026 · Innovation Hackathon · Stage I Proposal |
| 組別 | Battlefield Lightning · Lightweight Computing（輕量運算組） |
| 目標硬體 | ASUS UGen300 USB AI Accelerator（Hailo-10H） |
| Stage I | Simulated and validated on a laptop（CPU only） |

## 團隊成員

| 成員 | GitHub |
|---|---|
| 林源隆（Wilson Lin） | [@wilsonlin1411](https://github.com/wilsonlin1411) |
| 禤蔚灝（Asa Huen） | [@Asahuen](https://github.com/Asahuen) |

## 環境

兩個 conda 環境，runtime 不依賴 torch / ultralytics：

| 環境 | 用途 | 安裝 |
|---|---|---|
| `vah` | App runtime（推論、API） | `pip install -r requirements.txt` |
| `vah-export` | 匯出 ONNX、對照驗證 | torch CPU + `pip install -r requirements-export.txt` |

```bash
conda create -n vah python=3.11 && conda run -n vah pip install -r requirements.txt
```

## 模型匯出（只需一次）

```bash
conda run -n vah-export python tools/export_models.py
```
產生 `models/yolov8s-seg.onnx`（1×3×640×640 → `output0 [1,116,8400]`、`output1 [1,32,160,160]`）。

## 第 1 階段：單張照片分割

```bash
conda run -n vah python -m backend.cli.segment tests/images --out outputs --device cpu
```
每張照片輸出 `*_overlay.jpg`（遮罩視覺化）、`*_masks/*.png`（每個物件的 0/255 遮罩）、`*_detections.json`。

驗證：
```bash
conda run -n vah python -m pytest tests -q
conda run -n vah-export python tools/compare_ultralytics.py tests/images
```

## 第 2 階段：擷取管線（候選 → 勾選 → 透明 PNG）

```bash
conda run -n vah python -m backend.cli.capture tests/images --out outputs/capture
conda run -n vah python -m backend.cli.capture tests/images/pudding.jpg --select 0,1,3
```
每張照片在 `outputs/capture/<圖名>/` 輸出 `candidates.jpg`（編號候選；`?` = FastSAM 無類別）、
`candidates.json`（含被剔除項目與原因、各步驟耗時）、`assets/*.png`、`assets_preview.jpg`（棋盤格預覽）。
最後印出與 SPEC 時間目標（產生候選 ≤ 5 s、每個 PNG ≤ 3 s）的比對。

流程：YOLO-seg + FastSAM → 面積/碎片過濾 → 融合（IoU ≥ 0.7：YOLO 類別 + FastSAM 遮罩）→ 去重 →
精修（GrabCut 修正邊緣 band → guided filter 軟邊 → 羽化）→ 邊緣顏色校正 → RGBA PNG。
門檻都在 `backend/config.py` 的 `FASTSAM`、`MergeConfig`、`RefineConfig`。

## 第 3 階段：素材庫（SQLite + sqlite-vec）

```bash
conda run -n vah python -m backend.cli.library ingest tests/images        # 匯入照片 + 存候選
conda run -n vah python -m backend.cli.library candidates 2
conda run -n vah python -m backend.cli.library confirm 2 --select 0,1,3   # 或 --labeled / --all
conda run -n vah python -m backend.cli.library list --category bowl
conda run -n vah python -m backend.cli.library delete-asset 9             # 候選回到 pending
conda run -n vah python -m backend.cli.library stats
conda run -n vah python tools/bench_vector_index.py --n 10000
```
資料在 `data/`（`vah.db`、`photos/`、`candidates/`、`assets/`），schema 見 [docs/SPEC.md](docs/SPEC.md) §4。
候選存在資料庫，所以「看候選」和「確認」可以在不同 process / request 進行；確認時不載入分割模型。

## 第 4 階段：CLIP 標註、情境、中文搜尋、API

模型（只需匯出一次，約 1.2 GB）：
```bash
conda run -n vah-export python tools/export_clip.py          # CLIP 圖像 / 英文文字 / 多語文字編碼器 → models/
conda run -n vah-export python tools/check_clip_runtime.py   # 與 transformers / sentence-transformers 比對
```
素材庫：
```bash
conda run -n vah python -m backend.cli.library retag                  # 詞彙表修改後重新標註全部
conda run -n vah python -m backend.cli.library search "可愛的公仔" "適合下班小酌的東西"
conda run -n vah python -m backend.cli.library tags 126               # 看單一素材的類別與標籤
conda run -n vah python -m backend.api                                # http://127.0.0.1:8000/api/docs
```
評估與校準（先 `library ingest tests/images` 建立示範素材庫）：
```bash
conda run -n vah python tools/seed_library.py         # 模擬使用者勾選明顯物件（建立示範素材庫）
conda run -n vah python tools/eval_search.py          # tests/search_cases.toml → hit@5、MRR（案例以檔名指定照片）
conda run -n vah python tools/objectness_report.py    # objectness 分布 + 門檻附近候選縮圖
```
詞彙表在 `backend/vocab/vocabulary.toml`（與 docs/SPEC.md §3 同步）。

## 第 5 階段：前端（上傳 / 候選確認 / 素材庫 / 搜尋）

React 18 + Vite + TypeScript，純 CSS；Node.js 在 conda 環境 `vah-web`（`conda create -n vah-web -c conda-forge nodejs=22`）。

**Demo（單一 port）**：
```bash
cd frontend && conda run -n vah-web npm install && conda run -n vah-web npm run build
conda run -n vah python -m backend.api        # 開 http://localhost:8000
```
**開發**（前端熱更新）：
```bash
conda run -n vah python -m backend.api               # API :8000
cd frontend && conda run -n vah-web npm run dev      # http://localhost:5173（/api 轉給 :8000）
```
檢查：`npm run typecheck`、`npm test`（Vitest）。既有素材補縮圖：`python -m backend.cli.library thumbs`。

| 畫面 | 路徑 | 重點 |
|---|---|---|
| 上傳 | `/upload` | 拖放多張；背景處理狀態與佇列順位自動更新 |
| 候選確認 | `/photos/:id` | 照片上疊遮罩，點照片或清單選取；低分候選收合；逐一確認顯示進度 |
| 素材庫 | `/library` | 類別 / 標籤 / 照片篩選；詳情面板可改類別、增刪標籤、下載 PNG、刪除 |
| 搜尋 | `/search?q=` | 中文查詢、解析結果、分數與命中原因 |

畫面只載入縮圖（素材 512 px WebP、照片 1600 px、候選 256 px），存在 `data/thumbs/`。

## 第 6 階段：畫布與匯出

`/designs` 選尺寸（商品主圖 800×800、IG 貼文 1080×1350、Banner 1640×856）建立設計 → 編輯器：
- 左：搜尋 / 類別挑素材，點一下或拖進畫布；中：Konva 畫布（拖曳、等比縮放、旋轉、置中吸附）；右：圖層 / 畫布設定
- 素材圖層：翻轉、不透明度、陰影、置中並填滿、圖層順序、複製；文字圖層：字型（微軟正黑體 / 新細明體 / Arial）、字級、顏色、粗細、對齊、行距
- 背景：純色 / 雙色漸層 / 透明（PNG 保留透明，JPEG 填白）
- 版型（每種尺寸 2 個）：置中商品、商品＋底部文字；大圖＋標題、雙素材；左文右圖、左圖右文
- 換尺寸時圖層等比例換算；復原 / 重做 50 步；自動儲存；快捷鍵 Del、方向鍵（Shift ×10）、Ctrl+Z / Y / D、Esc
- 匯出：瀏覽器依設計尺寸輸出 PNG / JPEG 並下載，同時上傳存檔（`data/exports/`），後端檢查像素尺寸

畫面內容格式（scene JSON）定義在 `backend/services/designs.py`（Pydantic）與 `frontend/src/lib/scene.ts`。

## 線上 Demo 模式（Hugging Face Space）

`VAH_DEMO=1` 時 `python -m backend.api` 改用 `backend/api/demo.py`：每位訪客（瀏覽器）有一份自己的素材庫，
由 `VAH_DATA_DIR`（示範素材庫）複製到 `VAH_SESSIONS_DIR/<id>/`，閒置 60 分鐘後刪除；模型與背景 worker 所有訪客共用。
瀏覽器以隨機 id 識別（API 帶 `X-VAH-Session` header，圖片網址帶 `?s=`），不用 cookie，嵌在 iframe 裡也能用。
上限（人數、每人上傳張數、檔案大小）在 `backend/config.py` 的 `DemoConfig`。

```bash
VAH_DEMO=1 VAH_DATA_DIR=demo-data conda run -n vah python -m backend.api   # 先用 library ingest + seed_library 建好 demo-data
```

部署：[deploy/hf-space/](deploy/hf-space/) 的 `Dockerfile` 與 `README.md` 上傳到 Docker Space 即可；建置時從 GitHub 的 tag
下載程式、從 Release 下載模型，並用 `tests/images` 建好示範素材庫。

## 推論層結構

```
InferenceBackend (base.py)        ← 所有推論唯一入口；只回傳原始 tensor
 ├─ OnnxBackend (onnx_backend.py)  CUDA → CPU fallback
 └─ HailoBackend (hailo_backend.py) stub
postprocess.py                     letterbox / decode / NMS / mask 合成（主機端，純 numpy）
segmenter.py                       YoloSegmenter：前處理 → backend → 後處理 → Detection（YOLO-seg / FastSAM 共用）
masks.py                           RegionMask：遮罩只存 bbox 範圍（12 MP 照片不存整張）
services/merge.py                  過濾、融合、去重 → Candidate
services/refine.py                 Refiner 介面；GuidedFilterRefiner（GrabCut + guided filter + 羽化）
services/export_png.py             邊緣顏色校正、RGBA 裁切
services/capture.py                CapturePipeline.propose() / extract_assets()
services/library.py                AssetLibrary：add_photo / candidates / confirm / discard / delete
db/models.py, db/repo.py           schema、連線、遷移；純 sqlite3 repository
db/vector_index.py                 VectorIndex：SqliteVecIndex（vec0）/ NumpyIndex（備案、測試基準）
services/embedding.py              ClipImageEncoder（走影像 backend）、TextEncoder（永遠 CPU：英文 / 多語）
services/vocab.py, tagging.py      詞彙表；zero-shot 類別、objectness、顏色、材質風格、情境規則與場景
services/search.py                 中文查詢 → 詞彙比對 + 多語向量 → 重排序（附命中原因）
services/queue.py, api/app.py      單一背景 worker；FastAPI
api/demo.py                        線上 Demo：每位訪客一份素材庫，共用模型與 worker
services/designs.py                設計稿 / 匯出（scene 驗證、尺寸檢查）
services/thumbs.py                 縮圖：素材 512 WebP、畫布用 2048 PNG、照片 1600、候選 256
frontend/src/lib/scene.ts          畫布編輯的純函式（版型、換尺寸、圖層、復原）
frontend/src/components/editor/    Konva 畫布、素材挑選、屬性面板
```

## 授權

本專案以 [GNU AGPL-3.0](LICENSE) 釋出。使用的模型各有授權（模型檔不在 git 裡，見 Releases）：

| 模型 | 來源 | 授權 |
|---|---|---|
| yolov8s-seg、FastSAM-s | Ultralytics（`tools/export_models.py` 匯出） | AGPL-3.0 |
| CLIP ViT-B/32（影像 / 英文文字） | `openai/clip-vit-base-patch32` | MIT |
| 多語文字編碼器 | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | Apache-2.0 |

`tests/images/` 的 10 張照片由團隊拍攝，已移除 EXIF 等中繼資料，僅供測試與示範。
