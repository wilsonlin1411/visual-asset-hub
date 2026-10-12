# Visual Asset Hub — Demo 規格（待填）

## 1. 開發環境
- OS：Windows 10 家用版 64 位元（10.0.19045）
- Python 版本：系統 3.9.13；專案使用 conda 環境 `vah`（3.11，runtime）與 `vah-export`（3.11，匯出 ONNX 用）
- 有無 NVIDIA GPU：有，NVIDIA GeForce RTX 3060（12 GB VRAM，驅動 560.94，最高支援 CUDA 12.6）
- Node 版本：22（conda 環境 `vah-web`，conda-forge nodejs；不需系統安裝）
CPU：Intel Core i5-7500 @ 3.40GHz（4 核）
記憶體：16 GB

## 2. Demo 範圍
- [x] 上傳照片
- [x] 自動分割 + 使用者勾選確認
- [x] 輸出透明 PNG
- [x] 自動標籤 + 向量（含手動修改類別 / 標籤）
- [x] 中文自然語言搜尋
- [x] 畫布（匯出尺寸：商品主圖 800×800 / IG 貼文 1080×1350 / Banner 1640×856；每種尺寸 2 個版型）
- 不做：VLM、帳號系統、PostgreSQL

## 3. 標籤詞彙表
> 依 tests/images 的照片與「咖啡廳 / 商品 / 桌面」情境擬定。英文為送進 CLIP 的 prompt 用詞；
> 「同義詞」供中文查詢比對（例如查「威士忌」→ 酒瓶）。程式讀取的版本在 `backend/vocab/vocabulary.toml`，
> 兩者需同步修改；改完執行 `python -m backend.cli.library retag` 重新標註全部素材。

### 3.1 類別（44 個）
| 群組 | 類別 | CLIP 英文 | 中文同義詞 | 對應 COCO（YOLO） |
|---|---|---|---|---|
| 餐飲 | 甜點 | dessert, pudding, cake | 布丁、蛋糕、點心 | cake, donut |
| 餐飲 | 麵包 | bread, pastry, bun | 菠蘿麵包、可頌、吐司 | sandwich |
| 餐飲 | 披薩 | pizza | pizza | pizza |
| 餐飲 | 肉類主餐 | steak, meat dish | 牛排、豬排、主菜 | |
| 餐飲 | 飯類 | rice bowl, donburi | 丼飯、蓋飯、便當 | |
| 餐飲 | 蔬果 | vegetables, fruit | 水果、蔬菜、沙拉 | banana, apple, orange, broccoli, carrot |
| 餐飲 | 咖啡杯 | coffee cup, mug, teacup | 馬克杯、咖啡、拿鐵、茶杯 | cup |
| 餐飲 | 酒瓶 | liquor bottle, whisky bottle, wine bottle | 威士忌、紅酒、酒 | bottle, wine glass |
| 餐飲 | 罐裝飲料 | beverage can, beer can | 啤酒、汽水、易開罐 | |
| 餐飲 | 隨行杯 | tumbler, insulated bottle | 保溫杯、水壺、冰霸杯 | |
| 餐飲 | 碗 | bowl | 碗公 | bowl |
| 餐飲 | 盤子 | plate | 餐盤、碟子 | |
| 餐飲 | 餐具 | spoon, fork, knife, chopsticks | 湯匙、叉子、刀子、筷子 | spoon, fork, knife |
| 3C | 筆電 | laptop, tablet | 筆記型電腦、MacBook、平板、電腦 | laptop |
| 3C | 手機 | smartphone | 手機、iPhone | cell phone |
| 3C | 手機殼 | phone case | 保護殼、手機配件 | |
| 3C | 滑鼠 | computer mouse | 無線滑鼠 | mouse |
| 3C | 鍵盤 | keyboard | 機械鍵盤 | keyboard |
| 3C | 螢幕 | computer monitor | 顯示器 | tv |
| 3C | 滑鼠墊 | mouse pad | 桌墊 | |
| 3C | 耳機 | headphones, earbuds | 藍牙耳機 | |
| 3C | 相機 | camera | 單眼、相機 | |
| 動物 | 寵物 | pet cat, pet dog, hamster, pet rabbit | 貓、狗、毛小孩、倉鼠 | cat, dog |
| 動物 | 野生動物 | wild animal, bird, deer | 鳥、動物園、野生動物 | bird, horse, sheep, cow, elephant, bear, zebra, giraffe |
| 生活 | 提袋 | tote bag, handbag | 帆布袋、包包、手提袋 | handbag, backpack |
| 生活 | 飾品 | ring, hair clip, necklace, earrings, bracelet, wristwatch | 戒指、髮夾、項鍊、耳環、手環、手錶 | |
| 生活 | 個人護理用品 | nail clippers, comb, razor, toothbrush, hair dryer | 指甲剪、梳子、刮鬍刀、牙刷、吹風機 | toothbrush, hair drier |
| 生活 | 安全帽 | motorcycle helmet | 全罩、頭盔 | |
| 生活 | 鑰匙 | keys | 鑰匙圈 | |
| 生活 | 公仔 | figurine, toy figure | 玩偶、模型、咖波 | teddy bear |
| 生活 | 燈具 | lamp, night light | 夜燈、檯燈 | |
| 生活 | 包裝盒 | product box, packaging | 紙盒、外盒、禮盒 | |
| 生活 | 藥品 | medicine bottle, medicine box | 感冒藥、維他命、保健食品 | |
| 生活 | 書本 | book, notebook | 筆記本、雜誌 | book |
| 生活 | 文具 | pen, stationery | 筆、便利貼 | scissors |
| 生活 | 植物 | potted plant, flowers | 盆栽、花 | potted plant, vase |
| 生活 | 雕像 | statue, sculpture | 雕塑、石像 | |
| 生活 | 陶瓷器 | ceramic pot, ceramic figure | 陶鍋、瓷器、擺飾 | |
| 生活 | 時鐘 | clock | 掛鐘 | clock |
| 生活 | 眼鏡 | glasses | 墨鏡 | |
| 生活 | 鞋子 | shoes | 球鞋 | |
| 生活 | 服飾 | clothing | 衣服、T 恤 | tie |
| 其他 | 人物（搜尋時預設排除） | person, hand | 手、人 | person |
| 其他 | 其他 | object | | （其餘 COCO 類別） |

### 3.2 屬性
- **顏色**（由像素計算，不靠 CLIP）：黑、白、灰、紅、橙、黃、綠、藍、紫、粉紅、棕、米色；取占比 ≥ 25% 的前 1–2 個
- **材質**（CLIP，可多選）：塑膠 plastic、金屬 metal、玻璃 glass、陶瓷 ceramic、木頭 wood、布料 fabric、皮革 leather、紙 paper/cardboard、透明 transparent
- **風格**（CLIP，可多選）：可愛 cute/cartoon、簡約 minimalist、復古 vintage、奢華 luxury、繽紛 colorful

### 3.3 情境（14 個）
| 情境 | CLIP 場景 prompt（整張照片） | 中文同義詞 |
|---|---|---|
| 遠距工作 | working remotely on a laptop | 在家工作、WFH、咖啡廳工作 |
| 咖啡廳 | a cafe table | 咖啡店、下午茶店 |
| 辦公桌面 | an office desk setup | 桌面、工作桌、電腦周邊 |
| 用餐 | a meal at a restaurant | 吃飯、晚餐、午餐、餐廳 |
| 甜點時光 | afternoon tea with desserts | 下午茶、點心 |
| 小酌 | drinks at a bar | 喝酒、下班、微醺、酒吧 |
| 居家 | at home | 家裡、房間 |
| 戶外 | outdoors | 戶外、街上 |
| 通勤 | commuting by scooter | 騎車、機車、上班路上 |
| 購物 | a retail store display | 逛街、商店、展示櫃 |
| 開箱 | unboxing a new product | 新品、拆箱 |
| 收藏 | collectible toys on display | 收藏品、公仔展示 |
| 保健 | taking medicine | 生病、感冒、健康 |
| 旅行 | sightseeing at a museum or landmark | 博物館、展覽、景點、出遊 |

### 3.4 情境推論規則（同一張照片的物件組合 → 情境）
| 條件 | 情境 |
|---|---|
| 筆電 + 咖啡杯 | 遠距工作、咖啡廳 |
| 筆電 + 滑鼠 | 遠距工作 |
| 咖啡杯 + （甜點 或 麵包） | 咖啡廳、甜點時光 |
| （滑鼠 或 鍵盤 或 螢幕 或 滑鼠墊）任兩個 | 辦公桌面 |
| 甜點 | 甜點時光 |
| （肉類主餐 或 披薩 或 飯類）；或（盤子 / 碗 + 餐具） | 用餐 |
| 酒瓶 或 罐裝飲料（啤酒） | 小酌 |
| 包裝盒 + （滑鼠 或 手機 或 耳機 或 隨行杯 或 公仔） | 開箱 |
| 公仔 | 收藏 |
| 安全帽 | 通勤 |
| 藥品 | 保健 |
| 雕像 | 旅行 |
| 寵物 | 居家 |
| 野生動物 | 戶外、旅行 |
- 規則先產生情境（來源 `rule`），再用整張照片對 3.3 的場景 prompt 做 CLIP 比對補充（來源 `clip-scene`，分數低於門檻不加）

## 4. 資料結構
SQLite（`data/vah.db`）+ sqlite-vec；schema 定義在 `backend/db/models.py`。路徑皆相對於 `data/`。
- photos：id, path, original_name, sha256（重複上傳去重）, width, height, created_at
- candidates：id, photo_id, idx, class_id, class_name, score, label_model, mask_model, bbox, area, mask_path, objectness（第 4 階段 CLIP 填入）, status（pending / confirmed / discarded）, created_at
- assets：id, photo_id, candidate_id, png_path, bbox, origin（PNG 左上角在原圖座標）, width, height, category, source_model（= mask_model）, label_model, mask_model, score, created_at
- asset_tags：asset_id, kind（attribute / context）, value, score, source — 取代 SPEC 原本 assets 上的 tags / context 欄位，方便以 SQL 篩選
- embeddings：asset_id, embedding_model, backend, dim, vector(512, BLOB) — 唯一鍵 (asset_id, embedding_model, backend)，ONNX 與 Hailo 向量可並存
- vec_embeddings：sqlite-vec vec0 索引（cosine），可由 embeddings 重建
- 檔案：`photos/{id}.jpg`、`candidates/{photo_id}/{idx}.png`（1-bit 遮罩）、`assets/{id}.png`

## 5. API
`python -m backend.api` → 前端 http://127.0.0.1:8000/ 、API 文件 http://127.0.0.1:8000/api/docs
（FastAPI，定義在 `backend/api/app.py`；所有端點在 `/api` 底下，`frontend/dist` 存在時同一個 port 提供前端）
- POST /api/photos — 上傳（multipart），立即回 202 + photo id，背景佇列處理（status: processing → ready / failed）；同檔案重複上傳回傳既有照片
- GET  /api/photos（含候選 / 素材數）、/api/photos/{id}（含照片情境）；DELETE /api/photos/{id}
- GET  /api/photos/{id}/image（原圖）、/display（長邊 1600 顯示用）
- GET  /api/photos/{id}/candidates?include_hidden= — YOLO 有類別的在前，其餘依 objectness 排序；低於門檻預設隱藏（YOLO 有類別者不隱藏）
- GET  /api/photos/{id}/candidates/{idx}/mask（1-bit 遮罩）、/preview（256 px 預覽）、/overlay（顯示尺寸的 alpha 遮罩，前端疊色用）
- POST /api/assets/confirm {photo_id, idxs} — 精修、去背、CLIP 向量、類別與標籤、縮圖
- POST /api/candidates/discard {photo_id, idxs}
- GET  /api/assets?category=&tag=&photo_id=、/api/assets/{id}、/png（原尺寸透明 PNG）、/thumb（512 px WebP）；DELETE /api/assets/{id}
- PATCH /api/assets/{id} {category} — 手動設定類別（來源 user，情境規則會採用）
- POST /api/assets/{id}/tags {kind, value}、DELETE /api/assets/{id}/tags?kind=&value= — 手動增刪情境 / 屬性標籤（值須在詞彙表內；重新標註時保留）
- GET  /api/search?q=&k=&category=&tag=&include_people= — 回傳分數與命中原因
- GET  /api/vocabulary、/api/stats
- GET  /api/presets — 三種畫布尺寸
- GET/POST /api/designs、GET/PUT/DELETE /api/designs/{id}、GET /api/designs/{id}/thumb — 設計稿（scene JSON：背景 + 素材 / 文字圖層，座標為設計像素）
- POST /api/designs/{id}/export（multipart：file + format png/jpeg）— 瀏覽器 Konva 輸出的成品，後端檢查像素尺寸 = 設計尺寸後存檔（即原規劃的 POST /export）
- GET  /api/exports?design_id=、/api/exports/{id}/file
- GET  /api/assets/{id}/medium — 長邊 2048 px 透明 PNG（畫布用）

## 6. 驗收標準
- 單張照片處理時間（CPU，onnxruntime，i5-7500）：
  - 產生候選（YOLO-seg + FastSAM + 合併去重）≤ 5 秒／張
  - 每個勾選物件產生透明 PNG（精修 + 去背）≤ 3 秒／個
  - 對應 `backend/config.py` 的 `PROPOSE_BUDGET_MS`、`EXTRACT_PER_ASSET_BUDGET_MS`；`python -m backend.cli.capture tests/images` 會自動比對
- 搜尋測試案例（查詢 → 預期結果；以 tests/images 的檔名指定照片；機器可讀版在 `tests/search_cases.toml`，
  先 `library ingest tests/images` + `tools/seed_library.py` 建立示範素材庫，再以 `python tools/eval_search.py` 評估）：
  - 「辦公桌上的電腦周邊」 → 滑鼠 PXL_20260617_055925417、鍵盤 PXL_20260829_193653800
  - 「適合下班小酌的東西」 → 啤酒罐 PXL_20260829_193653800
  - 「可愛的公仔」 → 兔子 PXL_20260601_143321659、咖波 PXL_20260705_065356505
  - 「甜點」 → 布丁 pudding、菠蘿麵包 PXL_20260516_122439108
  - 「晚餐的主菜」 → 丼飯 PXL_20260819_104454281
  - 「藍色的東西」 → 咖波 PXL_20260705_065356505、安全帽 PXL_20260731_171541198
  - 「帆布提袋」 → PXL_20260630_122217844
  - 「騎車要用的」 → 安全帽 PXL_20260731_171541198
  - 驗收指標：每個查詢前 5 名內出現預期照片的素材（hit@5），通過率 ≥ 80%（目前 8/8）
- 單次搜尋延遲：< 300 ms（目前中位數 22 ms，126 個素材）
- 畫布匯出：像素尺寸與預設完全一致；3 個素材 + 文字的成品匯出 < 2 秒（實測 75–246 ms）
- 測試照片：放在 tests/images/（公開版 10 張，涵蓋商品、食物、公仔、桌面等，已移除 EXIF；> 4096 px 的照片匯入時縮小）
