# 在新電腦安裝 Visual Asset Hub

從 GitHub 取得原始碼，再從 **Releases** 下載模型與建置好的網頁，最後在瀏覽器開 **http://localhost:8000** 使用。

> 模型（約 1.2 GB）與建置好的網頁不在 git 裡，放在本專案 GitHub 頁面的 **Releases**：
> - `vah-models.zip`：5 個 ONNX 模型 + 2 個 tokenizer 資料夾（解壓後是 `models\`）
> - `vah-frontend-dist.zip`：建置好的網頁（解壓後是 `frontend\dist\`）
>
> 不想下載也可以自己建置，見附錄 B。

---

## 快速版（熟悉的人看這段就好）

在 **命令提示字元（CMD）** 依序輸入（`<repo 網址>` 換成 GitHub 頁面綠色 **Code** 按鈕裡的網址）：

```bat
git clone <repo 網址> C:\vah\visual-asset-hub
cd /d C:\vah\visual-asset-hub
rem 先從 Releases 下載 vah-models.zip、vah-frontend-dist.zip 放進這個資料夾
tar -xf vah-models.zip
tar -xf vah-frontend-dist.zip
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m backend.api
```

看到 `Uvicorn running on http://127.0.0.1:8000` 後，用瀏覽器開 **http://localhost:8000**。

---

## 1. 電腦需求

| 項目 | 需求 |
|---|---|
| 作業系統 | Windows 10 / 11 64 位元 |
| 記憶體 | 8 GB 以上（建議 16 GB；三個 CLIP 模型載入後約佔 2 GB） |
| 硬碟空間 | 約 3 GB（模型 zip 約 1 GB、解壓後 1.2 GB、Python 套件約 0.5 GB） |
| 顯示卡 | 不需要（預設用 CPU；有 NVIDIA 顯卡可選擇加速，見第 7 節） |
| 網路 | 只有下載與第一次安裝套件時需要；之後可離線使用 |
| 瀏覽器 | Chrome 或 Edge |

## 2. 需要安裝的軟體

- **Python 3.11**（必要）
- **Git**（選用；沒有 Git 可以改用第 3 節的「Download ZIP」）
- 不需要 Node.js：網頁已經建置好放在 Releases（要自己建置才需要，見附錄 B）

安裝 Python：

1. 到 https://www.python.org/downloads/windows/ 下載 **Python 3.11.x 的 Windows installer (64-bit)**。
2. 安裝時：
   - 勾選 **「Add python.exe to PATH」**
   - 保持勾選 **「py launcher」**（預設已勾）
3. 確認安裝成功：開 CMD 輸入
   ```bat
   py -3.11 --version
   ```
   應顯示 `Python 3.11.x`。

安裝 Git（選用）：到 https://git-scm.com/download/win 下載安裝，選項都用預設即可。

> 已經有 Anaconda / Miniconda 的話也可以用 conda，見第 8 節，不必另外安裝 Python。

## 3. 取得原始碼

**方法 A：用 Git（之後可以用 `git pull` 更新）**

到本專案的 GitHub 頁面，按綠色 **Code** 按鈕複製網址，然後在 CMD 輸入：

```bat
git clone <repo 網址> C:\vah\visual-asset-hub
```

**方法 B：下載 ZIP（不用裝 Git）**

1. GitHub 頁面 → 綠色 **Code** 按鈕 →「**Download ZIP**」。
2. 右鍵 →「**解壓縮全部**」→ 目的地選 `C:\vah`。
3. 解壓出來的資料夾叫 `visual-asset-hub-main`，把它改名成 `visual-asset-hub`。

> 建議路徑不要有中文或空白（例如不要放在「桌面\新資料夾」底下），可避免少數套件出錯。

## 4. 下載模型與網頁

1. 在 GitHub 頁面右側點 **Releases**，打開最新版本。
2. 在 **Assets** 下載 `vah-models.zip` 和 `vah-frontend-dist.zip`，把兩個檔案放進 `C:\vah\visual-asset-hub\`。
3. 在 CMD 解壓縮（Windows 10 / 11 內建 `tar`，會直接解到正確位置）：

```bat
cd /d C:\vah\visual-asset-hub
tar -xf vah-models.zip
tar -xf vah-frontend-dist.zip
del vah-models.zip vah-frontend-dist.zip
```

> 不要用右鍵「解壓縮全部」：它會多包一層 `vah-models\` 資料夾，程式會找不到模型。

完成後，資料夾裡要有這些（缺一不可）：

```
visual-asset-hub\
  backend\                 後端程式
  frontend\dist\           建置好的網頁（index.html 在這裡）
  models\                  5 個 .onnx 模型 + 2 個 tokenizer 資料夾（約 1.2 GB）
  requirements.txt
  docs\                    本文件、操作說明
```

## 5. 第一次安裝（只做一次）

開啟 **CMD**（開始 → 輸入 `cmd` → Enter），逐行輸入：

```bat
cd /d C:\vah\visual-asset-hub
```
進入專案資料夾。

```bat
py -3.11 -m venv .venv
```
建立專案專用的 Python 環境（資料夾 `.venv`，不影響電腦上其他 Python）。

```bat
.venv\Scripts\activate
```
啟用環境，之後提示字元前面會出現 `(.venv)`。

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
安裝套件（onnxruntime、OpenCV、FastAPI、sqlite-vec 等，下載約 200–300 MB，需要幾分鐘）。

## 6. 啟動與使用

```bat
python -m backend.api
```

看到下面這行就代表啟動成功：

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

打開瀏覽器，網址輸入 **http://localhost:8000** 即可使用（操作方式見 [USER_GUIDE.md](USER_GUIDE.md)）。

- **CMD 視窗要一直開著**，關掉視窗伺服器就停了。
- **停止**：在 CMD 視窗按 `Ctrl + C`。
- **第一次上傳照片會比較慢**（約 15 秒，要載入模型），之後每張約 3–4 秒。
- 第一次啟動會自動建立空的素材庫（`data\vah.db`）。

### 想先看到示範內容（選用）

用 `tests\images` 的 10 張測試照片建立示範素材庫（先停止伺服器，約 1–2 分鐘）：

```bat
python -m backend.cli.library ingest tests\images
python tools\seed_library.py
python -m backend.api
```

`seed_library.py` 會模擬使用者勾選每張照片中明顯的物件，完成後素材庫與搜尋頁就有內容可以操作。

### 之後每次開啟

```bat
cd /d C:\vah\visual-asset-hub
.venv\Scripts\activate
python -m backend.api
```

或不啟用環境、一行搞定：

```bat
cd /d C:\vah\visual-asset-hub && .venv\Scripts\python -m backend.api
```

> 想要雙擊就啟動，可以在專案資料夾建立 `start.bat`，內容放上面那一行，再加一行 `pause`。

## 7. 選用設定

| 需求 | 做法 |
|---|---|
| 8000 埠被佔用 | `python -m backend.api --port 8001`，網址改成 http://localhost:8001 |
| 讓同網路的其他電腦連線 | `python -m backend.api --host 0.0.0.0`，其他電腦開 `http://這台電腦的IP:8000`；Windows 防火牆詢問時選「允許」。⚠ 系統沒有帳號登入，只在信任的網路使用 |
| 用 NVIDIA 顯卡加速 | `pip uninstall -y onnxruntime` → `pip install "onnxruntime-gpu[cuda,cudnn]"` → 啟動前 `set VAH_DEVICE=cuda`。兩個 onnxruntime 套件不能同時安裝；CUDA 載入失敗時會自動退回 CPU。（此選項尚未在本專案實測，demo 以 CPU 為準） |
| 帶入原本的素材庫 | 把舊電腦的 `data\` 資料夾整個複製到 `visual-asset-hub\data\`（**複製前先在舊電腦按 Ctrl+C 停止伺服器**）。`data\` 裡有原始照片，不會進 git |
| 更新到最新版 | `git pull`；Release 有新的模型或網頁時，重新做第 4 節 |
| 執行自動測試（確認安裝完整） | `python -m pytest tests -q` |

## 8. 改用 Anaconda / Miniconda（替代第 2、5 節）

開 **Anaconda Prompt**：

```bat
conda create -n vah python=3.11 -y
conda activate vah
cd /d C:\vah\visual-asset-hub
pip install -r requirements.txt
python -m backend.api
```

之後每次：`conda activate vah` → `cd /d C:\vah\visual-asset-hub` → `python -m backend.api`。

## 9. 常見問題

| 訊息 / 狀況 | 原因與處理 |
|---|---|
| `'py' 不是內部或外部命令` | Python 沒裝好或沒加入 PATH。重新安裝並勾選「Add python.exe to PATH」，或把指令中的 `py -3.11` 改成 `python` |
| `'git' 不是內部或外部命令` | 沒有安裝 Git。安裝 Git 後重開 CMD，或改用第 3 節方法 B |
| `pip install` 出現 SSL / 憑證錯誤 | 公司網路的 Proxy 攔截。請 IT 提供 Proxy 設定後用 `pip install --proxy http://主機:埠 -r requirements.txt` |
| `DLL load failed`（onnxruntime 或 cv2） | 安裝 Microsoft Visual C++ 可轉散發套件（x64）：https://aka.ms/vs/17/release/vc_redist.x64.exe ，裝完重開 CMD |
| 網頁一片空白，或 `/` 出現 404 | 缺 `frontend\dist\`：重做第 4 節，或見附錄 B 自行建置 |
| 上傳後狀態變「失敗」，錯誤提到 `model file not found` | 缺 `models\` 或其中的 .onnx：確認是用 `tar -xf` 解壓（第 4 節），或見附錄 B 自行匯出 |
| 啟動訊息出現 `sqlite-vec unavailable` | 可以忽略，系統會改用備援的向量搜尋，功能相同 |
| `Address already in use` / `10048` | 8000 埠被佔用（可能已經開了一個），關掉舊的 CMD 視窗或改用 `--port 8001` |
| 畫布上的文字字型不同 | 畫布使用系統字型；沒有「微軟正黑體」的電腦會用替代字型 |

---

## 附錄 A：發佈 Release（維護者）

模型或前端有更新時，在已經有 `models\` 的電腦打包，上傳到 GitHub Releases。

1. 先建置最新的網頁（需要 Node.js 22，見附錄 B）。
2. 在專案資料夾執行（zip 輸出到上一層，避免放進 repo）：

```bat
cd /d C:\vah\visual-asset-hub
tar -a -c -f ..\vah-models.zip --exclude=*.pt models
tar -a -c -f ..\vah-frontend-dist.zip frontend/dist
```

3. GitHub 頁面 → **Releases** →「**Draft a new release**」→ 建立新 tag（例如 `v1.0.0`）→ 把兩個 zip 拖進 **Attach binaries** → **Publish release**。單一檔案上限 2 GB。

| 內容 | 放在哪裡 | 說明 |
|---|---|---|
| 原始碼、`docs\`、`tests\`（含 10 張測試照片） | git | `git clone` 就有 |
| `models\`（.onnx + tokenizer） | Release：`vah-models.zip` | 太大不適合放 git；`models\*.pt` 是匯出用的原始權重，不需要 |
| `frontend\dist\` | Release：`vah-frontend-dist.zip` | 建置產物，讓使用者不用裝 Node.js |
| `data\` | 不公開 | 個人素材庫（含原始照片） |

> 模型授權：YOLOv8-seg 與 FastSAM-s 權重來自 Ultralytics（AGPL-3.0）；CLIP ViT-B/32 為 MIT；
> clip-ViT-B-32-multilingual-v1 為 Apache-2.0。本專案以 AGPL-3.0 釋出，見 README。

## 附錄 B：自行建置（不下載 Release）

**建置網頁 `frontend\dist\`**：需要 Node.js 22 LTS（https://nodejs.org/）。

```bat
cd /d C:\vah\visual-asset-hub\frontend
npm ci
npm run build
```

**匯出模型 `models\`**：需要另一個「匯出用」環境下載並轉換模型（下載約 1.3 GB、安裝約 3 GB，耗時較久，能用 Release 就用 Release）。

```bat
cd /d C:\vah\visual-asset-hub
py -3.11 -m venv .venv-export
.venv-export\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-export.txt
python tools\export_models.py
python tools\export_clip.py
deactivate
```

完成後 `models\` 會有 `yolov8s-seg.onnx`、`fastsam-s.onnx`、`clip-vit-b32-image.onnx`、`clip-vit-b32-text.onnx`、`clip-multilingual-text.onnx` 與兩個 tokenizer 資料夾，再回到第 6 節啟動即可。
