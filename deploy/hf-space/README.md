---
title: Visual Asset Hub
emoji: 🖼️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: 照片拆成去背素材、中文搜尋、畫布匯出（線上 Demo）
---

# Visual Asset Hub 線上 Demo

> Capture once. Reuse everywhere.

把一張照片拆成多個去背素材（透明 PNG），自動加上類別與情境標籤，用中文自然語言搜尋素材，
再排進商品主圖、IG 貼文、Banner 匯出。

**原始碼、安裝說明與授權：https://github.com/wilsonlin1411/visual-asset-hub**

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

## 這個 Demo 的運作方式

- 每位訪客都有**自己的素材庫**，一開始已放入 10 張示範照片。你上傳的照片、建立的素材與設計只有你看得到。
- 閒置 60 分鐘後自動刪除；Space 重新啟動時也會全部清空。每人最多上傳 10 張、單張 20 MB 以內。
- 請勿上傳含個人資料的照片。
- 主機是 2 核心 CPU，每張照片的處理時間比在一般筆電上慢（約 10 秒以上）。
- 一段時間沒人使用時 Space 會休眠，再次開啟需要等候 1–2 分鐘啟動。

正式產品的設計是在使用者自己的裝置上處理（目標硬體：ASUS UGen300 / Hailo-10H），
這個 Demo 用 ONNX Runtime 在 CPU 上模擬。

## 授權

本專案以 AGPL-3.0 釋出。使用的模型：yolov8s-seg、FastSAM-s（Ultralytics，AGPL-3.0）、
CLIP ViT-B/32（OpenAI，MIT）、clip-ViT-B-32-multilingual-v1（sentence-transformers，Apache-2.0）。
