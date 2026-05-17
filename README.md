# AI 旅平險專員：本地端 RAG 文件檢索系統

系統透過 PDF 條款解析、中文 Embedding、向量檢索與本地端 Ollama LLM，讓使用者能針對旅平險、登山險、海域活動險與《保險法》提出問題，並取得具備文件來源、頁碼與條款依據的回答。

---

## 一、專案目標

保險條款通常包含大量法律文字、理賠限制、除外責任、表格與數字條件。  
若只使用一般大型語言模型回答，容易產生幻覺或引用不存在的條款。

因此本系統採用 **RAG（Retrieval-Augmented Generation）** 架構：

```text
PDF 條款資料
→ PDF 文字解析
→ 條款分塊 Chunking
→ 中文 Embedding 向量化
→ Hybrid Search 檢索
→ Ollama llama3.1:8b 回答
→ 顯示文件、頁碼、條款依據
```
---

## 二、資料來源

本專案資料來源分為兩類：

### 1. 泰安產物保險文件下載專區

資料來源網站：

```text
https://accessibility.taian.com.tw/download
```

該網站提供各險種商品簡介、要保書、條款及各式申請書下載。本專案使用其中與健康暨傷害保險、旅遊保險相關的保單條款 PDF，例如：

- 海外旅行綜合保險保單條款
- 國內旅行綜合保險保單條款
- 旅行平安保險（標準型）保單條款
- 旅行保障綜合保險條款
- 旅行泡泡綜合保險保單條款
- 海域活動綜合保險保單條款
- 登山綜合保險保單條款

### 2. 全國法規資料庫：保險法

資料來源網站：

```text
https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0390002
```

本專案同時加入《保險法》作為法規背景資料，讓系統能查詢保險契約、保險人、要保人、被保險人、受益人等基本法律定義。

---

## 三、使用技術

| 類別 | 技術 |
|---|---|
| PDF 解析 | PyMuPDF |
| 中文文字處理 | 正規化、條號偵測、Chunking |
| 中文 Embedding | BAAI/bge-m3 |
| 關鍵字檢索 | Jieba + BM25 |
| 向量檢索 | Sentence-Transformers + NumPy |
| 可選重排序 | BAAI/bge-reranker-v2-m3 |
| 本地 LLM | Ollama llama3.1:8b |
| Web 介面 | Streamlit |
| 驗證分析 | evaluate.py 測試題集 |

---

## 四、系統特色

### 1. 僅根據 PDF 文件回答

系統 Prompt 限制模型只能依據 Context 回答，不允許使用外部保險知識。  
若檢索內容不足，系統應回覆「目前無法提供該資訊」，避免模型自行猜測。

### 2. 保留文件來源與頁碼

每個 chunk 都會保存：

```text
文件名稱
頁碼
條款名稱
chunk_id
原文內容
```

回答時會輸出檢索來源，方便回查原始 PDF。

### 3. 支援中文保險條款

程式會處理保險 PDF 中常見的異體字與特殊字，例如：

```text
旅 → 旅
行 → 行
金 → 金
契 → 契
不 → 不
臺 → 台
```

這可以提升中文檢索穩定性。

### 4. 混合檢索

本系統同時使用：

```text
Dense Embedding 語意檢索
+
BM25 關鍵字檢索
```

語意檢索適合處理「意思相近但用詞不同」的問題；BM25 適合處理條號、專有名詞、理賠項目與明確關鍵字。

### 5. 可選 Reranker

若需要更高精度，可啟用：

```bash
--rerank
```

系統會使用 BGE Reranker 對候選條款重新排序。  
缺點是速度較慢，但回答依據通常更準確。

---

## 五、專案結構

```text
project/
├── data/
│   ├── 海外旅行綜合保險保單條款-11504版.pdf
│   ├── 國內旅行綜合保險保單條款-11306版.pdf
│   ├── 旅行平安保險(標準型)保單條款-11408版.pdf
│   ├── 旅行保障綜合保險條款.pdf
│   ├── 旅行泡泡綜合保險保單條款.pdf
│   ├── 海域活動綜合保險保單條款-11103版.pdf
│   ├── 登山綜合保險保單條款-11103版.pdf
│   └── 保險法.pdf
│
├── index/
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   └── meta.json
│
├── rag_core.py
├── rag_app.py
├── streamlit_app.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 六、安裝方式

建議使用 Conda 建立乾淨環境：

```bash
conda create -n travelrag python=3.10 -y
conda activate travelrag
```

安裝套件：

```bash
pip install -r requirements.txt
```

---

## 七、安裝 Ollama 與模型

請先安裝 Ollama，並下載本專案使用的本地模型：

```bash
ollama pull llama3.1:8b
```

啟動 Ollama 服務：

```bash
ollama serve
```

若已經在背景執行 Ollama，則不需要重複執行。

---

## 八、建立 PDF 向量索引

將所有 PDF 放入 `data/` 後，執行：

```bash
python rag_app.py --build --data data --index index --embedding BAAI/bge-m3
```

系統會執行：

```text
1. 解析 PDF
2. 依條款與長度切分 chunk
3. 使用 BGE-M3 建立中文語意向量
4. 儲存 chunks.jsonl、embeddings.npy、meta.json
```

若有 NVIDIA GPU，可指定 CUDA：

```bash
python rag_app.py --build --data data --index index --embedding BAAI/bge-m3 --device cuda
```

---

## 九、命令列問答

單次提問：

```bash
python rag_app.py --ask "班機延誤5小時可以理賠嗎？" --index index --model llama3.1:8b --top-k 6
```

啟用 Reranker：

```bash
python rag_app.py --ask "海外突發疾病是否可以理賠？" --index index --model llama3.1:8b --top-k 6 --rerank
```

互動模式：

```bash
python rag_app.py --interactive --index index --model llama3.1:8b
```

---

## 十、Streamlit 網頁介面

啟動 Web Demo：

```bash
streamlit run streamlit_app.py
```

啟動後可在網頁中：

1. 設定 PDF 資料夾
2. 設定索引資料夾
3. 選擇 Embedding 模型
4. 選擇 Ollama 模型
5. 調整 Top-K
6. 選擇是否啟用 Reranker
7. 輸入問題並查看回答與檢索來源

---



## 十一、目前限制

1. PDF 若為掃描圖片，PyMuPDF 可能無法直接抽取文字，需要 OCR。
2. Llama 3.1 8B 對中文法律條款的數字推理仍可能出錯。
3. 表格型條款可能因 PDF 排版造成順序錯亂。
4. Reranker 雖然可提升精度，但會增加執行時間。
5. 本系統僅作為課程專案展示，不構成正式保險理賠或法律建議。

---

## 十二、未來改進方向

1. 加入 OCR 支援掃描型 PDF。
2. 加入規則式數字判斷器，處理「以上、以下、未滿、超過、不得超過」。
3. 加入條款層級索引，讓系統以完整條文而非固定長度 chunk 回答。
4. 加入測試集與人工標註正解。
5. 加入回答可信度分數。
6. 加入多模型比較，例如 llama3.1:8b、qwen2.5:7b、gemma3 等。
7. 加入 GitHub Actions 自動測試。

---

## 免責聲明

本專案僅供課程期末報告與技術展示使用。  
系統回答內容來自上傳 PDF 與檢索結果，不代表保險公司正式理賠結論。  
實際投保、理賠與法律解釋仍應以保險公司正式文件、主管機關公告與專業人員說明為準。